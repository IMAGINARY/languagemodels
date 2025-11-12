import { AutoTokenizer, AutoModel, Tensor } from "@huggingface/transformers";

function headMean(t /* [1,H,T,T] */) {
  const [B, H, T, S] = t.dims;
  if (B !== 1 || T !== S) throw new Error("Unexpected attention shape");
  const out = Array.from({ length: T }, () => Array(S).fill(0));
  const data = t.data; // Float32Array
  for (let h = 0; h < H; h++) {
    for (let i = 0; i < T; i++) {
      const base = (h * T + i) * S;
      for (let j = 0; j < S; j++) out[i][j] += data[base + j];
    }
  }
  const invH = 1 / H;
  for (let i = 0; i < T; i++) for (let j = 0; j < S; j++) out[i][j] *= invH;
  return out;
}

function headSlice(t /* [1,H,T,T] */, hSel /* int */) {
  const [B, H, T, S] = t.dims;
  if (B !== 1 || T !== S) throw new Error("Unexpected attention shape");
  if (hSel < 0 || hSel >= H)
    throw new Error(`Head index out of range 0..${H - 1}`);
  const out = Array.from({ length: T }, () => Array(S).fill(0));
  const startH = hSel * T * S;
  const data = t.data;
  for (let i = 0; i < T; i++) {
    const base = startH + i * S;
    for (let j = 0; j < S; j++) out[i][j] = data[base + j];
  }
  return out;
}

function layerMean(mats /* Array<[T,T]> */) {
  const L = mats.length;
  const T = mats[0].length,
    S = mats[0][0].length;
  const out = Array.from({ length: T }, () => Array(S).fill(0));
  for (let l = 0; l < L; l++) {
    const M = mats[l];
    for (let i = 0; i < T; i++) {
      const r = M[i];
      for (let j = 0; j < S; j++) out[i][j] += r[j];
    }
  }
  const invL = 1 / L;
  for (let i = 0; i < T; i++) for (let j = 0; j < S; j++) out[i][j] *= invL;
  return out;
}

/**
 * Get attention matrix with selectable layer/head.
 *
 * @param {string} sentence
 * @param {object} opts
 *  - modelId: HF repo with attentions (default: bradynapier MiniLM ONNX)
 *  - layer: index (0-based) or 'last' or 'mean' (default 'last')
 *  - head: index (0-based) or 'mean' (default 'mean')
 */
export async function computeAttention(
  sentence = "The quick brown fox jumps over the lazy dog.",
  {
    modelId = "bradynapier/all_miniLM_L6_v2_with_attentions_onnx",
    layer = "last", // 0..L-1, 'last', or 'mean'
    head = "mean", // 0..H-1, 'mean', or 'none'
  } = {}
) {
  const tokenizer = await AutoTokenizer.from_pretrained(modelId);
  const model = await AutoModel.from_pretrained(modelId, {
    dtype: "fp32",
    quantized: false,
    model_file_name: "model", // ensures onnx/model.onnx
  });

  const enc = await tokenizer(sentence, { return_tensors: "pt" });
  const T = enc.input_ids.dims[1];
  const mask = new Tensor(
    "int64",
    BigInt64Array.from({ length: T }, () => 1n),
    [1, T]
  );

  const out = await model(
    {
      ...enc,
      attention_mask: mask,
    },
    { output_attentions: true, output_hidden_states: true }
  );

  // Collect attentions as array of tensors [1,H,T,T]
  const attentions = Array.isArray(out.attentions)
    ? out.attentions
    : Object.keys(out)
        .filter((k) => /^attention_\d+$/i.test(k))
        .sort(
          (a, b) =>
            parseInt(a.split("_")[1], 10) - parseInt(b.split("_")[1], 10)
        )
        .map((k) => out[k]);

  if (!attentions?.length) {
    if (head === "none") {
      // allow embeddings-only flows
    } else {
      throw new Error("No attentions returned by this graph.");
    }
  }
  const L = attentions.length;
  const H = attentions?.[0]?.dims?.[1] || 0;

  // Compute per-layer [T,T] by either head mean or single head
  let attention;
  if (attentions?.length && head !== "none") {
    const perLayer = attentions.map((t) =>
      head === "mean" ? headMean(t) : headSlice(t, head)
    );
    if (layer === "mean") {
      attention = layerMean(perLayer);
    } else if (layer === "last") {
      attention = perLayer[L - 1];
    } else {
      const li = Number(layer);
      if (!Number.isInteger(li) || li < 0 || li >= L)
        throw new Error(`Layer index out of range 0..${L - 1}`);
      attention = perLayer[li];
    }
  }

  // Try to provide token embeddings (last_hidden_state) for PCA/visuals
  let embeddings = null; // Array of length T, each Float32Array(hidden)
  const lhs = out.last_hidden_state || (out.hidden_states && out.hidden_states[out.hidden_states.length - 1]);
  if (lhs && lhs.data && lhs.dims?.length === 3 && lhs.dims[0] === 1) {
    const T = lhs.dims[1];
    const D = lhs.dims[2];
    embeddings = [];
    for (let i = 0; i < T; i++) {
      const start = i * D;
      embeddings.push(new Float32Array(lhs.data.slice(start, start + D)));
    }
  }

  // ids -> readable token strings (prefer tokenizer conversion; fallback aligns specials)
  const ids = Array.from(enc.input_ids.data);
  let tokens;
  if (typeof tokenizer.convert_ids_to_tokens === "function") {
    tokens = tokenizer.convert_ids_to_tokens(ids);
  } else {
    // Fallback: build from raw tokenize() and add specials if present
    const base = await tokenizer.tokenize(sentence);
    const T = enc.input_ids.dims?.[1] ?? ids.length;
    if (T === base.length + 2) {
      tokens = ["[CLS]", ...base, "[SEP]"];
    } else if (T === base.length + 1) {
      tokens = ["[CLS]", ...base];
    } else if (T === base.length) {
      tokens = base;
    } else {
      tokens = Array.from({ length: T }, (_, i) => `#${i + 1}`);
    }
  }

  return { tokens, attention, numHeads: H, embeddings };
}

// Example (no UI yet):
// const { tokens, attention } = await getAttention("A small red cat sat on the mat.", { layer: 'last', head: 3 });
// console.table(attention.map(r => r.map(v => +v.toFixed(4))));
