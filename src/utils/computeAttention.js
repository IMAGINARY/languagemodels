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
    layer = "last", // 0..L-1, 'last', or 'mean' (mean = average across layers)
    head = "mean", // 0..H-1 or 'mean'
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

  const out = await model({
    ...enc,
    attention_mask: mask,
    output_attentions: true,
  });

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

  if (!attentions?.length)
    throw new Error("No attentions returned by this graph.");
  const L = attentions.length;

  // Compute per-layer [T,T] by either head mean or single head
  const perLayer = attentions.map((t) =>
    head === "mean" ? headMean(t) : headSlice(t, head)
  );

  // Pick layer: last, specific index, or mean across layers
  let attention;
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

  // ids -> tokens
  const ids = Array.from(enc.input_ids.data);
  const tokens = tokenizer.convert_ids_to_tokens
    ? tokenizer.convert_ids_to_tokens(ids)
    : ids.map(String);

  return { tokens, attention };
}

// Example (no UI yet):
// const { tokens, attention } = await getAttention("A small red cat sat on the mat.", { layer: 'last', head: 3 });
// console.table(attention.map(r => r.map(v => +v.toFixed(4))));
