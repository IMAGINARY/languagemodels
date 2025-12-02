// similar_tokens.ts
import * as ort from "onnxruntime-web";
import { AutoTokenizer, env as txEnv } from "@huggingface/transformers";

// (Optional) prefer WebGPU, fallback to WASM
const epPromise: Promise<ort.InferenceSession.SessionOptions> = (async () => {
  try {
    const webgpu = await ort.env.webgpu;
    if (webgpu?.adapterInfo) {
      return { executionProviders: ["webgpu", "wasm"] };
    }
  } catch (err) {
    console.warn("WebGPU probe failed; falling back to WASM", err);
  }
  return { executionProviders: ["wasm"] };
})();

const ONNX_URL = new URL("../models/minilm_token_knn.onnx", import.meta.url)
  .href; // host this file

// 1) Prepare tokenizer (the Xenova mirror)
txEnv.allowLocalModels = false; // typical default
const TOKENIZER_ID = "Xenova/all-MiniLM-L6-v2";
const tokenizerPromise = AutoTokenizer.from_pretrained(TOKENIZER_ID);

// 2) Prepare ONNX session
const sessionPromise = epPromise.then((ep) =>
  ort.InferenceSession.create(ONNX_URL, ep)
);

// Cache for any additional KNN sessions keyed by their URL
const sessionCache = new Map<string, Promise<ort.InferenceSession>>();

// 3) Utils
function toInt64Tensor(value: number) {
  const data = new BigInt64Array([BigInt(value)]);
  return new ort.Tensor("int64", data, [1]); // shape [1]
}

function toFloatTensor(vec: Float32Array | number[]) {
  const arr = vec instanceof Float32Array ? vec : Float32Array.from(vec);
  return new ort.Tensor("float32", arr, [1, arr.length]); // shape [1, D]
}

type SimilarToken = { id: number; token: string; score: number };

function idToToken(
  tokenizer: Awaited<typeof tokenizerPromise>,
  id: number | bigint
) {
  if (typeof tokenizer.convert_ids_to_tokens === "function") {
    // convert_ids_to_tokens expects a list; take the first result
    return tokenizer.convert_ids_to_tokens([Number(id)])[0];
  }
  // Fallback: decode the single ID
  return tokenizer.decode([Number(id)], { skip_special_tokens: false });
}

export async function mostSimilarTokensToToken(
  tokenString: string,
  k = 5
): Promise<SimilarToken[]> {
  if (k !== 5) {
    // The exported graph bakes in K=5. Re-export if you want a different K.
    console.warn("This ONNX was exported with TopK=5; ignoring custom k.");
  }

  const [tokenizer, session] = await Promise.all([
    tokenizerPromise,
    sessionPromise,
  ]);

  // Tokenize without special tokens; ensure it is exactly one token
  const ids = tokenizer.encode(tokenString, { add_special_tokens: false });
  if (ids.length !== 1) {
    throw new Error(
      `Input "${tokenString}" splits into ${ids.length} tokens: ` +
        `[${ids.map((i) => idToToken(tokenizer, i)).join(", ")}]. ` +
        `Please provide a *single* token.`
    );
  }

  const tokenId = ids[0];

  // Run ONNX: feed token_id -> get top_indices, top_scores
  const feeds: Record<string, ort.Tensor> = {
    token_id: toInt64Tensor(tokenId),
  };
  const results = await session.run(feeds);
  const topIdx = results.top_indices.data as BigInt64Array; // [5]
  const topScores = results.top_scores.data as Float32Array; // [5]

  // Map back to tokens
  const out: SimilarToken[] = Array.from({ length: topIdx.length }, (_, i) => {
    const id = Number(topIdx[i]);
    const token = idToToken(tokenizer, id);
    const score = topScores[i];
    return { id, token, score };
  });

  return out;
}

/**
 * Find nearest tokens for an arbitrary embedding vector.
 * Requires an ONNX exported with input `query_emb` (see prep/onnx/export_minilm_token_knn.py).
 */
export async function mostSimilarTokensToVector(
  queryVector: Float32Array | number[],
  opts?: { k?: number; onnxUrl?: string }
): Promise<SimilarToken[]> {
  const k = opts?.k ?? 5;
  if (k !== 5) {
    console.warn("This ONNX was exported with TopK=5; ignoring custom k.");
  }

  const onnxUrl = opts?.onnxUrl;
  if (!onnxUrl) {
    throw new Error(
      "Provide the ONNX URL/path for the vector KNN model (e.g. ../models/minilm_vector_knn.onnx)."
    );
  }

  let vecSession = sessionCache.get(onnxUrl);
  if (!vecSession) {
    vecSession = epPromise.then((ep) =>
      ort.InferenceSession.create(onnxUrl, ep)
    );
    sessionCache.set(onnxUrl, vecSession);
  }

  const [tokenizer, session] = await Promise.all([
    tokenizerPromise,
    vecSession,
  ]);

  const feeds: Record<string, ort.Tensor> = {
    query_emb: toFloatTensor(queryVector),
  };

  const results = await session.run(feeds);
  const topIdx = results.top_indices.data as BigInt64Array; // [5]
  const topScores = results.top_scores.data as Float32Array; // [5]

  const out: SimilarToken[] = Array.from({ length: topIdx.length }, (_, i) => {
    const id = Number(topIdx[i]);
    const token = idToToken(tokenizer, id);
    const score = topScores[i];
    return { id, token, score };
  });

  return out;
}
