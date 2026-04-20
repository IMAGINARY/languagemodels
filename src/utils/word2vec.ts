import * as ort from "onnxruntime-web";
import vocab from "../models/word2vec_vocab.json";

const EMBED_ONNX_URL = new URL("../models/word2vec_embed.onnx", import.meta.url)
  .href;
const VECTOR_KNN_URL = new URL(
  "../models/word2vec_vector_knn.onnx",
  import.meta.url
).href;

const tokenToId = new Map(vocab.map((token, index) => [token, index]));
let embedSession: ort.InferenceSession | null = null;
let knnSession: ort.InferenceSession | null = null;
const EPS = 1e-8;

const epPromise: Promise<ort.InferenceSession.SessionOptions> = (async () => {
  try {
    const webgpu = await ort.env.webgpu;
    if (webgpu?.adapterInfo) {
      return { executionProviders: ["webgpu", "wasm"] };
    }
  } catch (err) {
    console.warn("WebGPU probe failed; using WASM", err);
  }
  return { executionProviders: ["wasm"] };
})();

function createRunLock() {
  let chain = Promise.resolve();
  return <T>(task: () => Promise<T>) => {
    const next = chain.then(task, task);
    chain = next.catch(() => {});
    return next;
  };
}

const runEmbedLocked = createRunLock();
const runKnnLocked = createRunLock();

function dot(a: Float32Array | number[], b: Float32Array | number[]) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += a[i] * b[i];
  return sum;
}

function norm(a: Float32Array | number[]) {
  return Math.sqrt(dot(a, a));
}

function normalize(a: Float32Array | number[]) {
  const length = norm(a);
  if (length < EPS) return null;
  const out = new Float32Array(a.length);
  const inv = 1 / length;
  for (let i = 0; i < a.length; i++) out[i] = a[i] * inv;
  return out;
}

function toInt64Tensor(value: number) {
  return new ort.Tensor("int64", new BigInt64Array([BigInt(value)]), [1]);
}

function toFloatTensor(vec: Float32Array | number[]) {
  const arr = vec instanceof Float32Array ? vec : Float32Array.from(vec);
  return new ort.Tensor("float32", arr, [1, arr.length]);
}

export function getWord2VecTokenMetadata(token: string) {
  const tokenId = tokenToId.get(token);
  if (typeof tokenId === "number") {
    return { tokenId, singleToken: true };
  }
  return { tokenId: undefined, singleToken: false };
}

export async function ensureWord2VecSessions() {
  if (!embedSession) {
    embedSession = await ort.InferenceSession.create(
      EMBED_ONNX_URL,
      await epPromise
    );
  }
  if (!knnSession) {
    knnSession = await ort.InferenceSession.create(
      VECTOR_KNN_URL,
      await epPromise
    );
  }
}

export async function embedWord2VecToken(token: string) {
  const { tokenId } = getWord2VecTokenMetadata(token);
  if (typeof tokenId !== "number") {
    throw new Error(
      `Token "${token}" is not in the word2vec vocabulary. Try a different token.`
    );
  }

  await ensureWord2VecSessions();
  const output = await runEmbedLocked(() =>
    embedSession!.run({ token_id: toInt64Tensor(tokenId) })
  );
  const embedding = output.embedding?.data;
  if (!embedding) {
    throw new Error("Embedding output missing.");
  }

  return {
    tokenId,
    singleToken: true,
    vector: new Float32Array(embedding),
  };
}

export async function mostSimilarWord2VecTokensToTokenId(tokenId: number) {
  await ensureWord2VecSessions();
  const embedding = await embedWord2VecTokenById(tokenId);
  const results = await runKnnLocked(() =>
    knnSession!.run({ query_emb: toFloatTensor(embedding) })
  );
  return mapKnnResults(results);
}

async function embedWord2VecTokenById(tokenId: number) {
  const output = await runEmbedLocked(() =>
    embedSession!.run({ token_id: toInt64Tensor(tokenId) })
  );
  const embedding = output.embedding?.data;
  if (!embedding) {
    throw new Error("Embedding output missing.");
  }
  return new Float32Array(embedding);
}

export async function mostSimilarWord2VecTokensToVector(
  queryVector: Float32Array | number[]
) {
  await ensureWord2VecSessions();
  const normalized = normalize(queryVector) ?? Float32Array.from(queryVector);
  const results = await runKnnLocked(() =>
    knnSession!.run({ query_emb: toFloatTensor(normalized) })
  );
  return mapKnnResults(results);
}

function mapKnnResults(results: Record<string, ort.Tensor>) {
  const topIdx = Array.from(
    (results.top_indices?.data as BigInt64Array | undefined) ?? []
  );
  const topScores = Array.from(
    (results.top_scores?.data as Float32Array | undefined) ?? []
  );

  return topIdx.map((id, index) => {
    const numericId = Number(id);
    return {
      id: numericId,
      token: vocab[numericId] ?? `#${numericId}`,
      score: topScores[index] ?? 0,
    };
  });
}
