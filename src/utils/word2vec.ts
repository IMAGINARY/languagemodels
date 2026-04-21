import * as ort from "onnxruntime-web";
import { getWord2VecModelConfig, resolveLanguage } from "./modelRegistry.js";

type Word2VecRuntimeState = {
  embedSession: ort.InferenceSession | null;
  knnSession: ort.InferenceSession | null;
  tokenToId: Map<string, number>;
  vocab: string[];
  vocabPromise: Promise<string[]> | null;
};

const runtimeStateByLanguage = new Map<string, Word2VecRuntimeState>();
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

function getRuntimeState(language: string): Word2VecRuntimeState {
  const resolvedLanguage = resolveLanguage(language);
  let state = runtimeStateByLanguage.get(resolvedLanguage);
  if (!state) {
    state = {
      embedSession: null,
      knnSession: null,
      tokenToId: new Map(),
      vocab: [],
      vocabPromise: null,
    };
    runtimeStateByLanguage.set(resolvedLanguage, state);
  }
  return state;
}

async function loadVocab(language = "en") {
  const resolvedLanguage = resolveLanguage(language);
  const state = getRuntimeState(resolvedLanguage);
  if (!state.vocabPromise) {
    const { vocabUrl } = getWord2VecModelConfig(resolvedLanguage);
    state.vocabPromise = fetch(vocabUrl)
      .then(async (response) => {
        if (!response.ok) {
          throw new Error(
            `Failed to load ${resolvedLanguage} word2vec vocabulary.`
          );
        }
        return response.json();
      })
      .then((vocab) => {
        if (!Array.isArray(vocab)) {
          throw new Error("Word2Vec vocabulary has an invalid format.");
        }
        state.vocab = vocab;
        state.tokenToId = new Map(
          vocab.map((token, index) => [String(token), index])
        );
        return state.vocab;
      });
  }
  return state.vocabPromise;
}

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

export function getWord2VecTokenMetadata(token: string, language = "en") {
  const state = getRuntimeState(language);
  const tokenId = state.tokenToId.get(token);
  if (typeof tokenId === "number") {
    return { tokenId, singleToken: true };
  }
  return { tokenId: undefined, singleToken: false };
}

export async function ensureWord2VecSessions(language = "en") {
  const resolvedLanguage = resolveLanguage(language);
  const state = getRuntimeState(resolvedLanguage);
  const { embedUrl, vectorKnnUrl } = getWord2VecModelConfig(resolvedLanguage);

  await loadVocab(resolvedLanguage);

  if (!state.embedSession) {
    state.embedSession = await ort.InferenceSession.create(
      embedUrl,
      await epPromise
    );
  }
  if (!state.knnSession) {
    state.knnSession = await ort.InferenceSession.create(
      vectorKnnUrl,
      await epPromise
    );
  }
}

export async function embedWord2VecToken(token: string, language = "en") {
  await ensureWord2VecSessions(language);
  const state = getRuntimeState(language);
  const { tokenId } = getWord2VecTokenMetadata(token, language);
  if (typeof tokenId !== "number") {
    throw new Error(
      `Token "${token}" is not in the word2vec vocabulary. Try a different token.`
    );
  }

  const output = await runEmbedLocked(() =>
    state.embedSession!.run({ token_id: toInt64Tensor(tokenId) })
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

export async function mostSimilarWord2VecTokensToTokenId(
  tokenId: number,
  language = "en"
) {
  await ensureWord2VecSessions(language);
  const state = getRuntimeState(language);
  const embedding = await embedWord2VecTokenById(tokenId, language);
  const results = await runKnnLocked(() =>
    state.knnSession!.run({ query_emb: toFloatTensor(embedding) })
  );
  return mapKnnResults(results, language);
}

async function embedWord2VecTokenById(tokenId: number, language = "en") {
  const state = getRuntimeState(language);
  const output = await runEmbedLocked(() =>
    state.embedSession!.run({ token_id: toInt64Tensor(tokenId) })
  );
  const embedding = output.embedding?.data;
  if (!embedding) {
    throw new Error("Embedding output missing.");
  }
  return new Float32Array(embedding);
}

export async function mostSimilarWord2VecTokensToVector(
  queryVector: Float32Array | number[],
  language = "en"
) {
  await ensureWord2VecSessions(language);
  const state = getRuntimeState(language);
  const normalized = normalize(queryVector) ?? Float32Array.from(queryVector);
  const results = await runKnnLocked(() =>
    state.knnSession!.run({ query_emb: toFloatTensor(normalized) })
  );
  return mapKnnResults(results, language);
}

function mapKnnResults(results: Record<string, ort.Tensor>, language = "en") {
  const state = getRuntimeState(language);
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
      token: state.vocab[numericId] ?? `#${numericId}`,
      score: topScores[index] ?? 0,
    };
  });
}
