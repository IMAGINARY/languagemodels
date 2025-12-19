import React, { useCallback, useMemo, useState } from "react";
import { PCA as PCAClass } from "ml-pca";
import * as ort from "onnxruntime-web";
import Embedding3D from "./Embedding3D.jsx";
import vocab from "../models/word2vec_vocab.json";

const EMBED_ONNX_URL = new URL(
  "../models/word2vec_embed.onnx",
  import.meta.url
).href;
const VECTOR_KNN_URL = new URL(
  "../models/word2vec_vector_knn.onnx",
  import.meta.url
).href;

// Prefer WebGPU; fallback to WASM
const epPromise = (async () => {
  try {
    const webgpu = await ort.env.webgpu;
    if (webgpu?.adapterInfo) return { executionProviders: ["webgpu", "wasm"] };
  } catch (err) {
    console.warn("WebGPU probe failed; using WASM", err);
  }
  return { executionProviders: ["wasm"] };
})();

const tokenToId = new Map(vocab.map((t, i) => [t, i]));
let embedSession = null;
let knnSession = null;
const EPS = 1e-8;

function createRunLock() {
  let chain = Promise.resolve();
  return (task) => {
    const next = chain.then(task, task);
    chain = next.catch(() => {});
    return next;
  };
}

// ONNX Runtime web sessions cannot handle concurrent runs; serialize calls per session.
const runEmbedLocked = createRunLock();
const runKnnLocked = createRunLock();

function dot(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function norm(a) {
  return Math.sqrt(dot(a, a));
}

function normalize(a) {
  const n = norm(a);
  if (n < EPS) return null;
  const out = new Float32Array(a.length);
  const inv = 1 / n;
  for (let i = 0; i < a.length; i++) out[i] = a[i] * inv;
  return out;
}

async function ensureSessions() {
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

function toInt64Tensor(value) {
  return new ort.Tensor("int64", new BigInt64Array([BigInt(value)]), [1]);
}

function toFloatTensor(vec) {
  const arr = vec instanceof Float32Array ? vec : Float32Array.from(vec);
  return new ort.Tensor("float32", arr, [1, arr.length]);
}

async function embedToken(token) {
  const id = tokenToId.get(token);
  if (typeof id !== "number") {
    throw new Error(
      `Token "${token}" is not in the word2vec vocabulary. Try a different token.`
    );
  }
  await ensureSessions();
  const output = await runEmbedLocked(() =>
    embedSession.run({ token_id: toInt64Tensor(id) })
  );
  const emb = output.embedding?.data;
  if (!emb) throw new Error("Embedding output missing.");
  return new Float32Array(emb);
}

export default function ModelDemo4() {
  const [t1, setT1] = useState("king");
  const [t2, setT2] = useState("man");
  const [t3, setT3] = useState("woman");
  const [status, setStatus] = useState("idle"); // idle | loading | running | ready | error
  const [message, setMessage] = useState("Model not loaded");
  const [error, setError] = useState(null);
  const [neighbors, setNeighbors] = useState([]);
  const [points, setPoints] = useState([]);
  const [explained, setExplained] = useState([]);

  const loadModel = useCallback(async () => {
    try {
      setStatus("loading");
      setMessage("Loading word2vec ONNX…");
      setError(null);
      await ensureSessions();
      setStatus("ready");
      setMessage("Model ready");
    } catch (e) {
      console.error(e);
      setStatus("error");
      setError("Failed to load model.");
    }
  }, []);

  const run = useCallback(async () => {
    const a = t1.trim();
    const b = t2.trim();
    const c = t3.trim();
    if (!a || !b || !c) {
      setError("Please provide three tokens.");
      return;
    }
    try {
      setStatus("running");
      setMessage("Embedding tokens…");
      setError(null);
      setNeighbors([]);
      setPoints([]);
      setExplained([]);

      const [v1, v2, v3] = await Promise.all([
        embedToken(a),
        embedToken(b),
        embedToken(c),
      ]);

      const w = new Float32Array(v1.length);
      for (let i = 0; i < v1.length; i++) {
        w[i] = v1[i] - v2[i] + v3[i];
      }
      const wQuery = normalize(w) ?? w;

      setMessage("Searching nearest neighbors…");
      await ensureSessions();
      const results = await runKnnLocked(() =>
        knnSession.run({ query_emb: toFloatTensor(wQuery) })
      );
      const topIdx = Array.from(results.top_indices.data || []);
      const topScores = Array.from(results.top_scores.data || []);
      const nn = topIdx.map((id, i) => ({
        id: Number(id),
        token: vocab[Number(id)] ?? `#${id}`,
        score: topScores[i] ?? 0,
      }));
      setNeighbors(nn);

      // Project everything using PCA trained on {v1, v2, v3}
      const baseMatrix = [v1, v2, v3].map((v) => Array.from(v));
      const p = new PCAClass(baseMatrix, { center: true, scale: false });

      const project = (vec) => {
        const proj = p.predict([Array.from(vec)], { nComponents: 3 });
        const arr = proj.to2DArray ? proj.to2DArray() : proj;
        const c0 = arr[0] ?? [];
        return [c0[0] ?? 0, c0[1] ?? 0, c0[2] ?? 0];
      };

      const [p1, p2, p3] = baseMatrix.map((_, idx) => project(baseMatrix[idx]));
      const pw = project(w);

      const neighborVectors = await Promise.all(
        nn.map(async (n) => ({
          token: n.token,
          coords: project(await embedToken(n.token)),
        }))
      );

      const vizPoints = [
        { x: p1[0], y: p1[1], z: p1[2], label: a, color: "#34d399" }, // green
        { x: p2[0], y: p2[1], z: p2[2], label: b, color: "#facc15" }, // yellow
        { x: p3[0], y: p3[1], z: p3[2], label: c, color: "#facc15" }, // yellow
        {
          x: pw[0],
          y: pw[1],
          z: pw[2],
          label: `${a} + ${c} - ${b}`,
          color: "#ef4444",
        }, // red
        {
          x: pw[0],
          y: pw[1],
          z: pw[2],
          x0: p1[0],
          y0: p1[1],
          z0: p1[2],
          label: `${c} - ${b} (from ${a})`,
          color: "#60a5fa",
        }, // blue offset from v1
        ...neighborVectors.map((n) => ({
          x: n.coords[0],
          y: n.coords[1],
          z: n.coords[2],
          label: n.token,
          color: "#22d3ee",
        })),
      ];

      setPoints(vizPoints);
      setExplained(p.getExplainedVariance()?.slice(0, 3) ?? []);

      setStatus("ready");
      setMessage("Done");
    } catch (e) {
      console.error(e);
      setStatus("error");
      setError(e.message || "Failed to compute analogy.");
    }
  }, [t1, t2, t3]);

  const explainedLabel = useMemo(() => {
    if (!explained?.length) return null;
    return explained.map((v, i) => `PC${i + 1}: ${v.toFixed(3)}`).join(", ");
  }, [explained]);

  return (
    <div className="panel">
      <div className="row" style={{ gap: 8 }}>
        <input
          className="input"
          type="text"
          value={t1}
          onChange={(e) => setT1(e.target.value)}
          placeholder="Token t1 (e.g., king)"
        />
        <input
          className="input"
          type="text"
          value={t2}
          onChange={(e) => setT2(e.target.value)}
          placeholder="Token t2 (e.g., man)"
        />
        <input
          className="input"
          type="text"
          value={t3}
          onChange={(e) => setT3(e.target.value)}
          placeholder="Token t3 (e.g., woman)"
        />
      </div>

      <div className="row row--actions">
        <button
          className="btn"
          onClick={loadModel}
          disabled={status === "loading" || status === "running"}
        >
          {status === "loading" ? "Loading…" : "Load model"}
        </button>
        <button
          className="btn btn--primary"
          onClick={run}
          disabled={status === "loading" || status === "running"}
        >
          {status === "running" ? "Working…" : "Run analogy"}
        </button>
        <span className={`status status--${status}`}>{message}</span>
      </div>

      {error && <div className="alert alert--error">{error}</div>}

      {neighbors.length > 0 && (
        <div className="result">
          <p style={{ marginBottom: 8 }}>
            The five closest tokens to <strong>{t1}</strong> + (
            <strong>{t3}</strong> - <strong>{t2}</strong>) are:
          </p>
          <ol>
            {neighbors.map((n, i) => (
              <li key={i}>
                <strong>{n.token}</strong>{" "}
                <small>(score {n.score.toFixed(3)})</small>
              </li>
            ))}
          </ol>
        </div>
      )}

      {points.length > 0 && (
        <div className="result">
          <div className="result__meta">
            {explainedLabel ? <span>{explainedLabel}</span> : null}
          </div>
          <Embedding3D
            points={points}
            width={640}
            height={360}
            title="Analogy vectors (PCA)"
          />
        </div>
      )}
    </div>
  );
}
