import React, { useCallback, useMemo, useState } from "react";
import { env, pipeline } from "@huggingface/transformers";
import { PCA as PCAClass } from "ml-pca";
import Embedding3D from "./Embedding3D.jsx";
import {
  mostSimilarTokensToToken,
  mostSimilarTokensToVector,
} from "../utils/similarTokens.ts";

env.allowLocalModels = false;

// Simple in-memory cache for the embedding pipeline
let extractor = null;
const LOCAL_MODEL_ID = "Xenova/all-MiniLM-L6-v2";
const VECTOR_KNN_URL = new URL(
  "../models/minilm_vector_knn.onnx",
  import.meta.url
).href;

const EPS = 1e-8;

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

// Build an orthonormal frame (v1, v2, v3) given v1.
function orthonormalFrame(v1) {
  const v1n = normalize(v1);
  if (!v1n) throw new Error("Zero-length vector");
  const d = v1n.length;

  // Pick a basis vector least aligned with v1
  let minIdx = 0;
  let minVal = Math.abs(v1n[0] ?? 0);
  for (let i = 1; i < d; i++) {
    const val = Math.abs(v1n[i] ?? 0);
    if (val < minVal) {
      minVal = val;
      minIdx = i;
    }
  }
  const e2 = new Float32Array(d);
  e2[minIdx] = 1;

  // Gram-Schmidt for v2
  let v2 = new Float32Array(d);
  const proj12 = dot(e2, v1n);
  for (let i = 0; i < d; i++) v2[i] = e2[i] - proj12 * v1n[i];
  v2 = normalize(v2);
  if (!v2) throw new Error("Failed to build v2");

  // Choose another axis different from minIdx
  const e3 = new Float32Array(d);
  e3[(minIdx + 1) % d] = 1;

  // Gram-Schmidt for v3 against v1 and v2
  let v3 = new Float32Array(d);
  const proj13 = dot(e3, v1n);
  const proj23 = dot(e3, v2);
  for (let i = 0; i < d; i++)
    v3[i] = e3[i] - proj13 * v1n[i] - proj23 * v2[i];
  v3 = normalize(v3);
  if (!v3) throw new Error("Failed to build v3");

  return { v1: v1n, v2, v3 };
}

export default function ModelDemo3() {
  const [text, setText] = useState("king");
  const [status, setStatus] = useState("idle"); // idle | loading | running | ready | error
  const [message, setMessage] = useState("Model not loaded");
  const [error, setError] = useState(null);
  const [similar, setSimilar] = useState([]); // [{token, score}]
  const [vectors, setVectors] = useState([]); // [{label, vector, hidden?}]

  const loadModel = useCallback(async () => {
    try {
      setStatus("loading");
      setMessage("Loading model…");
      setError(null);
      extractor = await pipeline("feature-extraction", LOCAL_MODEL_ID, {
        revision: "main",
        progress_callback: (p) => {
          if (p?.status && p?.name) setMessage(`${p.status}: ${p.name}`);
        },
      });
      setStatus("ready");
      setMessage("Model ready");
    } catch (e) {
      console.error(e);
      setStatus("error");
      setError("Failed to load model.");
    }
  }, []);

  const embedText = useCallback(async (inputText) => {
    if (!extractor) {
      extractor = await pipeline("feature-extraction", LOCAL_MODEL_ID, {
        revision: "main",
      });
    }
    const output = await extractor(inputText, {
      pooling: "mean",
      normalize: true,
    });
    return new Float32Array(output?.data ?? []);
  }, []);

  const run = useCallback(async () => {
    const trimmed = text.trim();
    if (!trimmed) return;
    try {
      setStatus("running");
      setMessage("Embedding and finding neighbors…");
      setError(null);

      let neighbors;
      let queryVec;

      // Prefer the token KNN path if the input is a single token
      try {
        const raw = await mostSimilarTokensToToken(trimmed);
        // Drop the first self-hit and keep the next 5
        neighbors = raw.slice(1, 6);
        queryVec = await embedText(trimmed);
      } catch {
        queryVec = await embedText(trimmed);
        neighbors = await mostSimilarTokensToVector(queryVec, {
          onnxUrl: VECTOR_KNN_URL,
        });
      }
      setSimilar(neighbors);

      // 3) Build orthonormal frame (v1, v2, v3)
      const { v1, v2, v3 } = orthonormalFrame(queryVec);
      // 4) Collect vectors for visualization: v1, v2, v3, plus top-5 neighbors of v1
      const neighborVectors = await Promise.all(
        neighbors.map(async (n) => ({
          label: n.token,
          vector: await embedText(n.token),
        }))
      );
      const frames = [
        { label: trimmed, vector: v1, color: "#ff4d4d" },
        { label: "v2 (orthogonal)", vector: v2, hidden: true },
        { label: "v3 (orthogonal)", vector: v3, hidden: true },
      ];
      setVectors([...frames, ...neighborVectors]);

      setStatus("ready");
      setMessage("Done");
    } catch (e) {
      console.error(e);
      setStatus("error");
      setError("Failed to compute neighbors.");
    }
  }, [embedText, text]);

  const pcaResult = useMemo(() => {
    if (vectors.length < 2) return null;
    try {
      const X = vectors.map((d) => Array.from(d.vector));
      const p = new PCAClass(X, { center: true, scale: false });
      const proj = p.predict(X, { nComponents: 3 });
      const coords = proj.to2DArray ? proj.to2DArray() : proj;
      return {
        coords,
        explained: p.getExplainedVariance()?.slice(0, 3) ?? [],
      };
    } catch (e) {
      console.error(e);
      return null;
    }
  }, [vectors]);

  const points3d = useMemo(() => {
    if (!pcaResult?.coords) return [];
    const pts = [];
    for (let i = 0; i < pcaResult.coords.length; i++) {
      const meta = vectors[i];
      if (!meta || meta.hidden) continue;
      const c = pcaResult.coords[i];
      pts.push({
        x: c[0] ?? 0,
        y: c[1] ?? 0,
        z: c[2] ?? 0,
        label: meta.label ?? `#${i + 1}`,
        color: meta.color,
      });
    }
    return pts;
  }, [pcaResult, vectors]);

  return (
    <div className="panel">
      <div className="row">
        <input
          className="input"
          type="text"
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Enter a word or short phrase (e.g., king)"
        />
      </div>

      <div className="row row--actions">
        <button className="btn" onClick={loadModel} disabled={status === "loading" || status === "running"}>
          {status === "loading" ? "Loading…" : "Load model"}
        </button>
        <button
          className="btn btn--primary"
          onClick={run}
          disabled={status === "loading" || status === "running"}
        >
          {status === "running" ? "Working…" : "Find similar tokens"}
        </button>
        <span className={`status status--${status}`}>{message}</span>
      </div>

      {error && <div className="alert alert--error">{error}</div>}

      {similar.length > 0 && (
        <div className="result">
          <h3>Top 5 similar tokens</h3>
          <ol>
            {similar.map((s, i) => (
              <li key={i}>
                <strong>{s.token}</strong> <small>(score {s.score.toFixed(3)})</small>
              </li>
            ))}
          </ol>
        </div>
      )}

      {points3d.length > 0 && (
        <div className="result">
          <div className="result__meta">
            {pcaResult?.explained?.length ? (
              <span>
                <strong>Explained:</strong>{" "}
                {pcaResult.explained
                  .map((v, i) => v.toFixed(3) + (i < pcaResult.explained.length - 1 ? ", " : ""))}
              </span>
            ) : null}
          </div>
          <Embedding3D
            points={points3d}
            width={640}
            height={360}
            title="PCA of query + nearest tokens"
          />
        </div>
      )}
    </div>
  );
}
