import React, { useCallback, useMemo, useRef, useState, useEffect } from "react";

import { env, pipeline } from "@huggingface/transformers";
env.allowLocalModels = false;
// See bug at: https://github.com/huggingface/transformers.js/issues/366#event-1351036909
// If models load html files instead of json/binary, this is likely due to the bundler vite.
// Open devtools, Application, Cache Storage, transformersjs-cache, and delete it to fix.

// TODO: optional: add a local/offline mode in a separate component

// Simple in-memory cache for the pipeline during the session
let extractor = null;
const LOCAL_MODEL_ID = "Xenova/all-MiniLM-L6-v2";
import Embedding3D from "./Embedding3D.jsx";
import PCAImport from "ml-pca";
const PCAClass = PCAImport?.default ?? PCAImport;

export default function ModelDemo() {
  const [status, setStatus] = useState("idle");
  const [message, setMessage] = useState("Model not loaded");
  const [vectorInfo, setVectorInfo] = useState(null);
  const [error, setError] = useState(null);
  const [text, setText] = useState(
    "Transformers are really cool for embeddings!"
  );
  const [lastVector, setLastVector] = useState(null);
  const [dataset, setDataset] = useState([]); // {text, vector: Float32Array}
  const [pca, setPca] = useState(null); // {coords, components, explainedVariance}

  const abortRef = useRef(null);

  const canRun = useMemo(
    () => status !== "loading" && status !== "running",
    [status]
  );

  const loadModel = useCallback(async () => {
    try {
      setError(null);
      setStatus("loading");
      setMessage("Loading model (first time may take a while) …");

      // Model: Small sentence embedding model
      // Note: transformers.js caches model files in IndexedDB by default
      extractor = await pipeline("feature-extraction", LOCAL_MODEL_ID, {
        revision: "main",
        progress_callback: (p) => {
          if (p?.status && p?.name) {
            setMessage(`${p.status}: ${p.name}`);
          }
        },
      });

      setStatus("ready");
      setMessage("Model ready");
    } catch (e) {
      setStatus("error");
      setError("Failed to load model. Check console for details.");
    }
  }, []);

  const runEmbedding = useCallback(async () => {
    try {
      setError(null);
      setStatus("running");
      setMessage("Computing embedding …");

      if (!extractor) {
        // Lazy-load if user skips the explicit load step (remote)
        extractor = await pipeline("feature-extraction", LOCAL_MODEL_ID, {
          revision: "main",
          progress_callback: (p) => {
            if (p?.status && p?.name) setMessage(`${p.status}: ${p.name}`);
          },
        });
      }

      const output = await extractor(text, {
        pooling: "mean",
        normalize: true,
      });
      const data = output?.data ?? [];
      const dims = output?.dims ?? [data.length];

      const preview = Array.from(data).slice(0, 8);
      setVectorInfo({
        length: data.length,
        dims: dims.join("×"),
        preview,
      });
      setLastVector(new Float32Array(data));

      setStatus("ready");
      setMessage("Done");
    } catch (e) {
      setStatus("error");
      setError("Failed to compute embedding. Check console for details.");
    }
  }, [text]);

  const cancel = useCallback(() => {
    try {
      abortRef.current?.abort?.();
      setStatus("idle");
      setMessage("Cancelled");
    } catch {}
  }, []);

  const addToDataset = useCallback(() => {
    if (!lastVector) return;
    setDataset((prev) => [...prev, { text, vector: lastVector }]);
  }, [lastVector, text]);

  const clearDataset = useCallback(() => {
    setDataset([]);
    setPca(null);
  }, []);

  // Recompute PCA when dataset changes
  useEffect(() => {
    if (dataset.length >= 3) {
      try {
        const X = dataset.map((d) => Array.from(d.vector));
        const p = new PCAClass(X, { center: true, scale: false });
        const proj = p.predict(X, { nComponents: 3 });
        const coords = proj.to2DArray ? proj.to2DArray() : proj;
        const explainedVariance = p.getExplainedVariance();
        setPca({ coords, explainedVariance });
      } catch (e) {
        console.error(e);
        setPca(null);
      }
    } else {
      setPca(null);
    }
  }, [dataset]);

  const points3d = useMemo(() => {
    if (!pca || !pca.coords) return [];
    return pca.coords.map((c, i) => ({
      x: c[0] ?? 0,
      y: c[1] ?? 0,
      z: c[2] ?? 0,
      label: dataset[i]?.text?.slice(0, 24) || `#${i + 1}`,
    }));
  }, [pca, dataset]);

  return (
    <div className="panel">
      <div className="row">
        <textarea
          className="input"
          rows={4}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Type some text to embed…"
        />
      </div>

      <div className="row row--actions">
        <button className="btn" onClick={loadModel} disabled={!canRun}>
          {status === "loading" ? "Loading…" : "Load model"}
        </button>
        <button
          className="btn btn--primary"
          onClick={runEmbedding}
          disabled={!canRun}
        >
          {status === "running" ? "Running…" : "Compute embedding"}
        </button>
        <span className={`status status--${status}`}>{message}</span>
      </div>

      {error && <div className="alert alert--error">{error}</div>}

      {vectorInfo && (
        <div className="result">
          <div className="result__meta">
            <span>
              <strong>Dims:</strong> {vectorInfo.dims}
            </span>
            <span>
              <strong>Length:</strong> {vectorInfo.length}
            </span>
          </div>
          <div className="result__preview">
            <strong>Preview:</strong>
            <code>
              [{" "}
              {vectorInfo.preview.map((v, i) => (
                <span key={i}>
                  {i ? ", " : ""}
                  {v.toFixed(4)}
                </span>
              ))}{" "}
              … ]
            </code>
          </div>
          <div className="row" style={{ marginTop: 8 }}>
            <button className="btn" onClick={addToDataset} disabled={!lastVector}>
              Add point to dataset
            </button>
            <button className="btn" onClick={clearDataset} disabled={!dataset.length}>
              Clear dataset ({dataset.length})
            </button>
          </div>
        </div>
      )}

      {dataset.length > 0 && (
        <div className="result">
          <div className="result__meta">
            <span><strong>Dataset size:</strong> {dataset.length}</span>
            {pca?.explainedVariance && pca.explainedVariance.length >= 3 && (
              <span>
                <strong>Explained:</strong> {pca.explainedVariance.slice(0, 3).map((v, i) => (v.toFixed(3) + (i < 2 ? ", " : "")))}
              </span>
            )}
          </div>
          {pca ? (
            <Embedding3D
              points={points3d}
              width={640}
              height={360}
              title="PCA Projection (Top 3 PCs)"
            />
          ) : (
            <div className="alert">Add at least 3 points to compute PCA.</div>
          )}
        </div>
      )}
    </div>
  );
}
