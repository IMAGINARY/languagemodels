import React, { useCallback, useMemo, useRef, useState } from "react";

import { env, pipeline } from "@huggingface/transformers";
env.allowLocalModels = false;
// See bug at: https://github.com/huggingface/transformers.js/issues/366#event-1351036909
// If models load html files instead of json/binary, this is likely due to the bundler vite.
// Open devtools, Application, Cache Storage, transformersjs-cache, and delete it to fix.

// TODO: optional: add a local/offline mode in a separate component

// Simple in-memory cache for the pipeline during the session
let extractor = null;
const LOCAL_MODEL_ID = "Xenova/all-MiniLM-L6-v2";

export default function ModelDemo() {
  const [status, setStatus] = useState("idle");
  const [message, setMessage] = useState("Model not loaded");
  const [vectorInfo, setVectorInfo] = useState(null);
  const [error, setError] = useState(null);
  const [text, setText] = useState(
    "Transformers are really cool for embeddings!"
  );

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
        </div>
      )}
    </div>
  );
}
