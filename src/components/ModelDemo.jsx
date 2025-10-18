import React, { useCallback, useMemo, useRef, useState } from "react";

import { env, pipeline } from "@huggingface/transformers";
env.allowLocalModels = false;
// See bug at: https://github.com/huggingface/transformers.js/issues/366#event-1351036909
// If models load html files instead of json/binary, this is likely due to the bundler vite.
// Open devtools, Application, Cache Storage, transformersjs-cache, and delete it to fix.

//TODO: implement local models

// Simple in-memory cache for the pipeline during the session
let extractor = null;
const LOCAL_MODEL_ID = "Xenova/all-MiniLM-L6-v2";

const logPrefix = "[transformers-demo]";

export default function ModelDemo() {
  const [status, setStatus] = useState("idle");
  const [message, setMessage] = useState("Model not loaded");
  const [vectorInfo, setVectorInfo] = useState(null);
  const [error, setError] = useState(null);
  const [text, setText] = useState(
    "Transformers are really cool for embeddings!"
  );
  const [events, setEvents] = useState([]);
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
      setEvents((e) =>
        [{ t: Date.now(), m: "init loadModel" }, ...e].slice(0, 500)
      );
      console.log(logPrefix, "loadModel: start");

      // Model: Small sentence embedding model
      // Note: transformers.js caches model files in IndexedDB by default
      extractor = await pipeline("feature-extraction", LOCAL_MODEL_ID, {
        revision: "main",
        progress_callback: (p) => {
          setEvents((e) =>
            [{ t: Date.now(), m: JSON.stringify(p) }, ...e].slice(0, 50)
          );
          if (p?.status && p?.name) {
            setMessage(`${p.status}: ${p.name}`);
          }
          // Also log to console for Network tab correlation
          console.log(logPrefix, "progress", p);
        },
      });

      setStatus("ready");
      setMessage("Model ready");
      console.log(logPrefix, "loadModel: ready");
    } catch (e) {
      console.error(e);
      setStatus("error");
      setError("Failed to load model. Check console for details.");
      setEvents((ev) =>
        [{ t: Date.now(), m: `error: ${e?.message || e}` }, ...ev].slice(0, 50)
      );
    }
  }, []);

  const runEmbedding = useCallback(async () => {
    try {
      setError(null);
      setStatus("running");
      setMessage("Computing embedding …");

      if (!extractor) {
        // Lazy-load if user skips the explicit load step
        const checks = await verifyLocalModel();
        const missing = checks.filter((c) => !c.ok && !c.optional);
        if (missing.length) {
          const lines = missing.map(
            (m) => `missing or invalid (${m.status} ${m.ct || ""}): ${m.url}`
          );
          const msg = `Local model files not found or invalid.\n- ${lines.join(
            "\n- "
          )}`;
          console.error(logPrefix, msg);
          setStatus("error");
          setError(msg);
          setMessage("Local model missing");
          setEvents((e) => [{ t: Date.now(), m: msg }, ...e].slice(0, 50));
          return;
        }
        const missingOptional = checks.filter((c) => !c.ok && c.optional);
        if (missingOptional.length) {
          const lines = missingOptional.map(
            (m) => `optional missing (${m.status} ${m.ct || ""}): ${m.url}`
          );
          setEvents((e) =>
            [{ t: Date.now(), m: lines.join("\n") }, ...e].slice(0, 50)
          );
        }
        const { pipeline } = await getTransformers();
        extractor = await pipeline("feature-extraction", LOCAL_MODEL_ID, {
          revision: "main",
          local_files_only: true,
          progress_callback: (p) => {
            setEvents((e) =>
              [{ t: Date.now(), m: JSON.stringify(p) }, ...e].slice(0, 50)
            );
            if (p?.status && p?.name) {
              setMessage(`${p.status}: ${p.name}`);
            }
            console.log(logPrefix, "progress", p);
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
      console.error(e);
      setStatus("error");
      setError("Failed to compute embedding. Check console for details.");
      setEvents((ev) =>
        [{ t: Date.now(), m: `error: ${e?.message || e}` }, ...ev].slice(0, 50)
      );
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

      {events.length > 0 && (
        <div className="result">
          <div className="result__meta">
            <strong>Events</strong>
            <span className="status">(latest first)</span>
          </div>
          <div className="result__preview">
            <code>
              {events.slice(0, 8).map((e, i) => (
                <div key={i}>
                  • {new Date(e.t).toLocaleTimeString()} — {e.m}
                </div>
              ))}
            </code>
          </div>
        </div>
      )}
    </div>
  );
}
