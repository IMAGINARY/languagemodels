import React, {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { env, pipeline } from "@huggingface/transformers";
env.allowLocalModels = false;
// See bug at: https://github.com/huggingface/transformers.js/issues/366#event-1351036909
// If models load html files instead of json/binary, this is likely due to the bundler vite.
// Open devtools, Application, Cache Storage, transformersjs-cache, and delete it to fix.

// TODO: optional: add a local/offline mode in a separate component

// Simple in-memory cache for the pipeline during the session
let extractor = null;
const LOCAL_MODEL_ID = "Xenova/all-MiniLM-L6-v2";
import Embedding3DUI from "./Embedding3DUI.jsx";

export default function ModelDemo() {
  const [status, setStatus] = useState("idle");
  const [message, setMessage] = useState("Loading model …");
  const [vectorInfo, setVectorInfo] = useState(null);
  const [error, setError] = useState(null);
  const [text, setText] = useState("");
  const [dataset, setDataset] = useState([]); // {text, vector: Float32Array}
  const textareaRef = useRef(null);
  const trimmedText = text.trim();

  const canRun = useMemo(
    () => status !== "loading" && status !== "running" && trimmedText.length > 0,
    [status, trimmedText]
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
      setMessage("Model failed to load");
    }
  }, []);

  useEffect(() => {
    if (!extractor) {
      loadModel();
      return;
    }
    setStatus("ready");
    setMessage("Model ready");
  }, [loadModel]);

  useEffect(() => {
    textareaRef.current?.focus();
  }, []);

  const runEmbedding = useCallback(async () => {
    try {
      setError(null);
      setStatus("running");
      setMessage("Computing embedding …");

      if (!extractor) {
        await loadModel();
      }

      if (!extractor) {
        throw new Error("Model unavailable");
      }

      const output = await extractor(trimmedText, {
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
      const nextVector = new Float32Array(data);
      setDataset((prev) => [...prev, { text: trimmedText, vector: nextVector }]);
      setText("");
      textareaRef.current?.focus();

      setStatus("ready");
      setMessage("Embedding added");
    } catch (e) {
      setStatus("error");
      setError("Failed to compute embedding. Check console for details.");
      setMessage("Embedding failed");
    }
  }, [loadModel, trimmedText]);

  const clearDataset = useCallback(() => {
    setDataset([]);
  }, []);

  const deleteDatasetItem = useCallback((indexToDelete) => {
    setDataset((prev) => prev.filter((_, index) => index !== indexToDelete));
  }, []);

  const handleTextareaKeyDown = useCallback(
    (event) => {
      if (event.key !== "Enter" || event.shiftKey) return;
      event.preventDefault();
      if (canRun) {
        runEmbedding();
      }
    },
    [canRun, runEmbedding]
  );

  window.showDataset = () => {
    console.table(
      dataset.map((d, i) => ({
        "#": i + 1,
        text: d.text,
        vector: d.vector,
      }))
    );
  };
  return (
    <div className="panel">
      <div className="row">
        <textarea
          ref={textareaRef}
          className="input"
          rows={4}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={handleTextareaKeyDown}
          placeholder="Type a word to embed…"
        />
      </div>

      <div className="row row--actions">
        <button
          className="btn btn--primary"
          onClick={runEmbedding}
          disabled={!canRun}
        >
          {status === "running" ? "Running…" : "Embed"}
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
              {"[ "}
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

      {dataset.length > 0 && (
        <div className="result">
          <Embedding3DUI
            items={dataset.map((item, index) => ({
              label: item.text?.slice(0, 40) || `#${index + 1}`,
              vector: item.vector,
            }))}
            onClearItems={clearDataset}
            onDeleteItem={deleteDatasetItem}
            height={420}
            title="Embedding Projection"
          />
        </div>
      )}
    </div>
  );
}
