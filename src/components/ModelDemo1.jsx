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
import {
  getTokenMetadata,
  mostSimilarTokensToTokenId,
  mostSimilarTokensToVector,
} from "../utils/similarTokens.ts";

const VECTOR_KNN_URL = new URL(
  "../models/minilm_vector_knn.onnx",
  import.meta.url
).href;

export default function ModelDemo() {
  const [status, setStatus] = useState("idle");
  const [message, setMessage] = useState("Loading model …");
  const [vectorInfo, setVectorInfo] = useState(null);
  const [error, setError] = useState(null);
  const [text, setText] = useState("");
  const [dataset, setDataset] = useState([]); // {labelFull, labelVec, tokenId, singleToken, vector}
  const textareaRef = useRef(null);
  const trimmedText = text.trim();
  const [selectedItems, setSelectedItems] = useState([]);
  const [toolStatus, setToolStatus] = useState("idle");
  const [toolMessage, setToolMessage] = useState(
    "Select one vector to use a tool."
  );
  const [toolError, setToolError] = useState(null);
  const [nearestTokens, setNearestTokens] = useState([]);

  const nonSingleTokenWarning =
    "Warning: your input is not a single token. In this model, the embedding is the mean-pooled representation across its token parts.";

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
      const { tokenId, singleToken } = await getTokenMetadata(trimmedText);
      const nextVector = new Float32Array(data);
      setDataset((prev) => [
        ...prev,
        {
          labelFull: trimmedText,
          labelVec: trimmedText.slice(0, 40) || `#${prev.length + 1}`,
          tokenId,
          singleToken,
          vector: nextVector,
        },
      ]);
      setText("");
      textareaRef.current?.focus();

      setStatus("ready");
      setMessage(singleToken ? "Embedding added" : nonSingleTokenWarning);
    } catch (e) {
      setStatus("error");
      setError("Failed to compute embedding. Check console for details.");
      setMessage("Embedding failed");
    }
  }, [loadModel, nonSingleTokenWarning, trimmedText]);

  const clearDataset = useCallback(() => {
    setDataset([]);
    setSelectedItems([]);
    setNearestTokens([]);
    setToolStatus("idle");
    setToolMessage("Select one vector to use a tool.");
    setToolError(null);
  }, []);

  const deleteDatasetItem = useCallback((indexToDelete) => {
    setDataset((prev) => prev.filter((_, index) => index !== indexToDelete));
  }, []);

  const embedText = useCallback(
    async (inputText) => {
      if (!extractor) {
        await loadModel();
      }

      if (!extractor) {
        throw new Error("Model unavailable");
      }

      const output = await extractor(inputText, {
        pooling: "mean",
        normalize: true,
      });
      return new Float32Array(output?.data ?? []);
    },
    [loadModel]
  );

  const handleFindNearestTokens = useCallback(async () => {
    if (selectedItems.length !== 1) {
      setToolStatus("error");
      setToolError("Select exactly one vector to find nearest tokens.");
      setToolMessage("Nearest-token lookup needs one selected vector.");
      return;
    }

    try {
      setToolStatus("running");
      setToolError(null);
      setToolMessage("Finding nearest tokens…");

      const queryVector = selectedItems[0].item.vector;
      let neighbors;

      if (
        selectedItems[0].item.singleToken &&
        typeof selectedItems[0].item.tokenId === "number"
      ) {
        const raw = await mostSimilarTokensToTokenId(
          selectedItems[0].item.tokenId
        );
        neighbors = raw.slice(1, 6);
      } else {
        neighbors = (
          await mostSimilarTokensToVector(queryVector, {
            onnxUrl: VECTOR_KNN_URL,
            k: 5,
          })
        ).slice(0, 5);
      }

      const neighborVectors = await Promise.all(
        neighbors.map(async (neighbor) => {
          const labelFull = neighbor.token;
          const { tokenId, singleToken } = await getTokenMetadata(labelFull);
          return {
            labelFull,
            labelVec: labelFull.slice(0, 40) || neighbor.token,
            tokenId: typeof tokenId === "number" ? tokenId : neighbor.id,
            singleToken,
            vector: await embedText(labelFull),
          };
        })
      );

      setNearestTokens(neighbors);
      setDataset((prev) => [...prev, ...neighborVectors]);
      setToolStatus("ready");
      setToolMessage("Added 5 nearest tokens to the dataset.");
    } catch (e) {
      console.error(e);
      setToolStatus("error");
      setToolError("Failed to find nearest tokens.");
      setToolMessage("Nearest-token lookup failed.");
    }
  }, [embedText, selectedItems]);

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
        labelFull: d.labelFull,
        labelVec: d.labelVec,
        tokenId: d.tokenId,
        singleToken: d.singleToken,
        vector: d.vector,
      }))
    );
  };

  window.debugLabeledVectorsDataset = () => {
    console.log("Labeled vectors dataset:", dataset);
    return dataset;
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
          <div className="model1-workspace">
            <div className="model1-workspace__main">
              <Embedding3DUI
                items={dataset}
                onClearItems={clearDataset}
                onDeleteItem={deleteDatasetItem}
                onSelectionChange={setSelectedItems}
                height={420}
                title="Embedding Projection"
              />
            </div>

            <aside className="model1-tools">
              <div className="model1-tools__header">
                <h3 className="model1-tools__title">Tools</h3>
                <p className="model1-tools__subtitle">
                  {selectedItems.length
                    ? `${selectedItems.length} selected`
                    : "No vectors selected"}
                </p>
              </div>

              <button
                type="button"
                className="btn btn--primary"
                onClick={handleFindNearestTokens}
                disabled={toolStatus === "running" || selectedItems.length !== 1}
              >
                {toolStatus === "running" ? "Working…" : "Find 5 nearest tokens"}
              </button>

              <div className="model1-tools__status">
                <span className={`status status--${toolStatus}`}>{toolMessage}</span>
                {toolError ? <div className="alert alert--error">{toolError}</div> : null}
              </div>

              <div className="model1-tools__results">
                <div className="model1-tools__panel-header">
                  <h4 className="model1-tools__panel-title">Nearest tokens</h4>
                </div>
                {nearestTokens.length ? (
                  <ol className="model1-tools__list">
                    {nearestTokens.map((item) => (
                      <li key={item.id} className="model1-tools__list-item">
                        <span>{item.token}</span>
                        <span>{item.score.toFixed(3)}</span>
                      </li>
                    ))}
                  </ol>
                ) : (
                  <div className="model1-tools__empty">
                    Run a tool to show token matches here.
                  </div>
                )}
              </div>
            </aside>
          </div>
        </div>
      )}
    </div>
  );
}
