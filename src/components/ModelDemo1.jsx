import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import Embedding3DUI from "./Embedding3DUI.jsx";
import {
  embedWord2VecToken,
  ensureWord2VecSessions,
  getWord2VecTokenMetadata,
  mostSimilarWord2VecTokensToTokenId,
  mostSimilarWord2VecTokensToVector,
} from "../utils/word2vec.ts";

function getDatasetItemKey(item) {
  if (item.singleToken && typeof item.tokenId === "number") {
    return `token:${item.tokenId}`;
  }
  return `text:${String(item.labelFull || "").trim().toLowerCase()}`;
}

function dot(a, b) {
  let total = 0;
  for (let i = 0; i < a.length; i++) total += a[i] * b[i];
  return total;
}

function norm(vector) {
  return Math.sqrt(dot(vector, vector));
}

function combineLabel(labelA, labelB, labelC) {
  return `${labelA} - ${labelB} + ${labelC}`;
}

function hasSelectionChanged(previousItems, nextItems) {
  if (previousItems.length !== nextItems.length) return true;

  return previousItems.some((item, index) => {
    const nextItem = nextItems[index];
    if (!nextItem) return true;
    return (
      item.index !== nextItem.index ||
      item.item?.labelFull !== nextItem.item?.labelFull
    );
  });
}

export default function ModelDemo() {
  const [status, setStatus] = useState("idle");
  const [message, setMessage] = useState("Loading model ...");
  const [error, setError] = useState(null);
  const [text, setText] = useState("");
  const [dataset, setDataset] = useState([]); // {labelFull, labelVec, tokenId, singleToken, vector}
  const textareaRef = useRef(null);
  const trimmedText = text.trim();
  const [selectedItems, setSelectedItems] = useState([]);
  const [autoSelectIndexes, setAutoSelectIndexes] = useState(null);
  const [preserveToolOnNextSelectionChange, setPreserveToolOnNextSelectionChange] =
    useState(false);
  const [activeTool, setActiveTool] = useState(null);
  const [runningTool, setRunningTool] = useState(null);
  const [toolError, setToolError] = useState(null);
  const [nearestTokens, setNearestTokens] = useState([]);
  const [neighborSourceLabel, setNeighborSourceLabel] = useState("");
  const [similarityResult, setSimilarityResult] = useState(null);
  const [analogyResult, setAnalogyResult] = useState(null);
  const selectedVectorItem =
    selectedItems.length === 1 ? selectedItems[0].item : null;

  const canRun = useMemo(
    () => status !== "loading" && status !== "running" && trimmedText.length > 0,
    [status, trimmedText]
  );

  const loadModel = useCallback(async () => {
    try {
      setError(null);
      setStatus("loading");
      setMessage("Loading word2vec ONNX...");
      await ensureWord2VecSessions();

      setStatus("ready");
      setMessage("Model ready");
    } catch (e) {
      console.error(e);
      setStatus("error");
      setError("Failed to load model.");
      setMessage("Model failed to load");
    }
  }, []);

  useEffect(() => {
    loadModel();
  }, [loadModel]);

  useEffect(() => {
    textareaRef.current?.focus();
  }, []);

  const resetToolPanels = useCallback(() => {
    setActiveTool(null);
    setRunningTool(null);
    setToolError(null);
  }, []);

  const appendUniqueDatasetItems = useCallback((itemsToAdd) => {
    setDataset((prev) => {
      const seen = new Set(prev.map(getDatasetItemKey));
      const uniqueItems = [];
      for (const item of itemsToAdd) {
        const key = getDatasetItemKey(item);
        if (seen.has(key)) continue;
        seen.add(key);
        uniqueItems.push(item);
      }
      return uniqueItems.length ? [...prev, ...uniqueItems] : prev;
    });
  }, []);

  const runEmbedding = useCallback(async () => {
    try {
      setError(null);
      resetToolPanels();
      setStatus("running");
      setMessage("Embedding token...");

      const { tokenId, singleToken, vector } = await embedWord2VecToken(trimmedText);
      appendUniqueDatasetItems([
        {
          labelFull: trimmedText,
          labelVec: trimmedText.slice(0, 40),
          tokenId,
          singleToken,
          vector,
        },
      ]);
      setText("");
      textareaRef.current?.focus();

      setStatus("ready");
      setMessage("Embedding added");
    } catch (e) {
      console.error(e);
      setStatus("error");
      setError(e.message || "Failed to compute embedding.");
      setMessage("Embedding failed");
    }
  }, [appendUniqueDatasetItems, resetToolPanels, trimmedText]);

  const clearDataset = useCallback(() => {
    setDataset([]);
    setSelectedItems([]);
    setAutoSelectIndexes(null);
    setPreserveToolOnNextSelectionChange(false);
    resetToolPanels();
    setNearestTokens([]);
    setNeighborSourceLabel("");
    setSimilarityResult(null);
    setAnalogyResult(null);
  }, [resetToolPanels]);

  const deleteDatasetItem = useCallback((indexToDelete) => {
    setDataset((prev) => prev.filter((_, index) => index !== indexToDelete));
    setAutoSelectIndexes(null);
    setPreserveToolOnNextSelectionChange(false);
    resetToolPanels();
  }, [resetToolPanels]);

  const handleFindNearestTokens = useCallback(async () => {
    setActiveTool("neighbors");
    setRunningTool("neighbors");
    setToolError(null);

    if (selectedItems.length !== 1) {
      setToolError("Select exactly one vector to find nearest tokens.");
      setRunningTool(null);
      return;
    }

    try {
      const sourceItem = selectedItems[0].item;
      const queryVector = sourceItem.vector;
      let neighbors;

      if (sourceItem.singleToken && typeof sourceItem.tokenId === "number") {
        const raw = await mostSimilarWord2VecTokensToTokenId(
          sourceItem.tokenId
        );
        neighbors = raw.slice(1, 6);
      } else {
        neighbors = (await mostSimilarWord2VecTokensToVector(queryVector)).slice(0, 5);
      }

      const neighborVectors = await Promise.all(
        neighbors.map(async (neighbor) => {
          const labelFull = neighbor.token;
          const { tokenId, singleToken } = getWord2VecTokenMetadata(labelFull);
          const { vector } = await embedWord2VecToken(labelFull);
          return {
            labelFull,
            labelVec: labelFull.slice(0, 40) || neighbor.token,
            tokenId: typeof tokenId === "number" ? tokenId : neighbor.id,
            singleToken,
            vector,
          };
        })
      );

      setNearestTokens(neighbors);
      setNeighborSourceLabel(sourceItem.labelFull);
      appendUniqueDatasetItems(neighborVectors);
    } catch (e) {
      console.error(e);
      setToolError(e.message || "Failed to find nearest tokens.");
    } finally {
      setRunningTool(null);
    }
  }, [appendUniqueDatasetItems, selectedItems]);

  const handleSimilarity = useCallback(() => {
    setActiveTool("similarity");
    setRunningTool("similarity");
    setToolError(null);

    if (selectedItems.length !== 2) {
      setToolError("Select exactly two vectors to compute cosine similarity.");
      setRunningTool(null);
      return;
    }

    try {
      const [itemA, itemB] = selectedItems.map((entry) => entry.item);
      const normA = norm(itemA.vector);
      const normB = norm(itemB.vector);
      if (!normA || !normB) {
        throw new Error("Zero-length vector");
      }

      const cosineSimilarity = dot(itemA.vector, itemB.vector) / (normA * normB);

      setSimilarityResult({
        labelA: itemA.labelFull,
        labelB: itemB.labelFull,
        cosineSimilarity,
      });
    } catch (e) {
      console.error(e);
      setToolError("Failed to compute cosine similarity.");
    } finally {
      setRunningTool(null);
    }
  }, [selectedItems]);

  const handleAnalogy = useCallback(() => {
    setActiveTool("analogy");
    setRunningTool("analogy");
    setToolError(null);

    if (selectedItems.length !== 3) {
      setToolError("Select exactly three vectors to compute an analogy.");
      setRunningTool(null);
      return;
    }

    try {
      const [itemA, itemB, itemC] = selectedItems.map((entry) => entry.item);
      const analogyVector = new Float32Array(itemA.vector.length);
      for (let i = 0; i < analogyVector.length; i++) {
        analogyVector[i] = itemA.vector[i] - itemB.vector[i] + itemC.vector[i];
      }

      const labelFull = combineLabel(
        itemA.labelFull,
        itemB.labelFull,
        itemC.labelFull
      );
      const labelVec = combineLabel(
        itemA.labelVec,
        itemB.labelVec,
        itemC.labelVec
      ).slice(0, 40);

      appendUniqueDatasetItems([
        {
          labelFull,
          labelVec,
          tokenId: undefined,
          singleToken: false,
          vector: analogyVector,
        },
      ]);
      setPreserveToolOnNextSelectionChange(true);
      setAutoSelectIndexes([dataset.length]);
      setAnalogyResult({
        labelA: itemA.labelVec,
        labelB: itemB.labelVec,
        labelC: itemC.labelVec,
      });
    } catch (e) {
      console.error(e);
      setToolError("Failed to compute analogy.");
    } finally {
      setRunningTool(null);
    }
  }, [appendUniqueDatasetItems, dataset.length, selectedItems]);

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

  const renderToolOutput = (toolName, defaultHint, content) => {
    const isActive = activeTool === toolName;
    const isRunning = runningTool === toolName;

    return (
      <div className="model1-tool">
        {content.button}
        <div className="model1-tool__panel">
          {isActive && !isRunning ? (
            <button
              type="button"
              className="model1-tool__close"
              aria-label={`Close ${toolName} output`}
              onClick={resetToolPanels}
            >
              ×
            </button>
          ) : null}
          <div className="model1-tool__panel-body">
            {isRunning ? (
              <div className="model1-tools__empty">Running tool...</div>
            ) : isActive ? (
              toolError ? <div className="alert alert--error">{toolError}</div> : content.output
            ) : (
              <div className="model1-tools__empty">{defaultHint}</div>
            )}
          </div>
        </div>
      </div>
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
          placeholder="Type a token to embed..."
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

      {dataset.length > 0 && (
        <div className="result">
          <div className="model1-workspace">
            <div className="model1-workspace__main">
              <Embedding3DUI
                items={dataset}
                onClearItems={clearDataset}
                onDeleteItem={deleteDatasetItem}
                onSelectionChange={(items) => {
                  const selectionChanged = hasSelectionChanged(selectedItems, items);
                  setSelectedItems(items);
                  if (selectionChanged && preserveToolOnNextSelectionChange) {
                    setPreserveToolOnNextSelectionChange(false);
                  } else if (selectionChanged) {
                    resetToolPanels();
                  }
                  if (autoSelectIndexes) {
                    setAutoSelectIndexes(null);
                  }
                }}
                selectedIndexesExternal={autoSelectIndexes}
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

              {renderToolOutput("vector", "Select 1 token", {
                button: (
                  <button
                    type="button"
                    className="btn btn--primary"
                    onClick={() => {
                      setActiveTool("vector");
                      setRunningTool(null);
                      setToolError(null);
                    }}
                    disabled={runningTool !== null || selectedItems.length !== 1}
                  >
                    Vector
                  </button>
                ),
                output: selectedVectorItem ? (
                  <div className="model1-tools__vector">
                    <div className="model1-tools__vector-row">
                      <span className="model1-tools__vector-label">Dimension:</span>
                      <strong>{selectedVectorItem.vector.length}</strong>
                    </div>
                    <div className="model1-tools__vector-block">
                      <span className="model1-tools__vector-label">Vector:</span>
                      <code className="model1-tools__vector-code">
                        [
                        {Array.from(selectedVectorItem.vector).map((value, index) => (
                          <span key={index}>
                            {index ? ", " : ""}
                            {value.toFixed(6)}
                          </span>
                        ))}
                        ]
                      </code>
                    </div>
                  </div>
                ) : (
                  <div className="model1-tools__empty">
                    No vector selected.
                  </div>
                ),
              })}

              {renderToolOutput("neighbors", "Select 1 token", {
                button: (
                  <button
                    type="button"
                    className="btn btn--primary"
                    onClick={handleFindNearestTokens}
                    disabled={runningTool !== null || selectedItems.length !== 1}
                  >
                    {runningTool === "neighbors" ? "Working..." : "Neighbors"}
                  </button>
                ),
                output: nearestTokens.length ? (
                  <div className="model1-tool__content">
                    <div className="model1-tool__text">
                      Nearest tokens to {neighborSourceLabel}:
                    </div>
                    <ol className="model1-tools__list">
                      {nearestTokens.map((item) => (
                        <li key={item.id} className="model1-tools__list-item">
                          <span>{item.token}</span>
                          <span>{item.score.toFixed(3)}</span>
                        </li>
                      ))}
                    </ol>
                  </div>
                ) : (
                  <div className="model1-tools__empty">
                    No neighbor result available.
                  </div>
                ),
              })}

              {renderToolOutput("similarity", "Select 2 tokens", {
                button: (
                  <button
                    type="button"
                    className="btn btn--primary"
                    onClick={handleSimilarity}
                    disabled={runningTool !== null || selectedItems.length !== 2}
                  >
                    {runningTool === "similarity" ? "Working..." : "Similarity"}
                  </button>
                ),
                output: similarityResult ? (
                  <div className="model1-tools__distance">
                    <div className="model1-tools__distance-pair">
                      <span>{similarityResult.labelA}</span>
                      <span>{similarityResult.labelB}</span>
                    </div>
                    <div className="model1-tools__distance-metric">
                      <span>Cosine similarity</span>
                      <strong>{similarityResult.cosineSimilarity.toFixed(4)}</strong>
                    </div>
                  </div>
                ) : (
                  <div className="model1-tools__empty">
                    No similarity result available.
                  </div>
                ),
              })}

              {renderToolOutput("analogy", "Select 3 tokens", {
                button: (
                  <button
                    type="button"
                    className="btn btn--primary"
                    onClick={handleAnalogy}
                    disabled={runningTool !== null || selectedItems.length !== 3}
                  >
                    {runningTool === "analogy" ? "Working..." : "Analogy"}
                  </button>
                ),
                output: analogyResult ? (
                  <div className="model1-tool__content">
                    <div className="model1-tool__text">
                      Computed the vector {analogyResult.labelA} - {analogyResult.labelB} + {analogyResult.labelC} as the aritmetic sum and difference of their embedding vectors. Now you can find the closest neighbors to this vector.
                    </div>
                  </div>
                ) : (
                  <div className="model1-tools__empty">
                    No analogy result available.
                  </div>
                ),
              })}
            </aside>
          </div>
        </div>
      )}
    </div>
  );
}
