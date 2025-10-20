import React, {
  useCallback,
  useMemo,
  useRef,
  useState,
  useEffect,
} from "react";

import { env, pipeline, AutoTokenizer, AutoModel } from "@huggingface/transformers";
env.allowLocalModels = false;
// See bug at: https://github.com/huggingface/transformers.js/issues/366#event-1351036909
// If models load html files instead of json/binary, this is likely due to the bundler vite.
// Open devtools, Application, Cache Storage, transformersjs-cache, and delete it to fix.

// TODO: optional: add a local/offline mode in a separate component

// Simple in-memory cache for the pipeline during the session
let extractor = null;
let baseModel = null; // for attention extraction
const LOCAL_MODEL_ID = "Xenova/all-MiniLM-L6-v2";
import Embedding3D from "./Embedding3D.jsx";
import { PCA as PCAClass } from "ml-pca";

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
  const [tokensInfo, setTokensInfo] = useState(null); // { tokens: string[], ids: number[] }
  const [tokensPca, setTokensPca] = useState(null); // PCA from token embeddings
  const [tokensPoints3d, setTokensPoints3d] = useState([]);
  const [attnInfo, setAttnInfo] = useState(null);

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

  const runTokenize = useCallback(async () => {
    try {
      setError(null);
      setStatus("tokenizing");
      setMessage("Tokenizing …");

      let tokenizer = extractor?.tokenizer;
      if (!tokenizer) {
        tokenizer = await AutoTokenizer.from_pretrained(LOCAL_MODEL_ID, {
          revision: "main",
          progress_callback: (p) => {
            if (p?.status && p?.name) setMessage(`${p.status}: ${p.name}`);
          },
        });
      }

      // Get token strings (no specials) and ids (with specials)
      const tokenStrings = await tokenizer.tokenize(text);
      const encoded = await tokenizer.encode(text, {
        add_special_tokens: true,
      });
      const ids = Array.from(encoded?.input_ids ?? encoded?.ids ?? []);

      // Also compute per-token embeddings and PCA for visualization
      if (!extractor) {
        extractor = await pipeline("feature-extraction", LOCAL_MODEL_ID, {
          revision: "main",
          progress_callback: (p) => {
            if (p?.status && p?.name) setMessage(`${p.status}: ${p.name}`);
          },
        });
      }

      const feat = await extractor(text, {
        pooling: undefined,
        normalize: true,
      });
      const dims = feat?.dims || [];
      const data = feat?.data || [];

      let seqLen = 0;
      let hidden = 0;
      if (dims.length === 3) {
        // [batch, seq_len, hidden]
        seqLen = dims[1] || 0;
        hidden = dims[2] || 0;
      } else if (dims.length === 2) {
        // [seq_len, hidden]
        seqLen = dims[0] || 0;
        hidden = dims[1] || 0;
      }

      const dataArr = Array.from(data);
      const tokenVectors = [];
      for (let i = 0; i < seqLen; i++) {
        const start = i * hidden;
        const end = start + hidden;
        tokenVectors.push(new Float32Array(dataArr.slice(start, end)));
      }

      // Determine specials alignment using ids length (with specials) when available
      const seqLenIds = ids.length || seqLen;
      let specialsHead = 0;
      let specialsTail = 0;
      if (seqLenIds === tokenStrings.length + 2) {
        specialsHead = 1;
        specialsTail = 1;
      } else if (seqLenIds === tokenStrings.length + 1) {
        specialsHead = 1;
        specialsTail = 0;
      }
      // Build labels array aligned to sequence tokens
      let labels = new Array(seqLen).fill("").map((_, i) => `#${i + 1}`);
      let cursor = specialsHead;
      for (
        let j = 0;
        j < tokenStrings.length && cursor < seqLen;
        j++, cursor++
      ) {
        labels[cursor] = tokenStrings[j];
      }
      if (specialsHead && seqLen > 0) labels[0] = "[CLS]";
      if (specialsTail && seqLen > 1) labels[seqLen - 1] = "[SEP]";

      // No subword/word grouping — visualize tokens only for clarity

      // Compute PCA on token vectors
      let tokenPCA = null;
      if (tokenVectors.length >= 3) {
        try {
          const X = tokenVectors.map((v) => Array.from(v));
          const p = new PCAClass(X, { center: true, scale: false });
          const proj = p.predict(X, { nComponents: 3 });
          const coords = proj.to2DArray ? proj.to2DArray() : proj;
          const explainedVariance = p.getExplainedVariance();
          tokenPCA = { coords, explainedVariance };
          setTokensPca(tokenPCA);
          // Group tokens into words from tokenStrings and map to sequence indices
          const isSpecial = (t) =>
            t === "[CLS]" ||
            t === "[SEP]" ||
            t === "[PAD]" ||
            t === "<s>" ||
            t === "</s>" ||
            t === "<pad>" ||
            t === "<unk>" ||
            t === "<mask>" ||
            /^(\[.*\])$/.test(t);
          const isStartOfWord = (t) => {
            if (isSpecial(t)) return false;
            if (t.startsWith("##")) return false; // WordPiece continuation
            if (t.startsWith("Ġ")) return true; // GPT-2 BPE word start
            if (t.startsWith("▁")) return true; // SentencePiece word start
            return true; // default assume start
          };
          const cleanTokenText = (t) =>
            t.replace(/^##/, "").replace(/^Ġ/, "").replace(/^▁/, "");

          const wordGroups = [];
          let current = null;
          let seqIndex = specialsHead; // skip head special if present
          for (
            let j = 0;
            j < tokenStrings.length && seqIndex < tokenVectors.length;
            j++, seqIndex++
          ) {
            const tok = tokenStrings[j];
            const i = seqIndex; // index into tokenVectors/coords
            if (!current || isStartOfWord(tok)) {
              if (current && current.vecs.length) wordGroups.push(current);
              current = { label: cleanTokenText(tok), vecs: [tokenVectors[i]] };
            } else {
              current.label += cleanTokenText(tok);
              current.vecs.push(tokenVectors[i]);
            }
          }
          if (current && current.vecs.length) wordGroups.push(current);

          // Build word-level vectors (single-token words pass through, multi-token words are pooled)
          const wordVecs = wordGroups.map((g) => {
            if (g.vecs.length === 1) return Array.from(g.vecs[0]);
            const len = g.vecs[0]?.length || 0;
            const acc = new Float32Array(len);
            for (const v of g.vecs) {
              for (let k = 0; k < len; k++) acc[k] += v[k] || 0;
            }
            for (let k = 0; k < len; k++) acc[k] /= g.vecs.length;
            return Array.from(acc);
          });

          let wordPts = [];
          if (wordVecs.length) {
            const projWords = p.predict(wordVecs, { nComponents: 3 });
            const coordsW = projWords.to2DArray
              ? projWords.to2DArray()
              : projWords;
            wordPts = coordsW.map((c, i) => ({
              x: c[0] ?? 0,
              y: c[1] ?? 0,
              z: c[2] ?? 0,
              label: wordGroups[i].label.slice(0, 24),
              color:
                (wordGroups[i].vecs?.length || 0) > 1 ? "darkcyan" : undefined,
            }));
          }

          // Also compute sentence embedding and project with the same PCA
          let sentencePt = null;
          try {
            const sent = await extractor(text, {
              pooling: "mean",
              normalize: true,
            });
            const svec = Array.from(sent?.data ?? []);
            if (svec.length) {
              const projS = p.predict([svec], { nComponents: 3 });
              const c = projS.to2DArray ? projS.to2DArray()[0] : projS[0];
              sentencePt = {
                x: c?.[0] ?? 0,
                y: c?.[1] ?? 0,
                z: c?.[2] ?? 0,
                label: "Sentence",
                color: "red",
              };
            }
          } catch (e) {
            // ignore sentence embedding failure for visualization
          }

          // Visualize words plus the sentence point; PCA basis computed from all tokens above
          const combinedPts = sentencePt ? [...wordPts, sentencePt] : wordPts;
          setTokensPoints3d(combinedPts);
        } catch (e) {
          console.error(e);
          setTokensPca(null);
          setTokensPoints3d([]);
        }
      } else {
        setTokensPca(null);
        setTokensPoints3d([]);
      }

      setTokensInfo({ tokens: labels, ids });
      setStatus("ready");
      setMessage("Tokenized + projected");
    } catch (e) {
      setStatus("error");
      setError("Failed to tokenize. Check console for details.");
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

  window.showDataset = () => {
    console.table(
      dataset.map((d, i) => ({
        "#": i + 1,
        text: d.text,
        vector: d.vector,
      }))
    );
  };
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

  window.showPoints3d = () => {
    console.table(
      points3d.map((p) => ({
        label: p.label,
        x: p.x,
        y: p.y,
        z: p.z,
        length: Math.hypot(p.x, p.y, p.z),
      }))
    );
  };
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

        {/* <button
          className="btn btn--primary"
          onClick={runEmbedding}
          disabled={!canRun}
        >
          {status === "running" ? "Running…" : "Compute embedding"}
        </button> */}

        <button
          className="btn btn--primary"
          onClick={runTokenize}
          disabled={status === "loading" || status === "running"}
        >
          Tokenize
        </button>
        <button className="btn" onClick={async () => {
          try {
            setError(null);
            setStatus('attn');
            setMessage('Computing attention …');

            // Prepare tokenizer/model
            const tokenizer = extractor?.tokenizer || (await AutoTokenizer.from_pretrained(LOCAL_MODEL_ID, { revision: 'main' }));
            baseModel = baseModel || (await AutoModel.from_pretrained(LOCAL_MODEL_ID, { revision: 'main' }));

            const encoded = await tokenizer(text, { add_special_tokens: true });
            const outputs = await baseModel(encoded, { output_attentions: true });
            let avg = null;
            let seq = 0;
            let usedFallback = false;

            const attentions = outputs?.attentions || [];
            if (attentions.length) {
              const last = attentions[attentions.length - 1];
              const dims = last?.dims || [];
              const data = last?.data || [];
              // Expect dims: [1, heads, seq, seq]
              const heads = dims[1] || 0;
              seq = dims[2] || 0;
              const seq2 = dims[3] || 0;
              if (!heads || seq !== seq2) throw new Error(`Unexpected attention dims ${dims}`);

              // Average across heads
              const strideHead = seq * seq;
              avg = Array.from({ length: seq }, () => new Float32Array(seq));
              for (let h = 0; h < heads; h++) {
                const hOff = h * strideHead;
                for (let i = 0; i < seq; i++) {
                  const rowOff = hOff + i * seq;
                  for (let j = 0; j < seq; j++) {
                    avg[i][j] += data[rowOff + j] || 0;
                  }
                }
              }
              for (let i = 0; i < seq; i++) {
                for (let j = 0; j < seq; j++) avg[i][j] /= heads;
              }
            } else {
              // Fallback: use cosine similarity of last hidden states as an attention proxy
              usedFallback = true;
              const lastHidden = outputs?.last_hidden_state;
              const hdims = lastHidden?.dims || [];
              const hdata = lastHidden?.data || [];
              // dims: [1, seq, hidden]
              seq = hdims[1] || 0;
              const hidden = hdims[2] || 0;
              const rows = [];
              for (let i = 0; i < seq; i++) {
                const start = i * hidden;
                const vec = new Float32Array(hdata.slice(start, start + hidden));
                rows.push(vec);
              }
              // Normalize and compute cosine similarities
              const norms = rows.map(v => Math.hypot(...v));
              avg = Array.from({ length: seq }, () => new Float32Array(seq));
              for (let i = 0; i < seq; i++) {
                for (let j = 0; j < seq; j++) {
                  let dot = 0;
                  const vi = rows[i], vj = rows[j];
                  for (let k = 0; k < hidden; k++) dot += (vi[k] || 0) * (vj[k] || 0);
                  const denom = (norms[i] || 1e-9) * (norms[j] || 1e-9);
                  avg[i][j] = dot / denom;
                }
              }
            }

            // Build readable labels aligned with specials
            const toks = await tokenizer.tokenize(text);
            const ids = Array.from((await tokenizer.encode(text, { add_special_tokens: true }))?.input_ids || []);
            const haveSpecials = ids.length === toks.length + 2;
            const labels = haveSpecials ? ['[CLS]', ...toks, '[SEP]'] : (ids.length === toks.length ? toks : Array.from({length: seq}, (_, i) => `#${i+1}`));

            // Log to console: row-wise attention with labels
            console.log(usedFallback
              ? 'Attention proxy (cosine similarity of last hidden states). Rows: query → columns: key'
              : 'Avg attention (last layer). Rows: query token → columns: key token');
            console.log('Tokens:', labels);
            // Create an array of objects for console.table
            const table = avg.map((row, i) => {
              const obj = { token: labels[i] || `#${i+1}` };
              for (let j = 0; j < seq; j++) {
                obj[labels[j] || `#${j+1}`] = Number(row[j].toFixed(3));
              }
              return obj;
            });
            console.table(table);

            setAttnInfo({ seq, heads });
            setStatus('ready');
            setMessage('Attention computed — see console');
          } catch (e) {
            console.error(e);
            setStatus('error');
            setError('Failed to compute attention. See console.');
          }
        }} disabled={status === 'loading' || status === 'running'}>
          Compute attention (console)
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
          <div className="row" style={{ marginTop: 8 }}>
            <button
              className="btn"
              onClick={addToDataset}
              disabled={!lastVector}
            >
              Add point to dataset
            </button>
            <button
              className="btn"
              onClick={clearDataset}
              disabled={!dataset.length}
            >
              Clear dataset ({dataset.length})
            </button>
          </div>
        </div>
      )}

      {tokensInfo && (
        <div className="result">
          <div className="result__meta">
            <span>
              <strong>Tokens:</strong> {tokensInfo.tokens.length}
            </span>
            <span>
              <strong>IDs:</strong> {tokensInfo.ids.length}
            </span>
          </div>
          <div className="result__preview">
            <strong>Token strings:</strong>
            <div>
              {tokensInfo.tokens.map((t, i) => (
                <code key={i} style={{ marginRight: 6 }}>
                  {t}
                </code>
              ))}
            </div>
          </div>
          <div className="result__preview">
            <strong>Token IDs:</strong>
            <code>
              [ {tokensInfo.ids.slice(0, 32).join(", ")}{" "}
              {tokensInfo.ids.length > 32 ? "…" : ""} ]
            </code>
          </div>
        </div>
      )}

      {dataset.length > 0 && (
        <div className="result">
          <div className="result__meta">
            <span>
              <strong>Dataset size:</strong> {dataset.length}
            </span>
            {pca?.explainedVariance && pca.explainedVariance.length >= 3 && (
              <span>
                <strong>Explained:</strong>{" "}
                {pca.explainedVariance
                  .slice(0, 3)
                  .map((v, i) => v.toFixed(3) + (i < 2 ? ", " : ""))}
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

      {tokensInfo && (
        <div className="result">
          <div className="result__meta">
            <span>
              <strong>Token PCA:</strong>{" "}
              {tokensPca?.explainedVariance &&
              tokensPca.explainedVariance.length >= 3
                ? tokensPca.explainedVariance
                    .slice(0, 3)
                    .map((v, i) => v.toFixed(3) + (i < 2 ? ", " : ""))
                : "n/a"}
            </span>
          </div>
          {tokensPca ? (
            <>
              <Embedding3D
                points={tokensPoints3d}
                width={640}
                height={360}
                title="Word + Sentence Embeddings on Token PCA"
              />
              <div style={{ color: "#a0a7b5", fontSize: 12, marginTop: 6 }}>
                PCA fit on all tokens; visualizing word-level embeddings and the
                sentence embedding (red). Multi-token words are dark cyan.
              </div>
            </>
          ) : (
            <div className="alert">Token PCA requires at least 3 tokens.</div>
          )}
        </div>
      )}
    </div>
  );
}
