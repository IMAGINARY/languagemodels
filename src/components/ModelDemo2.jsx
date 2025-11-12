import React, { useState } from "react";
import AttentionArcs from "./AttentionArcs.jsx";
import Embedding3D from "./Embedding3D.jsx";
import { PCA as PCAClass } from "ml-pca";
import { computeAttention } from "../utils/computeAttention.js";

export default function ModelDemo2() {
  const [text, setText] = useState("Transformers are really cool for embeddings!");
  const [status, setStatus] = useState("idle");
  const [message, setMessage] = useState("Idle");
  const [error, setError] = useState(null);
  const [tokens, setTokens] = useState(null); // string[]
  const [baseEmbeds, setBaseEmbeds] = useState(null); // Float32Array[] per token
  const [groups, setGroups] = useState(null); // [{label, idxs}]
  const [pcaModel, setPcaModel] = useState(null); // PCAClass instance
  const [points3d, setPoints3d] = useState([]); // current 3D points (words)
  const [attnMatrix, setAttnMatrix] = useState(null); // current attention (for arcs)
  const [numHeads, setNumHeads] = useState(0);
  const [attnSel, setAttnSel] = useState("none"); // 'none' | 'mean' | 'h1'..'hN'

  async function handleTokenize() {
    try {
      setError(null);
      setStatus("running");
      setMessage("Tokenizing + embeddings …");
      const { tokens, embeddings, numHeads } = await computeAttention(text, { head: "none" });
      setTokens(tokens);
      setBaseEmbeds(embeddings);
      setNumHeads(numHeads || 0);
      setAttnMatrix(null);
      setAttnSel("none");

      // Build word-level groups from tokens (skip specials; compose multi-token words)
      const isSpecial = (t) =>
        t === "[CLS]" ||
        t === "[SEP]" ||
        t === "[PAD]" ||
        t === "<s>" ||
        t === "</s>" ||
        t === "<pad>" ||
        t === "<unk>" ||
        t === "<mask>" ||
        /^\[.*\]$/.test(t);
      const isStartOfWord = (t) => {
        if (isSpecial(t)) return false;
        if (t.startsWith("##")) return false; // WordPiece continuation
        if (t.startsWith("Ġ")) return true; // GPT-2 BPE word start
        if (t.startsWith("▁")) return true; // SentencePiece word start
        return true; // default: treat non-## as a start
      };
      const clean = (t) => t.replace(/^##/, "").replace(/^Ġ/, "").replace(/^▁/, "");

      const groups = [];
      let current = null;
      for (let i = 0; i < (tokens?.length || 0); i++) {
        const tok = String(tokens[i]);
        if (isSpecial(tok)) continue;
        if (!current || isStartOfWord(tok)) {
          if (current && current.idxs.length) groups.push(current);
          current = { label: clean(tok), idxs: [i] };
        } else {
          current.label += clean(tok);
          current.idxs.push(i);
        }
      }
      if (current && current.idxs.length) groups.push(current);
      setGroups(groups);

      // Build PCA on word-level pooled embeddings (exclude specials); fixed basis until text changes
      let pts = [];
      try {
        const W = groups.length;
        if (W >= 3 && Array.isArray(embeddings) && embeddings.length === tokens.length) {
          // Pool embeddings per word
          const wordVecs = groups.map((g) => {
            const len = embeddings[0]?.length || 0;
            const acc = new Float32Array(len);
            for (const i of g.idxs) {
              const v = embeddings[i] || [];
              for (let k = 0; k < len; k++) acc[k] += v[k] || 0;
            }
            for (let k = 0; k < len; k++) acc[k] /= g.idxs.length || 1;
            return Array.from(acc);
          });
          const p = new PCAClass(wordVecs, { center: true, scale: false });
          setPcaModel(p);
          const proj = p.predict(wordVecs, { nComponents: 3 });
          const coords = proj.to2DArray ? proj.to2DArray() : proj;
          pts = coords.map((c, i) => ({
            x: c?.[0] ?? 0,
            y: c?.[1] ?? 0,
            z: c?.[2] ?? 0,
            label: groups[i].label.slice(0, 24),
            color: groups[i].idxs.length > 1 ? "darkcyan" : undefined,
          }));

          // Add [CLS] projection (do not include in PCA fit)
          const clsIndex = tokens.findIndex((t) => String(t) === "[CLS]");
          if (clsIndex >= 0 && baseEmbeds?.[clsIndex]) {
            try {
              const projCls = p.predict([Array.from(baseEmbeds[clsIndex])], { nComponents: 3 });
              const c = projCls.to2DArray ? projCls.to2DArray()[0] : projCls[0];
              pts.push({ x: c?.[0] ?? 0, y: c?.[1] ?? 0, z: c?.[2] ?? 0, label: "[CLS]", color: "#888" });
            } catch {}
          }
        }
      } catch (e) {
        // ignore PCA failures
      }
      setPoints3d(pts);

      setStatus("ready");
      setMessage("Done");
    } catch (e) {
      console.error(e);
      setStatus("error");
      setMessage(e?.message || "Failed");
      setError("Failed to compute attention. See console.");
    }
  }

  async function handleAttentionChange(sel) {
    setAttnSel(sel);
    if (!tokens || !baseEmbeds || !groups || !pcaModel) return;
    if (sel === "none") {
      // Recompute base word points from stored PCA model
      try {
        const wordVecs = groups.map((g) => {
          const len = baseEmbeds[0]?.length || 0;
          const acc = new Float32Array(len);
          for (const i of g.idxs) {
            const v = baseEmbeds[i] || [];
            for (let k = 0; k < len; k++) acc[k] += v[k] || 0;
          }
          for (let k = 0; k < len; k++) acc[k] /= g.idxs.length || 1;
          return Array.from(acc);
        });
        const proj = pcaModel.predict(wordVecs, { nComponents: 3 });
        const coords = proj.to2DArray ? proj.to2DArray() : proj;
        setPoints3d(coords.map((c, i) => ({ x: c?.[0] ?? 0, y: c?.[1] ?? 0, z: c?.[2] ?? 0, label: groups[i].label.slice(0,24), color: groups[i].idxs.length>1?"darkcyan":undefined })));
        setAttnMatrix(null);
      } catch {}
      return;
    }

    // Fetch attention for selected head/mean
    const head = sel === "mean" ? "mean" : Number(sel.replace(/^h/, ""));
    try {
      const { attention } = await computeAttention(text, { head });
      setAttnMatrix(attention);
      // Create contextualized token embeddings: E' = A * E
      const T = tokens.length;
      const D = baseEmbeds[0]?.length || 0;
      const ctx = Array.from({ length: T }, () => new Float32Array(D));
      for (let i = 0; i < T; i++) {
        const row = attention[i] || [];
        for (let j = 0; j < T; j++) {
          const w = row[j] || 0;
          if (!w) continue;
          const vj = baseEmbeds[j] || [];
          for (let k = 0; k < D; k++) ctx[i][k] += w * (vj[k] || 0);
        }
      }
      // Pool per word and project with existing PCA
      const wordVecs = groups.map((g) => {
        const acc = new Float32Array(D);
        for (const i of g.idxs) {
          const v = ctx[i] || [];
          for (let k = 0; k < D; k++) acc[k] += v[k] || 0;
        }
        for (let k = 0; k < D; k++) acc[k] /= g.idxs.length || 1;
        return Array.from(acc);
      });
      const proj = pcaModel.predict(wordVecs, { nComponents: 3 });
      const coords = proj.to2DArray ? proj.to2DArray() : proj;
      const newPts = coords.map((c, i) => ({ x: c?.[0] ?? 0, y: c?.[1] ?? 0, z: c?.[2] ?? 0, label: groups[i].label.slice(0,24), color: groups[i].idxs.length>1?"darkcyan":undefined }));
      // Also project contextualized [CLS] if present (using ctx row)
      const clsIndex2 = tokens.findIndex((t) => String(t) === "[CLS]");
      if (clsIndex2 >= 0 && ctx?.[clsIndex2]) {
        try {
          const projCls = pcaModel.predict([Array.from(ctx[clsIndex2])], { nComponents: 3 });
          const c = projCls.to2DArray ? projCls.to2DArray()[0] : projCls[0];
          newPts.push({ x: c?.[0] ?? 0, y: c?.[1] ?? 0, z: c?.[2] ?? 0, label: "[CLS]", color: "#888" });
        } catch {}
      }
      setPoints3d(newPts);
    } catch (e) {
      console.error(e);
    }
  }

  return (
    <div className="panel">
      <div className="row">
        <textarea
          className="input"
          rows={4}
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Type some text…"
        />
      </div>

      <div className="row row--actions">
        <button className="btn btn--primary" onClick={handleTokenize} disabled={status === "running"}>
          {status === "running" ? "Working…" : "Tokenization"}
        </button>
        {tokens && (
          <>
            <label htmlFor="attn-sel" style={{ marginLeft: 12, color: "#a0a7b5" }}>Attention:</label>
            <select id="attn-sel" className="input" style={{ width: 200, marginLeft: 6 }} value={attnSel} onChange={(e) => handleAttentionChange(e.target.value)}>
              <option value="none">None</option>
              <option value="mean">Mean</option>
              {Array.from({ length: numHeads || 0 }, (_, i) => (
                <option key={i} value={`h${i}`}>{`Head ${i + 1}`}</option>
              ))}
            </select>
          </>
        )}
        <span className={`status status--${status}`}>{message}</span>
      </div>

      {error && <div className="alert alert--error">{error}</div>}

      {tokens && attnMatrix && (
        <div className="result">
          <div className="result__preview">
            <strong>Tokens + attention:</strong>
            <AttentionArcs tokens={tokens} attnMatrix={attnMatrix} />
          </div>
        </div>
      )}

      {points3d?.length >= 3 && (
        <div className="result">
          <Embedding3D
            points={points3d}
            width={640}
            height={360}
            title="Token Attention Rows — PCA (3D)"
          />
        </div>
      )}
    </div>
  );
}
