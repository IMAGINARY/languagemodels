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
  const [attnMatrix, setAttnMatrix] = useState(null); // number[][]
  const [points3d, setPoints3d] = useState([]); // for Embedding3D

  async function handleCompute() {
    try {
      setError(null);
      setStatus("running");
      setMessage("Computing tokenization + attention …");
      const { tokens, attention } = await computeAttention(text);
      setTokens(tokens);
      setAttnMatrix(attention);

      // Derive 3D points from attention rows via PCA (one arrow per token)
      let pts = [];
      try {
        if (attention?.length >= 3) {
          const X = attention.map((row) => row.map((v) => +v));
          const p = new PCAClass(X, { center: true, scale: false });
          const proj = p.predict(X, { nComponents: 3 });
          const coords = proj.to2DArray ? proj.to2DArray() : proj;
          pts = coords.map((c, i) => ({
            x: c?.[0] ?? 0,
            y: c?.[1] ?? 0,
            z: c?.[2] ?? 0,
            label: String(tokens?.[i] ?? `#${i + 1}`),
          }));
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
        <button
          className="btn btn--primary"
          onClick={handleCompute}
          disabled={status === "running"}
        >
          {status === "running" ? "Working…" : "Tokenization and attention"}
        </button>
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

