import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import "./styles.css";
import {
  mostSimilarTokensToToken,
  mostSimilarTokensToVector,
} from "./utils/similarTokens.ts";
import { computeAttention } from "./utils/computeAttention.js";

const container = document.getElementById("root");
const root = createRoot(container);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

// Example usage (uncomment to test in Node ESM):
(async () => {
  const { tokens, attention } = await computeAttention();
  console.log("Tokens:", tokens);
  console.log("Attention matrix:");
  console.table(attention);
  // console.log(attention.map((row) => row.map((v) => v.toFixed(4)).join(" ")));
})();

////////////////////////////////////////////
/// TEST for similarTokens utilities
////////////////////////////////////////////
(async () => {
  const token = "king";
  const byToken = await mostSimilarTokensToToken(token);
  console.log("mostSimilarTokensToToken", byToken);

  // Load a sample vector (exported by prep/onnx/export_minilm_token_knn.py)
  let sampleVector;
  try {
    const resp = await fetch(
      new URL(
        /* @vite-ignore */ "./models/minilm_sample_vectors.json",
        import.meta.url
      )
    );
    if (resp.ok) {
      const json = await resp.json();
      sampleVector = json[token]?.vector;
      if (!sampleVector) {
        console.warn(
          `Sample vector for "${token}" missing in minilm_sample_vectors.json`
        );
      }
    } else {
      console.warn("minilm_sample_vectors.json not found; rerun export script");
    }
  } catch (err) {
    console.warn("Unable to load sample vectors; skipping vector test", err);
    return;
  }
  if (!sampleVector) return;

  const vecModelUrl = new URL(
    "./models/minilm_vector_knn.onnx",
    import.meta.url
  ).href;
  const byVector = await mostSimilarTokensToVector(sampleVector, {
    onnxUrl: vecModelUrl,
  });
  console.log("mostSimilarTokensToVector", byVector);
})();
