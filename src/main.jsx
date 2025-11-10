import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App.jsx";
import "./styles.css";

const container = document.getElementById("root");
const root = createRoot(container);
root.render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
);

import { computeAttention } from "./utils/computeAttention.js";

// Example usage (uncomment to test in Node ESM):
(async () => {
  const { tokens, attention } = await computeAttention();
  console.log("Tokens:", tokens);
  console.log("Attention matrix:");
  console.table(attention);
  // console.log(attention.map((row) => row.map((v) => v.toFixed(4)).join(" ")));
})();
