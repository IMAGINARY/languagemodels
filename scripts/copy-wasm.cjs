#!/usr/bin/env node
// Copies @xenova/transformers runtime assets (wasm/worker files) to public/transformers
// so the app can run without network access.

const fs = require('fs');
const path = require('path');

function ensureDir(p) {
  if (!fs.existsSync(p)) fs.mkdirSync(p, { recursive: true });
}

function copyDir(src, dest) {
  ensureDir(dest);
  const entries = fs.readdirSync(src, { withFileTypes: true });
  for (const e of entries) {
    const s = path.join(src, e.name);
    const d = path.join(dest, e.name);
    if (e.isDirectory()) {
      copyDir(s, d);
    } else {
      fs.copyFileSync(s, d);
    }
  }
}

try {
  const src = path.join(process.cwd(), 'node_modules', '@xenova', 'transformers', 'dist');
  const dest = path.join(process.cwd(), 'public', 'transformers');
  if (!fs.existsSync(src)) {
    console.warn('[copy-wasm] transformers dist not found at', src);
    process.exit(0);
  }
  ensureDir(dest);

  // Copy entire dist folder (small) so required .wasm/.js files are available
  copyDir(src, dest);
  console.log('[copy-wasm] Copied @xenova/transformers/dist -> public/transformers');
} catch (e) {
  console.warn('[copy-wasm] Failed to copy runtime assets:', e?.message || e);
}

