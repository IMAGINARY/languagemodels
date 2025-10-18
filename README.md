# AI Exhibits — Word Embeddings (transformers.js + React)

An educational web app that demonstrates core LLM concepts directly in the browser. It uses React for the UI and [`@huggingface/transformers`](https://github.com/huggingface/transformers.js) to run inference client‑side (no server required). The initial exhibit shows sentence embeddings using the `Xenova/all-MiniLM-L6-v2` model.

## Features

- Client‑side inference with `@huggingface/transformers`
- React UI with a ready‑to‑extend structure
- Minimal Vite setup for fast dev

## Getting started

1. Install dependencies:

   ```bash
   npm install
   ```

2. Start the dev server:

   ```bash
   npm run dev
   ```

3. Open the URL shown in the terminal (usually `http://localhost:5173`).

Notes:
- The first model load will download several files (~tens of MB) and cache them in the browser (IndexedDB) for subsequent runs.
- Everything runs locally in your browser; no backend is involved.

## Offline mode (recommended)

This project can run fully offline. Do this once while online:

1) Ensure ONNX runtime assets are available locally (optional)

- Remote mode requires no action. If you plan to host the ONNX runtime files locally for offline use, copy the runtime assets manually:

  ```bash
  mkdir -p public/transformers
  cp -R node_modules/@huggingface/transformers/dist/* public/transformers/
  ```

2) Download the model files and place them under `public/models/`

- Create: `public/models/Xenova/all-MiniLM-L6-v2/resolve/main/`
- From https://huggingface.co/Xenova/all-MiniLM-L6-v2/tree/main download:
  - `config.json`
  - `tokenizer.json`
  - `tokenizer_config.json`
  - `special_tokens_map.json` (optional)
  - `preprocessor_config.json` (optional)
  - `onnx/` folder (e.g., `onnx/model.onnx`, plus any `model.onnx_data` files)

Resulting structure should look like:

```
public/models/
  Xenova/all-MiniLM-L6-v2/resolve/main/
    config.json
    tokenizer.json
    tokenizer_config.json
    special_tokens_map.json
    onnx/
      model.onnx
      model.onnx_data          # if present
```

3) Start the app (now works offline):

```bash
npm run dev
```

The app is configured to prefer the local model path and local ONNX runtime assets.

**Using Git LFS to fetch model files**

If you have Git and Git LFS available, this is a quick way to download the model with large files correctly:

```bash
# 1) Ensure Git LFS is installed
#    macOS (brew):   brew install git-lfs
#    Debian/Ubuntu:  sudo apt-get install git-lfs
#    Windows:        https://git-lfs.com
git lfs install

# 2) Clone the model repo (includes large ONNX files via LFS)
git clone https://huggingface.co/Xenova/all-MiniLM-L6-v2

# 3) Copy files into the app's offline model path
mkdir -p public/models/Xenova/all-MiniLM-L6-v2/resolve/main
cp -R all-MiniLM-L6-v2/* public/models/Xenova/all-MiniLM-L6-v2/resolve/main/

# 4) (Optional) Remove the cloned repo after copying
rm -rf all-MiniLM-L6-v2
```

## Project structure

```
wordembeddings3/
├─ index.html
├─ package.json
├─ vite.config.js
├─ public/
│  └─ favicon.svg
├─ src/
│  ├─ App.jsx
│  ├─ main.jsx
│  ├─ styles.css
│  └─ components/
│     └─ ModelDemo.jsx
└─ README.md
```

## Extending the app

- Add new exhibits under `src/components/` and register them in `src/App.jsx`.
- For other LLM demos (e.g., text classification or QA), use `pipeline(<task>, <model>)`. See the transformers.js docs for supported tasks and models.

## Troubleshooting

- If models fail to load, check the browser console for detailed logs.
- Some corporate networks block large file downloads; try another network if model files fail repeatedly.
- To clear cached models, clear the site data (IndexedDB) in your browser devtools.
- If using offline mode and you see 404s for `/models/...`, verify filenames and that the `resolve/main/` folder structure matches the repository.
