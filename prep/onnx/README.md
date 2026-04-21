## Layout

- `scripts/`: tracked export scripts and preset config
- `downloads/`: local source files and downloaded inputs, ignored by git
- `build/`: generated ONNX and JSON artifacts, ignored by git

## Examples

```bash
./prep/onnx/.venv/bin/python prep/onnx/scripts/export_minilm_token_knn.py --all-languages --out-dir prep/onnx/build/i18n
./prep/onnx/.venv/bin/python prep/onnx/scripts/export_word2vec_knn.py --all-languages --out-dir prep/onnx/build/i18n
./prep/onnx/.venv/bin/python prep/onnx/scripts/export_attention_encoder.py --all-languages --out-dir prep/onnx/build/i18n_attention
```

## Full Attention Export

Use `export_attention_encoder.py` to build self-contained local encoder bundles for `ModelDemo2`.

Each language is exported into its own folder under `prep/onnx/build/i18n_attention/<basename>/` and contains:

- `onnx/model.onnx`: transformer encoder forward graph
- `metadata.json`: export metadata and output descriptions
- local tokenizer assets written by `tokenizer.save_pretrained(...)`
- local model config written by `config.save_pretrained(...)`

The exported ONNX model takes:

- `input_ids`
- `attention_mask`

It returns:

- `last_hidden_state`
- `last_attention`

This exporter uses `local_files_only=True`, so it depends on the relevant Hugging Face model and tokenizer already being cached in the local environment. It does not fetch models at app runtime.

To serve the exported bundles inside the app, copy them into `public/models/i18n_attention/` so `@huggingface/transformers` can load them through its local model path:

```bash
mkdir -p public/models
cp -R prep/onnx/build/i18n_attention public/models/
```
