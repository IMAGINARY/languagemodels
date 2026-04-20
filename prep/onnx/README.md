## Layout

- `scripts/`: tracked export scripts and preset config
- `downloads/`: local source files and downloaded inputs, ignored by git
- `build/`: generated ONNX and JSON artifacts, ignored by git

## Examples

```bash
./prep/onnx/.venv/bin/python prep/onnx/scripts/export_minilm_token_knn.py --all-languages --out-dir prep/onnx/build/i18n
./prep/onnx/.venv/bin/python prep/onnx/scripts/export_word2vec_knn.py --all-languages --out-dir prep/onnx/build/i18n
```
