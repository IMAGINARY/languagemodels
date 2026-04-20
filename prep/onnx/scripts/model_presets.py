from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
ONNX_DIR = SCRIPT_DIR.parent
DOWNLOADS_DIR = ONNX_DIR / "downloads"
BUILD_DIR = ONNX_DIR / "build"


TARGET_LANGUAGES = ("en", "fr", "de", "it")

TRANSFORMER_MODEL_PRESETS = {
    "en": {
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "tokenizer_id": "sentence-transformers/all-MiniLM-L6-v2",
        "basename": "minilm_en",
        "sample_tokens": ["king", "queen", "city"],
    },
    "fr": {
        "model_id": "camembert-base",
        "tokenizer_id": "camembert-base",
        "basename": "camembert_fr",
        "sample_tokens": ["roi", "reine", "ville"],
    },
    "de": {
        "model_id": "bert-base-german-cased",
        "tokenizer_id": "bert-base-german-cased",
        "basename": "bert_de",
        "sample_tokens": ["König", "Königin", "Stadt"],
    },
    "it": {
        "model_id": "dbmdz/bert-base-italian-cased",
        "tokenizer_id": "dbmdz/bert-base-italian-cased",
        "basename": "bert_it",
        "sample_tokens": ["re", "regina", "città"],
    },
}

WORD2VEC_MODEL_PRESETS = {
    "en": {
        "source": "transformer",
        "model_id": "sentence-transformers/all-MiniLM-L6-v2",
        "tokenizer_id": "sentence-transformers/all-MiniLM-L6-v2",
        "basename": "word2vec_en",
        "sample_tokens": ["king", "queen", "city"],
    },
    "fr": {
        "source": "transformer",
        "model_id": "camembert-base",
        "tokenizer_id": "camembert-base",
        "basename": "word2vec_fr",
        "sample_tokens": ["roi", "reine", "ville"],
    },
    "de": {
        "source": "transformer",
        "model_id": "bert-base-german-cased",
        "tokenizer_id": "bert-base-german-cased",
        "basename": "word2vec_de",
        "sample_tokens": ["König", "Königin", "Stadt"],
    },
    "it": {
        "source": "transformer",
        "model_id": "dbmdz/bert-base-italian-cased",
        "tokenizer_id": "dbmdz/bert-base-italian-cased",
        "basename": "word2vec_it",
        "sample_tokens": ["re", "regina", "città"],
    },
}
