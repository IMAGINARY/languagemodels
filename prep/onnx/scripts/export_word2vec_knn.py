"""
Export word2vec-style embeddings to ONNX:
 - *_embed.onnx: lookup normalized embedding for a token id
 - *_token_knn.onnx: nearest tokens for a token id
 - *_vector_knn.onnx: nearest tokens for an arbitrary vector

The exporter supports two sources:
 - local word2vec/KeyedVectors files
 - cached transformer token embedding tables, used for the locale presets
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from gensim.models import KeyedVectors
from transformers import AutoModel, AutoTokenizer

from model_presets import (
    DOWNLOADS_DIR,
    TARGET_LANGUAGES,
    TRANSFORMER_MODEL_PRESETS,
    WORD2VEC_MODEL_PRESETS,
)

VOCAB_LIMIT = 50000
K = 6
OPSET = 17
DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "build"


class TokenEmbedding(torch.nn.Module):
    def __init__(self, emb):
        super().__init__()
        self.register_buffer("emb", emb)

    def forward(self, token_id: torch.Tensor):
        return self.emb.index_select(0, token_id.view(-1))


class TokenKNN(torch.nn.Module):
    def __init__(self, emb_norm, k=5):
        super().__init__()
        self.register_buffer("emb_norm", emb_norm)
        self.k = k

    def forward(self, token_id: torch.Tensor):
        query = self.emb_norm.index_select(0, token_id.view(-1))
        query = F.normalize(query, p=2, dim=1)
        sims = torch.matmul(query, self.emb_norm.t()).squeeze(0)
        top_vals, top_idx = torch.topk(sims, self.k, dim=0)
        return top_idx, top_vals


class VectorKNN(torch.nn.Module):
    def __init__(self, emb_norm, k=5):
        super().__init__()
        self.register_buffer("emb_norm", emb_norm)
        self.k = k

    def forward(self, query_emb: torch.Tensor):
        query = F.normalize(query_emb, p=2, dim=1)
        sims = torch.matmul(query, self.emb_norm.t()).squeeze(0)
        top_vals, top_idx = torch.topk(sims, self.k, dim=0)
        return top_idx, top_vals


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export ONNX lookup/KNN models from local word2vec vectors."
    )
    parser.add_argument(
        "--language",
        choices=TARGET_LANGUAGES,
        help="Export one configured language preset.",
    )
    parser.add_argument(
        "--all-languages",
        action="store_true",
        help="Export every configured language preset.",
    )
    parser.add_argument("--model-path", help="Path to a local word2vec file.")
    parser.add_argument("--model-id", help="Transformer model id for embedding-table exports.")
    parser.add_argument("--tokenizer-id", help="Tokenizer id for transformer exports.")
    parser.add_argument("--basename", help="Output basename.")
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR.as_posix(),
        help="Directory where ONNX and metadata files are written.",
    )
    parser.add_argument(
        "--vocab-limit",
        type=int,
        default=VOCAB_LIMIT,
        help="Cap vocabulary size to keep artifacts small. Use 0 for full vocab.",
    )
    parser.add_argument("--k", type=int, default=K, help="Number of nearest neighbors.")
    parser.add_argument("--opset", type=int, default=OPSET, help="ONNX opset version.")
    parser.add_argument(
        "--sample-token",
        action="append",
        dest="sample_tokens",
        help="Token to include in the sample vectors JSON. Can be repeated.",
    )
    parser.add_argument(
        "--text",
        action="store_true",
        help="Load the word2vec file as text instead of binary.",
    )
    parser.add_argument(
        "--source",
        choices=("transformer", "word2vec"),
        help="Embedding source type. Defaults to the preset source.",
    )
    args = parser.parse_args()

    if args.all_languages and any(
        value is not None
        for value in (
            args.language,
            args.model_path,
            args.model_id,
            args.tokenizer_id,
            args.basename,
            args.source,
        )
    ):
        parser.error("--all-languages cannot be combined with single-export overrides.")

    if (
        not args.all_languages
        and args.language is None
        and args.model_path is None
        and args.model_id is None
    ):
        args.language = "en"

    return args


def build_export_config(args, language):
    preset = WORD2VEC_MODEL_PRESETS.get(language, {})
    transformer_preset = TRANSFORMER_MODEL_PRESETS.get(language, {})
    source = args.source or preset.get("source") or ("word2vec" if args.model_path else "transformer")
    model_path = Path(args.model_path) if args.model_path else preset.get("model_path")
    if model_path is not None and not model_path.is_absolute():
        model_path = DOWNLOADS_DIR / model_path
    model_id = args.model_id or preset.get("model_id") or transformer_preset.get("model_id")
    tokenizer_id = args.tokenizer_id or preset.get("tokenizer_id") or transformer_preset.get("tokenizer_id") or model_id
    basename = args.basename or preset.get("basename") or "word2vec"
    sample_tokens = args.sample_tokens or preset.get("sample_tokens") or []
    binary = not args.text if args.model_path else preset.get("binary", True)

    return {
        "language": language or "custom",
        "source": source,
        "model_path": model_path,
        "model_id": model_id,
        "tokenizer_id": tokenizer_id,
        "basename": basename,
        "sample_tokens": sample_tokens,
        "binary": binary,
    }


def load_transformer_vocab_and_embeddings(config, vocab_limit):
    model = AutoModel.from_pretrained(config["model_id"], local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(
        config["tokenizer_id"],
        local_files_only=True,
    )
    emb = model.get_input_embeddings().weight.detach().float().cpu()
    vocab_size = emb.shape[0]
    limit = min(vocab_limit, vocab_size) if vocab_limit else vocab_size
    token_ids = list(range(limit))
    vocab = tokenizer.convert_ids_to_tokens(token_ids)
    emb = emb[:limit]
    return vocab, F.normalize(emb, p=2, dim=1)


def load_word2vec_vocab_and_embeddings(config, vocab_limit):
    if config["model_path"] is None or not config["model_path"].exists():
        raise FileNotFoundError(
            f"Missing word2vec source for {config['language']}: {config['model_path']}"
        )
    kv = KeyedVectors.load_word2vec_format(
        config["model_path"].as_posix(),
        binary=config["binary"],
    )
    vocab = kv.index_to_key[:vocab_limit] if vocab_limit else kv.index_to_key
    emb = torch.tensor(kv.vectors[: len(vocab)], dtype=torch.float32)
    return vocab, F.normalize(emb, p=2, dim=1)


def export_language(config, out_dir, vocab_limit, k, opset):
    out_dir.mkdir(parents=True, exist_ok=True)

    if config["source"] == "transformer":
        vocab, emb_norm = load_transformer_vocab_and_embeddings(config, vocab_limit)
    else:
        vocab, emb_norm = load_word2vec_vocab_and_embeddings(config, vocab_limit)

    idx_map = {word: index for index, word in enumerate(vocab)}

    embed_module = TokenEmbedding(emb_norm).eval()
    token_knn = TokenKNN(emb_norm, k=k).eval()
    vector_knn = VectorKNN(emb_norm, k=k).eval()

    dummy_token = (torch.tensor([0], dtype=torch.long),)
    dummy_vec = (torch.randn(1, emb_norm.shape[1]),)

    embed_path = out_dir / f"{config['basename']}_embed.onnx"
    token_path = out_dir / f"{config['basename']}_token_knn.onnx"
    vector_path = out_dir / f"{config['basename']}_vector_knn.onnx"
    vocab_path = out_dir / f"{config['basename']}_vocab.json"
    sample_path = out_dir / f"{config['basename']}_sample_vectors.json"
    metadata_path = out_dir / f"{config['basename']}_metadata.json"

    torch.onnx.export(
        embed_module,
        dummy_token,
        embed_path.as_posix(),
        input_names=["token_id"],
        output_names=["embedding"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes={"token_id": {0: "batch"}},
    )
    torch.onnx.export(
        token_knn,
        dummy_token,
        token_path.as_posix(),
        input_names=["token_id"],
        output_names=["top_indices", "top_scores"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes={"token_id": {0: "batch"}},
    )
    torch.onnx.export(
        vector_knn,
        dummy_vec,
        vector_path.as_posix(),
        input_names=["query_emb"],
        output_names=["top_indices", "top_scores"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes={"query_emb": {0: "batch"}},
    )

    with vocab_path.open("w") as f:
        json.dump(vocab, f, ensure_ascii=False)

    sample_vectors = {}
    for token in config["sample_tokens"]:
        index = idx_map.get(token)
        if index is None:
            print(f"Skip sample '{token}' -> not in vocab")
            continue
        sample_vectors[token] = {"id": index, "vector": emb_norm[index].tolist()}

    with sample_path.open("w") as f:
        json.dump(sample_vectors, f, ensure_ascii=False, indent=2)

    metadata = {
        "language": config["language"],
        "source": config["source"],
        "basename": config["basename"],
        "embedding_dim": int(emb_norm.shape[1]),
        "vocab_size": len(vocab),
        "vocab_limit": vocab_limit,
        "k": k,
        "opset": opset,
        "files": {
            "embed": embed_path.name,
            "token_knn": token_path.name,
            "vector_knn": vector_path.name,
            "vocab": vocab_path.name,
            "sample_vectors": sample_path.name,
        },
    }
    if config["source"] == "transformer":
        metadata["model_id"] = config["model_id"]
        metadata["tokenizer_id"] = config["tokenizer_id"]
    else:
        metadata["model_path"] = config["model_path"].as_posix()
        metadata["binary"] = config["binary"]
    with metadata_path.open("w") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[{config['language']}] Exported: {embed_path}")
    print(f"[{config['language']}] Exported: {token_path}")
    print(f"[{config['language']}] Exported: {vector_path}")
    print(f"[{config['language']}] Exported vocab: {vocab_path} (size={len(vocab)})")
    print(f"[{config['language']}] Exported: {sample_path}")
    print(f"[{config['language']}] Exported: {metadata_path}")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    vocab_limit = None if args.vocab_limit == 0 else args.vocab_limit
    languages = TARGET_LANGUAGES if args.all_languages else (args.language,)

    for language in languages:
        config = build_export_config(args, language)
        export_language(config, out_dir, vocab_limit=vocab_limit, k=args.k, opset=args.opset)


if __name__ == "__main__":
    main()
