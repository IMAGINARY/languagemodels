import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

from model_presets import TARGET_LANGUAGES, TRANSFORMER_MODEL_PRESETS

K = 6
OPSET = 17
DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "build"


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


class TokenEmbed(torch.nn.Module):
    def __init__(self, emb_norm):
        super().__init__()
        self.register_buffer("emb_norm", emb_norm)

    def forward(self, token_id: torch.Tensor):
        return self.emb_norm.index_select(0, token_id.view(-1))


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
        description="Export token embedding/KNN ONNX models for locale-specific transformer encoders."
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
    parser.add_argument(
        "--model-id",
        help="Override the Hugging Face model id for a single export.",
    )
    parser.add_argument(
        "--tokenizer-id",
        help="Override the tokenizer id for a single export. Defaults to --model-id.",
    )
    parser.add_argument(
        "--basename",
        help="Output basename. Defaults to the preset basename or 'minilm'.",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR.as_posix(),
        help="Directory where ONNX and metadata files are written.",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=K,
        help="Number of neighbors stored in KNN outputs.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=OPSET,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--sample-token",
        action="append",
        dest="sample_tokens",
        help="Token to include in the sample vectors JSON. Can be repeated.",
    )
    args = parser.parse_args()

    if args.all_languages and any(
        value is not None for value in (args.language, args.model_id, args.tokenizer_id, args.basename)
    ):
        parser.error("--all-languages cannot be combined with single-export overrides.")

    if not args.all_languages and args.language is None and args.model_id is None:
        args.language = "en"

    return args


def build_export_config(args, language):
    preset = TRANSFORMER_MODEL_PRESETS.get(language, {})
    model_id = args.model_id or preset.get("model_id") or "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer_id = args.tokenizer_id or preset.get("tokenizer_id") or model_id
    basename = args.basename or preset.get("basename") or "minilm"
    sample_tokens = args.sample_tokens or preset.get("sample_tokens") or []

    return {
        "language": language or "custom",
        "model_id": model_id,
        "tokenizer_id": tokenizer_id,
        "basename": basename,
        "sample_tokens": sample_tokens,
    }


def export_language(config, out_dir, k, opset):
    out_dir.mkdir(parents=True, exist_ok=True)

    base = AutoModel.from_pretrained(config["model_id"])
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_id"])
    emb_norm = F.normalize(
        base.get_input_embeddings().weight.detach().float().cpu(),
        p=2,
        dim=1,
    )

    token_module = TokenKNN(emb_norm, k=k).eval()
    vector_module = VectorKNN(emb_norm, k=k).eval()
    embed_module = TokenEmbed(emb_norm).eval()

    dummy_token_input = (torch.tensor([0], dtype=torch.long),)
    dummy_vec_input = (torch.randn(1, emb_norm.shape[1]),)

    embed_path = out_dir / f"{config['basename']}_embed.onnx"
    token_path = out_dir / f"{config['basename']}_token_knn.onnx"
    vector_path = out_dir / f"{config['basename']}_vector_knn.onnx"
    sample_path = out_dir / f"{config['basename']}_sample_vectors.json"
    metadata_path = out_dir / f"{config['basename']}_metadata.json"

    torch.onnx.export(
        embed_module,
        dummy_token_input,
        embed_path.as_posix(),
        input_names=["token_id"],
        output_names=["embedding"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes={"token_id": {0: "batch"}},
    )
    torch.onnx.export(
        token_module,
        dummy_token_input,
        token_path.as_posix(),
        input_names=["token_id"],
        output_names=["top_indices", "top_scores"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes={"token_id": {0: "batch"}},
    )
    torch.onnx.export(
        vector_module,
        dummy_vec_input,
        vector_path.as_posix(),
        input_names=["query_emb"],
        output_names=["top_indices", "top_scores"],
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes={"query_emb": {0: "batch"}},
    )

    sample_vectors = {}
    for token in config["sample_tokens"]:
        ids = tokenizer.encode(token, add_special_tokens=False)
        if len(ids) != 1:
            print(f"Skip sample '{token}' -> split into {ids}")
            continue
        token_id = int(ids[0])
        sample_vectors[token] = {"id": token_id, "vector": emb_norm[token_id].tolist()}

    with sample_path.open("w") as f:
        json.dump(sample_vectors, f, ensure_ascii=False, indent=2)

    metadata = {
        "language": config["language"],
        "model_id": config["model_id"],
        "tokenizer_id": config["tokenizer_id"],
        "basename": config["basename"],
        "embedding_dim": int(emb_norm.shape[1]),
        "vocab_size": int(emb_norm.shape[0]),
        "k": k,
        "opset": opset,
        "files": {
            "embed": embed_path.name,
            "token_knn": token_path.name,
            "vector_knn": vector_path.name,
            "sample_vectors": sample_path.name,
        },
    }
    with metadata_path.open("w") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"[{config['language']}] Exported: {embed_path}")
    print(f"[{config['language']}] Exported: {token_path}")
    print(f"[{config['language']}] Exported: {vector_path}")
    print(f"[{config['language']}] Exported: {sample_path}")
    print(f"[{config['language']}] Exported: {metadata_path}")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    languages = TARGET_LANGUAGES if args.all_languages else (args.language,)

    for language in languages:
        config = build_export_config(args, language)
        export_language(config, out_dir, k=args.k, opset=args.opset)


if __name__ == "__main__":
    main()
