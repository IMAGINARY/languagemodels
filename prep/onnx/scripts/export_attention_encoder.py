import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from model_presets import TARGET_LANGUAGES, TRANSFORMER_MODEL_PRESETS

OPSET = 17
DEFAULT_OUT_DIR = Path(__file__).resolve().parent.parent / "build" / "i18n_attention"


class AttentionEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.eval()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            return_dict=True,
        )
        return outputs.last_hidden_state, outputs.attentions[-1]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Export self-contained ONNX encoder models with last-layer attention "
            "plus local tokenizer assets for locale-specific transformer presets."
        )
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
        help="Output basename. Defaults to the preset basename.",
    )
    parser.add_argument(
        "--out-dir",
        default=DEFAULT_OUT_DIR.as_posix(),
        help="Directory where exported model folders are written.",
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=OPSET,
        help="ONNX opset version.",
    )
    parser.add_argument(
        "--sample-text",
        action="append",
        dest="sample_texts",
        help="Sample text stored in metadata and used for validation. Can be repeated.",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=64,
        help="Maximum sequence length for the export sample input.",
    )
    args = parser.parse_args()

    if args.all_languages and any(
        value is not None
        for value in (
            args.language,
            args.model_id,
            args.tokenizer_id,
            args.basename,
            args.sample_texts,
        )
    ):
        parser.error("--all-languages cannot be combined with single-export overrides.")

    if not args.all_languages and args.language is None and args.model_id is None:
        args.language = "en"

    return args


def build_export_config(args, language):
    preset = TRANSFORMER_MODEL_PRESETS.get(language, {})
    model_id = args.model_id or preset.get("model_id") or "sentence-transformers/all-MiniLM-L6-v2"
    tokenizer_id = args.tokenizer_id or preset.get("tokenizer_id") or model_id
    basename = args.basename or preset.get("basename") or "encoder"
    sample_texts = args.sample_texts or [
        "Transformers are really cool for embeddings."
    ]

    return {
        "language": language or "custom",
        "model_id": model_id,
        "tokenizer_id": tokenizer_id,
        "basename": basename,
        "sample_texts": sample_texts,
    }


def clean_output_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def save_metadata(metadata_path: Path, payload: dict):
    with metadata_path.open("w") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def export_language(config, out_dir: Path, opset: int, max_seq_length: int):
    model_dir = out_dir / config["basename"]
    clean_output_dir(model_dir)
    onnx_dir = model_dir / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        config["tokenizer_id"],
        local_files_only=True,
    )
    hf_config = AutoConfig.from_pretrained(
        config["model_id"],
        local_files_only=True,
    )
    hf_config.output_attentions = True
    hf_config.return_dict = True

    model = AutoModel.from_pretrained(
        config["model_id"],
        config=hf_config,
        local_files_only=True,
    ).eval()
    wrapper = AttentionEncoder(model).eval()

    tokenizer.save_pretrained(model_dir.as_posix())
    hf_config.save_pretrained(model_dir.as_posix())

    sample_text = config["sample_texts"][0]
    encoded = tokenizer(
        sample_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_length,
    )
    dummy_input_ids = encoded["input_ids"].to(dtype=torch.long)
    dummy_attention_mask = encoded["attention_mask"].to(dtype=torch.long)

    model_path = onnx_dir / "model.onnx"
    metadata_path = model_dir / "metadata.json"

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_input_ids, dummy_attention_mask),
            model_path.as_posix(),
            input_names=["input_ids", "attention_mask"],
            output_names=["last_hidden_state", "last_attention"],
            opset_version=opset,
            do_constant_folding=True,
            dynamic_axes={
                "input_ids": {0: "batch", 1: "seq"},
                "attention_mask": {0: "batch", 1: "seq"},
                "last_hidden_state": {0: "batch", 1: "seq"},
                "last_attention": {0: "batch", 2: "seq", 3: "seq"},
            },
        )

    saved_files = sorted(
        path.name for path in model_dir.iterdir() if path.is_file()
    )
    metadata = {
        "language": config["language"],
        "model_id": config["model_id"],
        "tokenizer_id": config["tokenizer_id"],
        "basename": config["basename"],
        "opset": opset,
        "max_seq_length": max_seq_length,
        "sample_texts": config["sample_texts"],
        "files": {
            "model": "onnx/model.onnx",
            "metadata": "metadata.json",
            "tokenizer_dir": ".",
        },
        "outputs": {
            "last_hidden_state": {
                "description": "Final token representations",
            },
            "last_attention": {
                "description": "Last-layer attention tensor shaped [batch, heads, seq, seq]",
            },
        },
        "saved_files": saved_files,
    }
    save_metadata(metadata_path, metadata)

    print(f"[{config['language']}] Exported encoder folder: {model_dir}")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    languages = TARGET_LANGUAGES if args.all_languages else (args.language,)

    for language in languages:
        config = build_export_config(args, language)
        export_language(
            config,
            out_dir=out_dir,
            opset=args.opset,
            max_seq_length=args.max_seq_length,
        )


if __name__ == "__main__":
    main()
