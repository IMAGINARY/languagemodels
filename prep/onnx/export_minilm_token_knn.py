import json
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# Hugging Face ID matching Xenova/all-MiniLM-L6-v2 weights
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
K = 5
OPSET = 17
OUTPATH = "minilm_token_knn.onnx"
OUTPATH_VEC = "minilm_vector_knn.onnx"
SAMPLE_TOKENS = ["king"]

# 1) Load model and grab token embedding matrix
base = AutoModel.from_pretrained(MODEL_ID)
emb = base.get_input_embeddings().weight.detach().float()      # [V, D]
emb = emb.cpu()

# 2) Normalise embeddings row-wise for cosine similarity
emb_norm = F.normalize(emb, p=2, dim=1)                        # [V, D]

class TokenKNN(torch.nn.Module):
    def __init__(self, emb_norm, k=5):
        super().__init__()
        self.register_buffer("emb_norm", emb_norm)             # [V, D]
        self.k = k

    def forward(self, token_id: torch.Tensor):
        # token_id: shape [1] (int64)
        query = self.emb_norm.index_select(0, token_id.view(-1))
        query = F.normalize(query, p=2, dim=1)                 # [1, D]
        sims = torch.matmul(query, self.emb_norm.t())          # [1, V]
        sims = sims.squeeze(0)                                 # [V]
        top_vals, top_idx = torch.topk(sims, self.k, dim=0)    # ([K], [K])
        return top_idx, top_vals

class VectorKNN(torch.nn.Module):
    def __init__(self, emb_norm, k=5):
        super().__init__()
        self.register_buffer("emb_norm", emb_norm)             # [V, D]
        self.k = k

    def forward(self, query_emb: torch.Tensor):
        # query_emb: shape [1, D], float32
        query = F.normalize(query_emb, p=2, dim=1)             # [1, D]
        sims = torch.matmul(query, self.emb_norm.t())          # [1, V]
        sims = sims.squeeze(0)                                 # [V]
        top_vals, top_idx = torch.topk(sims, self.k, dim=0)    # ([K], [K])
        return top_idx, top_vals

token_module = TokenKNN(emb_norm, k=K).eval()
vector_module = VectorKNN(emb_norm, k=K).eval()
# Tokenizer for optional sample vectors
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

dummy_token_input = (torch.tensor([0], dtype=torch.long),)
dummy_vec_input = (torch.randn(1, emb_norm.shape[1]),)

torch.onnx.export(
    token_module,
    dummy_token_input,
    OUTPATH,
    input_names=["token_id"],
    output_names=["top_indices", "top_scores"],
    opset_version=OPSET,
    do_constant_folding=True,
    dynamic_axes={"token_id": {0: "batch"}}
)

torch.onnx.export(
    vector_module,
    dummy_vec_input,
    OUTPATH_VEC,
    input_names=["query_emb"],
    output_names=["top_indices", "top_scores"],
    opset_version=OPSET,
    do_constant_folding=True,
    dynamic_axes={"query_emb": {0: "batch"}}
)

print(f"Exported: {OUTPATH}")
print(f"Exported: {OUTPATH_VEC}")

# Optional: dump a few normalized token vectors for JS tests
sample_vectors = {}
for tok in SAMPLE_TOKENS:
    ids = tokenizer.encode(tok, add_special_tokens=False)
    if len(ids) != 1:
        print(f"Skip sample '{tok}' -> split into {ids}")
        continue
    tid = ids[0]
    vec = emb_norm[tid].tolist()
    sample_vectors[tok] = {"id": tid, "vector": vec}

with open("minilm_sample_vectors.json", "w") as f:
    json.dump(sample_vectors, f)
print("Exported: minilm_sample_vectors.json")
