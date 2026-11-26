import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mi300_llm.config import LlamaConfig  # noqa: E402

# Same shape ratios as Llama-3.3-70B (GQA 8:1, llama3 RoPE), scaled down.
TINY = dict(
    hidden_size=128,
    intermediate_size=320,
    num_hidden_layers=3,
    num_attention_heads=8,
    num_key_value_heads=1,
    head_dim=16,
    vocab_size=300,
    rms_norm_eps=1e-5,
    rope_theta=500000.0,
    max_position_embeddings=512,
    rope_scaling={
        "rope_type": "llama3",
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 64,
    },
    tie_word_embeddings=False,
)


@pytest.fixture(scope="session")
def hf_tiny():
    from transformers import LlamaConfig as HFConfig
    from transformers import LlamaForCausalLM

    torch.manual_seed(0)
    model = LlamaForCausalLM(HFConfig(**TINY, attn_implementation="eager")).float().eval()
    # Random init gives near-uniform logits; scale up so argmax is meaningful.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" not in name:
                p.mul_(3.0)
    return model


@pytest.fixture(scope="session")
def tiny_cfg(hf_tiny):
    return LlamaConfig.from_hf_dict(hf_tiny.config.to_dict())


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"
