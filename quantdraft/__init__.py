"""QuantDraft: lossless self-speculative decoding with quantized drafts (MLX)."""

from .controllers import CostModelGamma, FixedGamma, HeuristicGamma, make_controller
from .costmodel import CostModel, expected_tokens, optimal_gamma, speedup
from .drafts import load_draft, nested_draft, rtn_draft, weight_bytes
from .engine import (
    GenerationStats,
    autoregressive_generate,
    speculative_generate,
    verify_greedy,
    verify_match,
    verify_rejection,
)

__all__ = [
    "CostModel", "CostModelGamma", "FixedGamma", "GenerationStats", "HeuristicGamma",
    "autoregressive_generate", "expected_tokens", "load_draft", "make_controller",
    "nested_draft", "optimal_gamma", "rtn_draft", "speculative_generate", "speedup",
    "verify_greedy", "verify_match", "verify_rejection", "weight_bytes",
]
__version__ = "0.1.0"
