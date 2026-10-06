"""JSON Schema for the model-facing ``bayes_score`` tool. ``description`` is shown verbatim."""

from __future__ import annotations

from .scorer import DEFAULT_LR_TABLE

TOOL_NAME = "bayes_score"
TOOLSET = "bayes_check"

MAX_CLAIMS = 40
MAX_EVIDENCE = 12
MAX_TEXT = 1000

BAYES_SCORE = {
    "name": TOOL_NAME,
    "description": (
        "Score the factual or quantitative claims of your draft answer with deterministic Bayesian "
        "log-odds updating BEFORE you finalize. Pass every checkable claim with a prior (0-1, your "
        "base-rate belief before evidence) and the evidence you actually have, typed by source. "
        "Returns a posterior and an action per claim: state_plainly (>= plain threshold), qualify "
        "(hedge it explicitly), or remove (cut it or say it is unverified). Rewrite your answer to "
        "honor every action. Evidence you did not actually observe must not be listed. "
        "source_type values: " + ", ".join(sorted(DEFAULT_LR_TABLE)) + "."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "maxItems": MAX_CLAIMS,
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Short id unique within this call, e.g. c1."},
                        "text": {"type": "string", "description": "The claim exactly as it will appear."},
                        "claim_type": {
                            "type": "string",
                            "description": "Category, e.g. statistic, date, causal, definition, attribution, "
                                           "code_behavior, recommendation.",
                        },
                        "prior": {
                            "type": "number", "minimum": 0, "maximum": 1,
                            "description": "Belief before evidence. Omit to use the learned base rate.",
                        },
                        "evidence": {
                            "type": "array",
                            "maxItems": MAX_EVIDENCE,
                            "items": {
                                "type": "object",
                                "properties": {
                                    "source": {"type": "string", "description": "URL, file path, tool name, or citation."},
                                    "source_type": {"type": "string", "enum": sorted(DEFAULT_LR_TABLE)},
                                    "stance": {"type": "string", "enum": ["supports", "contradicts"],
                                               "description": "Default supports."},
                                    "lr_override": {
                                        "type": "number", "exclusiveMinimum": 0,
                                        "description": "Optional explicit likelihood ratio; clamped to [0.05, 20].",
                                    },
                                },
                                "required": ["source", "source_type"],
                            },
                        },
                    },
                    "required": ["id", "text", "claim_type", "evidence"],
                },
            },
        },
        "required": ["claims"],
    },
}
