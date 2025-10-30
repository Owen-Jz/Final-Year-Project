from __future__ import annotations

from typing import Dict

from . import config


def score_and_decide(ml_score: float, prnu_similarity: float, metadata_flag_score: float) -> Dict:
    # ensemble_score = w_ml * ml_score + w_prnu * (1 - prnu_similarity) + w_meta * metadata_flag_score
    ensemble_score = (
        config.W_ML * ml_score
        + config.W_PRNU * (1.0 - prnu_similarity)
        + config.W_META * metadata_flag_score
    )
    if ensemble_score >= 0.7:
        decision = "LIKELY_MANIPULATED"
    elif ensemble_score >= 0.4:
        decision = "SUSPECT"
    else:
        decision = "SAFE"
    return {"weighted_score": float(ensemble_score), "decision": decision}


