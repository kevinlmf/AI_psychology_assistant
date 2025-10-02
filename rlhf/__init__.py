"""RLHF package - Reward model and feedback collection"""

from .reward_model import (
    MultiModalRewardModel,
    Interaction,
    PreferenceComparison,
    get_reward_model,
)
from .feedback_collector import FeedbackCollector

__all__ = [
    "MultiModalRewardModel",
    "Interaction",
    "PreferenceComparison",
    "get_reward_model",
    "FeedbackCollector",
]
