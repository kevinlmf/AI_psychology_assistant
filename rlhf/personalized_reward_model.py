"""
Personalized Reward Model
Integrates personality assessment and reward exploration to provide
adaptive, personality-aware reward calculations
"""

from typing import Dict, Any, Optional
from datetime import datetime

from .reward_model import MultiModalRewardModel, Interaction
from .personality_assessment import BigFiveProfile, get_personality_assessor
from .reward_exploration import RewardWeightConfig, get_reward_explorer


class PersonalizedRewardModel(MultiModalRewardModel):
    """
    Personalized Reward Model that adapts to user personality

    Key features:
    1. Automatically assesses user personality from conversations
    2. Explores optimal reward weights for different personalities
    3. Adapts reward calculations based on personality profile
    4. Learns from feedback to improve personality-reward mappings
    """

    def __init__(self, storage_dir: str = "psychology_agent/data/rlhf"):
        super().__init__(storage_dir)

        # Initialize personality assessor and reward explorer
        self.personality_assessor = get_personality_assessor()
        self.reward_explorer = get_reward_explorer()

        # Cache personality profiles per user
        self.user_personalities: Dict[str, BigFiveProfile] = {}

        # Cache reward configurations per user
        self.user_reward_configs: Dict[str, RewardWeightConfig] = {}

    def calculate_reward(
        self,
        interaction: Interaction,
        personality_profile: Optional[BigFiveProfile] = None
    ) -> float:
        """
        Calculate personalized reward based on user's personality

        Args:
            interaction: The interaction to evaluate
            personality_profile: Optional pre-computed personality profile

        Returns:
            Reward score (-1.0 to 1.0)
        """
        # Get or create personality profile
        if personality_profile is None:
            personality_profile = self._get_user_personality(interaction.user_id)

        # Get personalized reward weights
        reward_config = self._get_user_reward_config(
            interaction.user_id,
            personality_profile
        )

        # Calculate component rewards (same as base model)
        rewards = {
            'explicit_feedback': self._explicit_feedback_reward(interaction),
            'behavioral': self._behavioral_reward(interaction),
            'clinical': self._clinical_reward(interaction),
            'safety': self._safety_reward(interaction),
            'engagement': self._engagement_reward(interaction),
        }

        # Use PERSONALIZED weights instead of fixed weights
        weights = reward_config.to_dict()
        total_reward = sum(
            rewards[key] * weights[key]
            for key in rewards
        )

        # Update exploration with performance feedback
        self._update_exploration_feedback(
            interaction,
            personality_profile,
            reward_config,
            total_reward
        )

        return total_reward

    def _get_user_personality(self, user_id: str) -> BigFiveProfile:
        """
        Get or assess user's personality profile
        Uses cached profile if available and recent enough
        """
        # Check cache
        if user_id in self.user_personalities:
            profile = self.user_personalities[user_id]

            # Check if assessment is recent (within 30 days)
            if profile.assessed_at:
                days_old = (datetime.now() - profile.assessed_at).days
                if days_old < 30 and profile.confidence > 0.5:
                    return profile

        # Load from disk
        profile = self.personality_assessor.load_profile(user_id)

        if profile and profile.confidence > 0.5:
            self.user_personalities[user_id] = profile
            return profile

        # Return default neutral profile if no assessment available
        return BigFiveProfile(
            openness=0.5,
            conscientiousness=0.5,
            extraversion=0.5,
            agreeableness=0.5,
            neuroticism=0.5,
            confidence=0.3,
            assessed_at=datetime.now()
        )

    def _get_user_reward_config(
        self,
        user_id: str,
        personality_profile: BigFiveProfile
    ) -> RewardWeightConfig:
        """
        Get personalized reward configuration for user
        Uses exploration framework to select optimal config
        """
        # Check cache
        if user_id in self.user_reward_configs:
            return self.user_reward_configs[user_id]

        # Use reward explorer to select configuration
        # This handles exploration vs exploitation automatically
        config = self.reward_explorer.select_reward_config(
            personality_profile,
            exploration_mode=True  # Enable exploration
        )

        # Cache for this session
        self.user_reward_configs[user_id] = config

        return config

    def _update_exploration_feedback(
        self,
        interaction: Interaction,
        personality_profile: BigFiveProfile,
        reward_config: RewardWeightConfig,
        total_reward: float
    ):
        """
        Update reward explorer with performance feedback
        This is the META-LEARNING step
        """
        # Calculate performance metrics from interaction
        performance_metrics = {
            'user_satisfaction': self._calculate_satisfaction(interaction),
            'clinical_improvement': self._calculate_clinical_improvement(interaction),
            'engagement': self._calculate_engagement(interaction),
        }

        # Update explorer (this improves personality-reward mappings over time)
        self.reward_explorer.update_performance(
            personality_profile,
            reward_config,
            performance_metrics
        )

    def _calculate_satisfaction(self, interaction: Interaction) -> float:
        """Calculate user satisfaction score (0-1)"""
        if interaction.explicit_rating is not None:
            # Normalize 1-5 rating to 0-1
            return (interaction.explicit_rating - 1) / 4.0

        # Implicit satisfaction signals
        score = 0.5  # Default neutral

        if interaction.continued_conversation:
            score += 0.2

        if interaction.user_satisfaction_indicators:
            if interaction.user_satisfaction_indicators.get('expressed_gratitude'):
                score += 0.3

        return min(score, 1.0)

    def _calculate_clinical_improvement(self, interaction: Interaction) -> float:
        """Calculate clinical improvement score (0-1)"""
        score = 0.5  # Default neutral

        # Emotion improvement
        emotion_change = self._assess_emotion_change(
            interaction.emotion_before,
            interaction.emotion_after
        )
        score += emotion_change * 0.25

        # Risk reduction
        risk_change = self._assess_risk_change(
            interaction.risk_level_before,
            interaction.risk_level_after
        )
        score += risk_change * 0.25

        return max(min(score, 1.0), 0.0)

    def _calculate_engagement(self, interaction: Interaction) -> float:
        """Calculate engagement score (0-1)"""
        score = 0.0

        if interaction.continued_conversation:
            score += 0.5

        # Session length indicates engagement
        if interaction.session_length > 0:
            # Normalize session length (assume 10 turns = high engagement)
            score += min(interaction.session_length / 10.0, 0.5)

        return min(score, 1.0)

    async def assess_user_personality(
        self,
        user_id: str,
        conversation_history: list,
        llm_orchestrator = None
    ) -> BigFiveProfile:
        """
        Manually trigger personality assessment
        Useful for periodic re-assessment or initial profiling
        """
        profile = await self.personality_assessor.assess_from_conversation(
            conversation_history,
            user_id,
            llm_orchestrator
        )

        # Update cache
        self.user_personalities[user_id] = profile

        # Clear reward config cache to get new config for updated personality
        if user_id in self.user_reward_configs:
            del self.user_reward_configs[user_id]

        return profile

    def get_exploration_stats(self) -> Dict[str, Any]:
        """Get statistics about reward exploration progress"""
        return self.reward_explorer.get_exploration_statistics()

    def get_user_profile_summary(self, user_id: str) -> Dict[str, Any]:
        """Get summary of user's personality and reward configuration"""
        personality = self._get_user_personality(user_id)
        reward_config = self._get_user_reward_config(user_id, personality)

        trait_descriptions = self.personality_assessor.get_trait_description(personality)

        return {
            'user_id': user_id,
            'personality': {
                'traits': personality.to_dict(),
                'descriptions': trait_descriptions,
            },
            'reward_weights': reward_config.to_dict(),
            'exploration_stats': self.get_exploration_stats(),
        }


# Global singleton
_personalized_reward_model = None


def get_personalized_reward_model() -> PersonalizedRewardModel:
    """Get global PersonalizedRewardModel instance"""
    global _personalized_reward_model
    if _personalized_reward_model is None:
        _personalized_reward_model = PersonalizedRewardModel()
    return _personalized_reward_model
