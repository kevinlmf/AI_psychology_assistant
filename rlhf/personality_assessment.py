"""
Personality Assessment Module
Assesses user personality using Big Five model through conversation analysis
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import json
from pathlib import Path
from datetime import datetime


@dataclass
class BigFiveProfile:
    """
    Big Five Personality Traits (OCEAN model)
    Each trait scored 0.0 - 1.0
    """
    openness: float  # Openness to experience
    conscientiousness: float  # Conscientiousness
    extraversion: float  # Extraversion
    agreeableness: float  # Agreeableness
    neuroticism: float  # Neuroticism (Emotional Stability)

    confidence: float = 0.5  # Assessment confidence
    assessed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'openness': self.openness,
            'conscientiousness': self.conscientiousness,
            'extraversion': self.extraversion,
            'agreeableness': self.agreeableness,
            'neuroticism': self.neuroticism,
            'confidence': self.confidence,
            'assessed_at': self.assessed_at.isoformat() if self.assessed_at else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BigFiveProfile':
        assessed_at = data.get('assessed_at')
        if assessed_at and isinstance(assessed_at, str):
            assessed_at = datetime.fromisoformat(assessed_at)

        return cls(
            openness=data['openness'],
            conscientiousness=data['conscientiousness'],
            extraversion=data['extraversion'],
            agreeableness=data['agreeableness'],
            neuroticism=data['neuroticism'],
            confidence=data.get('confidence', 0.5),
            assessed_at=assessed_at,
        )


class PersonalityAssessor:
    """
    Personality Assessor using LLM-based conversation analysis
    """

    # Trait indicators for conversation analysis
    TRAIT_INDICATORS = {
        'openness': {
            'high': [
                'curious', 'creative', 'imaginative', 'open-minded',
                'exploring new ideas', 'abstract thinking', 'artistic',
                'enjoys novelty', 'intellectually curious'
            ],
            'low': [
                'practical', 'conventional', 'routine-oriented',
                'prefers familiar', 'concrete thinking', 'traditional'
            ]
        },
        'conscientiousness': {
            'high': [
                'organized', 'disciplined', 'planned', 'goal-oriented',
                'responsible', 'detail-oriented', 'systematic',
                'punctual', 'follows through'
            ],
            'low': [
                'spontaneous', 'flexible', 'casual', 'disorganized',
                'procrastinate', 'impulsive', 'relaxed about deadlines'
            ]
        },
        'extraversion': {
            'high': [
                'outgoing', 'sociable', 'talkative', 'energetic',
                'enjoys social interaction', 'seeks stimulation',
                'enthusiastic', 'assertive', 'active lifestyle'
            ],
            'low': [
                'reserved', 'quiet', 'introspective', 'prefers solitude',
                'needs alone time', 'reflective', 'private',
                'small social circle', 'independent'
            ]
        },
        'agreeableness': {
            'high': [
                'cooperative', 'compassionate', 'trusting', 'helpful',
                'empathetic', 'considerate', 'kind', 'values harmony',
                'forgiving', 'team-oriented'
            ],
            'low': [
                'competitive', 'skeptical', 'critical', 'independent-minded',
                'direct', 'analytical', 'values truth over harmony',
                'challenges others'
            ]
        },
        'neuroticism': {
            'high': [
                'anxious', 'worried', 'stressed', 'emotional',
                'sensitive to stress', 'mood swings', 'self-conscious',
                'prone to negative emotions', 'rumination'
            ],
            'low': [
                'calm', 'stable', 'resilient', 'relaxed',
                'emotionally stable', 'handles stress well',
                'confident', 'secure', 'even-tempered'
            ]
        }
    }

    def __init__(self, storage_dir: str = "psychology_agent/data/personality_profiles"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    async def assess_from_conversation(
        self,
        conversation_history: List[Dict[str, str]],
        user_id: str,
        llm_orchestrator = None
    ) -> BigFiveProfile:
        """
        Assess personality from conversation history using LLM

        Args:
            conversation_history: List of conversation turns
            user_id: User identifier
            llm_orchestrator: LLM orchestrator for analysis

        Returns:
            BigFiveProfile with assessed traits
        """
        # Build analysis prompt
        prompt = self._build_assessment_prompt(conversation_history)

        # Use LLM to analyze (if available)
        if llm_orchestrator:
            try:
                from models.llm_orchestrator import LLMOrchestrator, TaskType, ModelRouter

                config = ModelRouter.get_model_config(TaskType.BEHAVIOR_ANALYSIS)
                response = await llm_orchestrator.generate(prompt, config)

                # Parse LLM response (expects JSON format)
                profile = self._parse_llm_assessment(response)

            except Exception as e:
                print(f"LLM assessment failed: {e}, using heuristic method")
                profile = self._heuristic_assessment(conversation_history)
        else:
            # Fallback to heuristic-based assessment
            profile = self._heuristic_assessment(conversation_history)

        # Save assessment
        self.save_profile(user_id, profile)

        return profile

    def _build_assessment_prompt(self, conversation_history: List[Dict[str, str]]) -> str:
        """Build prompt for LLM-based personality assessment"""

        # Extract recent user messages
        user_messages = [
            turn['user_message']
            for turn in conversation_history
            if 'user_message' in turn
        ][-10:]  # Last 10 messages

        conversation_text = "\n".join([f"User: {msg}" for msg in user_messages])

        prompt = f"""Analyze the following conversation and assess the user's Big Five personality traits.

Conversation:
{conversation_text}

Please assess each of the Big Five traits on a scale from 0.0 to 1.0:
- Openness (0=conventional, 1=curious/creative)
- Conscientiousness (0=spontaneous, 1=organized/disciplined)
- Extraversion (0=introverted, 1=extraverted)
- Agreeableness (0=challenging, 1=cooperative/compassionate)
- Neuroticism (0=emotionally stable, 1=anxious/sensitive)

Also provide a confidence score (0.0-1.0) for your assessment.

Respond in JSON format:
{{
    "openness": 0.0-1.0,
    "conscientiousness": 0.0-1.0,
    "extraversion": 0.0-1.0,
    "agreeableness": 0.0-1.0,
    "neuroticism": 0.0-1.0,
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""

        return prompt

    def _parse_llm_assessment(self, llm_response: str) -> BigFiveProfile:
        """Parse LLM response into BigFiveProfile"""
        try:
            # Extract JSON from response
            import re
            json_match = re.search(r'\{[^}]+\}', llm_response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                data = json.loads(llm_response)

            return BigFiveProfile(
                openness=float(data['openness']),
                conscientiousness=float(data['conscientiousness']),
                extraversion=float(data['extraversion']),
                agreeableness=float(data['agreeableness']),
                neuroticism=float(data['neuroticism']),
                confidence=float(data.get('confidence', 0.7)),
                assessed_at=datetime.now()
            )
        except Exception as e:
            print(f"Failed to parse LLM assessment: {e}")
            # Return neutral profile
            return BigFiveProfile(0.5, 0.5, 0.5, 0.5, 0.5, confidence=0.3, assessed_at=datetime.now())

    def _heuristic_assessment(self, conversation_history: List[Dict[str, str]]) -> BigFiveProfile:
        """
        Heuristic-based personality assessment using keyword matching
        Fallback method when LLM is unavailable
        """
        # Combine all user messages
        user_text = " ".join([
            turn.get('user_message', '')
            for turn in conversation_history
        ]).lower()

        scores = {}

        # Calculate score for each trait
        for trait, indicators in self.TRAIT_INDICATORS.items():
            high_count = sum(
                1 for keyword in indicators['high']
                if keyword.lower() in user_text
            )
            low_count = sum(
                1 for keyword in indicators['low']
                if keyword.lower() in user_text
            )

            total = high_count + low_count
            if total > 0:
                # Score between 0 and 1
                scores[trait] = high_count / total
            else:
                # No indicators found, default to middle
                scores[trait] = 0.5

        # Confidence based on amount of data
        confidence = min(len(conversation_history) / 20.0, 1.0)  # Full confidence after 20+ turns

        return BigFiveProfile(
            openness=scores.get('openness', 0.5),
            conscientiousness=scores.get('conscientiousness', 0.5),
            extraversion=scores.get('extraversion', 0.5),
            agreeableness=scores.get('agreeableness', 0.5),
            neuroticism=scores.get('neuroticism', 0.5),
            confidence=confidence,
            assessed_at=datetime.now()
        )

    def save_profile(self, user_id: str, profile: BigFiveProfile):
        """Save personality profile to disk"""
        file_path = self.storage_dir / f"{user_id}_personality.json"

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(profile.to_dict(), f, indent=2, ensure_ascii=False)

    def load_profile(self, user_id: str) -> Optional[BigFiveProfile]:
        """Load personality profile from disk"""
        file_path = self.storage_dir / f"{user_id}_personality.json"

        if not file_path.exists():
            return None

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return BigFiveProfile.from_dict(data)
        except Exception as e:
            print(f"Failed to load personality profile: {e}")
            return None

    def get_trait_description(self, profile: BigFiveProfile) -> Dict[str, str]:
        """Get human-readable description of personality traits"""

        def describe_trait(value: float, trait_name: str) -> str:
            if value > 0.7:
                return f"High {trait_name}"
            elif value > 0.3:
                return f"Moderate {trait_name}"
            else:
                return f"Low {trait_name}"

        return {
            'openness': describe_trait(profile.openness, "Openness"),
            'conscientiousness': describe_trait(profile.conscientiousness, "Conscientiousness"),
            'extraversion': describe_trait(profile.extraversion, "Extraversion"),
            'agreeableness': describe_trait(profile.agreeableness, "Agreeableness"),
            'neuroticism': describe_trait(profile.neuroticism, "Neuroticism"),
        }


# Global singleton
_personality_assessor = None


def get_personality_assessor() -> PersonalityAssessor:
    """Get global PersonalityAssessor instance"""
    global _personality_assessor
    if _personality_assessor is None:
        _personality_assessor = PersonalityAssessor()
    return _personality_assessor
