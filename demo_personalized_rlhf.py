"""
Demo: Personalized RLHF with Automatic Reward Function Exploration

This demo shows how the agent:
1. Assesses user personality from conversations
2. Explores different reward function configurations
3. Learns optimal personality-reward mappings through interaction
4. Adapts to individual users based on their personality traits
"""

import asyncio
from datetime import datetime
import random

from rlhf.personalized_reward_model import get_personalized_reward_model
from rlhf.personality_assessment import BigFiveProfile
from rlhf.reward_model import Interaction
from rlhf.reward_exploration import get_reward_explorer


def generate_mock_personality(profile_type: str) -> BigFiveProfile:
    """Generate mock personality profiles for testing"""

    profiles = {
        'anxious_introvert': BigFiveProfile(
            openness=0.6,
            conscientiousness=0.7,
            extraversion=0.3,  # Low extraversion = introvert
            agreeableness=0.8,
            neuroticism=0.8,  # High neuroticism = anxious
            confidence=0.8,
            assessed_at=datetime.now()
        ),
        'stable_extrovert': BigFiveProfile(
            openness=0.7,
            conscientiousness=0.6,
            extraversion=0.8,  # High extraversion
            agreeableness=0.7,
            neuroticism=0.3,  # Low neuroticism = stable
            confidence=0.8,
            assessed_at=datetime.now()
        ),
        'organized_perfectionist': BigFiveProfile(
            openness=0.5,
            conscientiousness=0.9,  # High conscientiousness
            extraversion=0.5,
            agreeableness=0.6,
            neuroticism=0.6,
            confidence=0.8,
            assessed_at=datetime.now()
        ),
        'creative_open': BigFiveProfile(
            openness=0.9,  # High openness
            conscientiousness=0.4,
            extraversion=0.6,
            agreeableness=0.7,
            neuroticism=0.5,
            confidence=0.8,
            assessed_at=datetime.now()
        ),
    }

    return profiles.get(profile_type, BigFiveProfile(0.5, 0.5, 0.5, 0.5, 0.5, 0.7, datetime.now()))


def simulate_interaction(
    user_id: str,
    personality: BigFiveProfile,
    interaction_num: int
) -> Interaction:
    """
    Simulate a therapy interaction with realistic feedback based on personality
    """

    # Simulate different response patterns based on personality
    # High neuroticism -> clinical improvement matters more
    # High extraversion -> engagement matters more
    # High conscientiousness -> explicit feedback more accurate

    base_rating = 3  # Neutral

    # Adjust rating based on personality preferences
    if personality.neuroticism > 0.7:
        # Anxious users value clinical improvement
        clinical_quality = random.uniform(0.5, 1.0)
        base_rating += (clinical_quality - 0.5) * 4  # -2 to +2

    if personality.extraversion > 0.7:
        # Extroverts value engagement
        engagement_quality = random.uniform(0.5, 1.0)
        base_rating += (engagement_quality - 0.5) * 2  # -1 to +1

    if personality.conscientiousness > 0.7:
        # Conscientious users give more consistent feedback
        base_rating += random.uniform(-0.5, 0.5)
    else:
        base_rating += random.uniform(-1.5, 1.5)

    # Clip to 1-5 range
    rating = int(max(1, min(5, base_rating)))

    # Create interaction
    interaction = Interaction(
        interaction_id=f"{user_id}_interaction_{interaction_num}",
        user_id=user_id,
        timestamp=datetime.now(),
        user_message=f"Simulated user message {interaction_num}",
        agent_response=f"Simulated agent response {interaction_num}",
        context={},
        explicit_rating=rating,
        continued_conversation=random.random() > 0.2,  # 80% continue
        session_length=random.randint(3, 12),
        emotion_before='anxious' if personality.neuroticism > 0.6 else 'neutral',
        emotion_after='stable' if random.random() > 0.3 else 'anxious',
        risk_level_before='low',
        risk_level_after='low',
        safety_violation=False,
    )

    return interaction


async def simulate_learning_curve(
    personality_type: str,
    n_interactions: int = 50
):
    """
    Simulate learning curve for a specific personality type
    Shows how the model learns optimal reward weights over time
    """
    print(f"\n{'='*70}")
    print(f"Simulating Learning for: {personality_type.upper()}")
    print(f"{'='*70}\n")

    # Get personalized reward model
    model = get_personalized_reward_model()

    # Generate personality profile
    personality = generate_mock_personality(personality_type)
    user_id = f"user_{personality_type}"

    # Store personality
    model.user_personalities[user_id] = personality

    print(f"Personality Profile:")
    print(f"  Openness: {personality.openness:.2f}")
    print(f"  Conscientiousness: {personality.conscientiousness:.2f}")
    print(f"  Extraversion: {personality.extraversion:.2f}")
    print(f"  Agreeableness: {personality.agreeableness:.2f}")
    print(f"  Neuroticism: {personality.neuroticism:.2f}\n")

    # Track rewards over time
    rewards_history = []

    # Simulate interactions
    for i in range(n_interactions):
        # Simulate interaction
        interaction = simulate_interaction(user_id, personality, i)

        # Calculate personalized reward
        reward = model.calculate_reward(interaction, personality)
        rewards_history.append(reward)

        if (i + 1) % 10 == 0:
            avg_reward = sum(rewards_history[-10:]) / 10
            print(f"Interactions {i-9:3d}-{i+1:3d}: Avg Reward = {avg_reward:.3f}")

    # Show final learned configuration
    print(f"\n{'='*70}")
    print("Final Learned Reward Weights:")
    print(f"{'='*70}")

    reward_config = model._get_user_reward_config(user_id, personality)
    for component, weight in reward_config.to_dict().items():
        print(f"  {component:20s}: {weight:.3f}")

    # Overall performance
    final_avg = sum(rewards_history[-20:]) / 20
    initial_avg = sum(rewards_history[:20]) / 20
    improvement = (final_avg - initial_avg) / abs(initial_avg) * 100 if initial_avg != 0 else 0

    print(f"\nPerformance:")
    print(f"  Initial Average (first 20): {initial_avg:.3f}")
    print(f"  Final Average (last 20):    {final_avg:.3f}")
    print(f"  Improvement: {improvement:+.1f}%")


async def compare_personality_types():
    """
    Compare how different personality types result in different optimal reward configs
    """
    print("\n" + "="*80)
    print("COMPARING OPTIMAL REWARD FUNCTIONS ACROSS PERSONALITY TYPES")
    print("="*80 + "\n")

    personality_types = [
        'anxious_introvert',
        'stable_extrovert',
        'organized_perfectionist',
        'creative_open'
    ]

    for ptype in personality_types:
        await simulate_learning_curve(ptype, n_interactions=30)
        await asyncio.sleep(0.1)  # Small delay for readability

    # Show exploration statistics
    print("\n" + "="*80)
    print("OVERALL EXPLORATION STATISTICS")
    print("="*80 + "\n")

    explorer = get_reward_explorer()
    print(explorer.visualize_exploration())


async def demonstrate_adaptive_learning():
    """
    Demonstrate how the system adapts in real-time as it learns more about a user
    """
    print("\n" + "="*80)
    print("DEMONSTRATING ADAPTIVE LEARNING")
    print("="*80 + "\n")

    model = get_personalized_reward_model()

    # Start with unknown user (neutral personality)
    user_id = "adaptive_demo_user"
    initial_personality = BigFiveProfile(0.5, 0.5, 0.5, 0.5, 0.5, 0.3, datetime.now())

    print("Stage 1: Initial interactions with unknown personality")
    print("-" * 80)

    for i in range(10):
        interaction = simulate_interaction(user_id, initial_personality, i)
        reward = model.calculate_reward(interaction, initial_personality)

        if i == 0:
            config = model._get_user_reward_config(user_id, initial_personality)
            print(f"\nInitial reward weights (default):")
            for k, v in config.to_dict().items():
                print(f"  {k}: {v:.3f}")

    # After some interactions, we learn the user is highly neurotic
    print("\n\nStage 2: Personality assessment reveals high neuroticism")
    print("-" * 80)

    learned_personality = BigFiveProfile(0.6, 0.7, 0.3, 0.8, 0.9, 0.8, datetime.now())
    model.user_personalities[user_id] = learned_personality

    # Clear config cache to get new weights
    if user_id in model.user_reward_configs:
        del model.user_reward_configs[user_id]

    print(f"\nLearned personality:")
    print(f"  High Neuroticism: {learned_personality.neuroticism:.2f}")
    print(f"  Low Extraversion: {learned_personality.extraversion:.2f}")

    for i in range(10, 20):
        interaction = simulate_interaction(user_id, learned_personality, i)
        reward = model.calculate_reward(interaction, learned_personality)

        if i == 10:
            config = model._get_user_reward_config(user_id, learned_personality)
            print(f"\nAdapted reward weights (for neurotic personality):")
            for k, v in config.to_dict().items():
                print(f"  {k}: {v:.3f}")

    print("\n✓ System successfully adapted to user's personality!")


async def main():
    """Main demo function"""
    print("\n" + "="*80)
    print(" "*20 + "PERSONALIZED RLHF DEMO")
    print(" "*10 + "Automatic Reward Function Exploration by Personality")
    print("="*80)

    print("""
This demo illustrates how the agent:

1. ASSESSES personality using Big Five model
2. EXPLORES different reward function configurations
3. LEARNS which configurations work best for each personality type
4. ADAPTS reward weights based on individual differences

The key innovation: The agent discovers optimal reward functions through
exploration rather than using hand-crafted rules.
""")

    # Run demos
    choice = input("Select demo:\n  1. Compare personality types\n  2. Show adaptive learning\n  3. Both\n\nChoice (1-3): ").strip()

    if choice == '1':
        await compare_personality_types()
    elif choice == '2':
        await demonstrate_adaptive_learning()
    elif choice == '3':
        await compare_personality_types()
        await demonstrate_adaptive_learning()
    else:
        print("Invalid choice")
        return

    print("\n" + "="*80)
    print("Demo completed!")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
