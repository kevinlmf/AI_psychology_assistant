"""
Quick Test Script for Personalized RLHF System
Verifies all components are working correctly
"""

import sys
from datetime import datetime


def test_imports():
    """Test if all modules can be imported"""
    print("Testing imports...")
    try:
        from rlhf.personality_assessment import BigFiveProfile, PersonalityAssessor
        from rlhf.reward_exploration import RewardWeightConfig, RewardExplorer, PersonalityCluster
        from rlhf.personalized_reward_model import PersonalizedRewardModel
        from rlhf.reward_model import Interaction
        from rlhf.visualization import ExplorationVisualizer
        print("✓ All imports successful\n")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}\n")
        return False


def test_personality_assessment():
    """Test personality assessment module"""
    print("Testing personality assessment...")
    try:
        from rlhf.personality_assessment import BigFiveProfile, PersonalityAssessor

        # Create a personality profile
        profile = BigFiveProfile(
            openness=0.7,
            conscientiousness=0.8,
            extraversion=0.4,
            agreeableness=0.9,
            neuroticism=0.6,
            confidence=0.8,
            assessed_at=datetime.now()
        )

        # Test serialization
        profile_dict = profile.to_dict()
        restored = BigFiveProfile.from_dict(profile_dict)

        assert abs(restored.openness - 0.7) < 0.01
        assert abs(restored.neuroticism - 0.6) < 0.01

        # Test assessor
        assessor = PersonalityAssessor()

        # Test heuristic assessment
        mock_history = [
            {'user_message': 'I feel very anxious and worried about everything'},
            {'user_message': 'I like to plan things carefully and stay organized'},
        ]

        heuristic_profile = assessor._heuristic_assessment(mock_history)
        assert 0.0 <= heuristic_profile.neuroticism <= 1.0

        print("✓ Personality assessment working\n")
        return True
    except Exception as e:
        print(f"✗ Personality assessment failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_reward_exploration():
    """Test reward exploration framework"""
    print("Testing reward exploration...")
    try:
        from rlhf.reward_exploration import RewardWeightConfig, RewardExplorer
        from rlhf.personality_assessment import BigFiveProfile

        # Test weight config
        config = RewardWeightConfig(
            explicit_feedback=0.3,
            behavioral=0.25,
            clinical=0.25,
            safety=0.15,
            engagement=0.05
        )

        # Test normalization
        normalized = config.normalize()
        total = sum(normalized.to_dict().values())
        assert abs(total - 1.0) < 0.01

        # Test explorer
        explorer = RewardExplorer(
            storage_dir="test_temp_exploration",
            exploration_rate=0.2
        )

        # Create test personality
        personality = BigFiveProfile(0.5, 0.5, 0.5, 0.5, 0.5, 0.7, datetime.now())

        # Select config
        selected_config = explorer.select_reward_config(personality, exploration_mode=True)
        assert isinstance(selected_config, RewardWeightConfig)

        # Update performance
        metrics = {
            'user_satisfaction': 0.8,
            'clinical_improvement': 0.7,
            'engagement': 0.6,
        }
        explorer.update_performance(personality, selected_config, metrics)

        print("✓ Reward exploration working\n")
        return True
    except Exception as e:
        print(f"✗ Reward exploration failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_personalized_reward_model():
    """Test personalized reward model"""
    print("Testing personalized reward model...")
    try:
        from rlhf.personalized_reward_model import PersonalizedRewardModel
        from rlhf.reward_model import Interaction
        from rlhf.personality_assessment import BigFiveProfile

        # Create model
        model = PersonalizedRewardModel(storage_dir="test_temp_rlhf")

        # Create test personality
        personality = BigFiveProfile(0.6, 0.7, 0.3, 0.8, 0.8, 0.8, datetime.now())
        model.user_personalities['test_user'] = personality

        # Create test interaction
        interaction = Interaction(
            interaction_id="test_001",
            user_id="test_user",
            timestamp=datetime.now(),
            user_message="I'm feeling anxious",
            agent_response="I understand. Let's work through this together.",
            context={},
            explicit_rating=4,
            continued_conversation=True,
            session_length=5,
            emotion_before='anxious',
            emotion_after='stable',
            risk_level_before='low',
            risk_level_after='low',
        )

        # Calculate reward
        reward = model.calculate_reward(interaction, personality)
        assert -1.0 <= reward <= 1.0

        # Get profile summary
        summary = model.get_user_profile_summary('test_user')
        assert 'personality' in summary
        assert 'reward_weights' in summary

        print(f"✓ Personalized reward model working (test reward: {reward:.3f})\n")
        return True
    except Exception as e:
        print(f"✗ Personalized reward model failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_visualization():
    """Test visualization module"""
    print("Testing visualization...")
    try:
        from rlhf.visualization import ExplorationVisualizer

        visualizer = ExplorationVisualizer()

        # Generate text report (should work even with no data)
        report = visualizer.generate_text_report()
        assert len(report) > 0

        print("✓ Visualization working\n")
        return True
    except Exception as e:
        print(f"✗ Visualization failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def test_clustering():
    """Test personality clustering"""
    print("Testing personality clustering...")
    try:
        from rlhf.reward_exploration import PersonalityCluster
        from rlhf.personality_assessment import BigFiveProfile

        cluster = PersonalityCluster(n_clusters=3)

        # Create test profiles
        profiles = [
            ('user1', BigFiveProfile(0.8, 0.7, 0.3, 0.9, 0.2, 0.8, datetime.now())),
            ('user2', BigFiveProfile(0.3, 0.4, 0.8, 0.5, 0.7, 0.8, datetime.now())),
            ('user3', BigFiveProfile(0.5, 0.5, 0.5, 0.5, 0.5, 0.8, datetime.now())),
            ('user4', BigFiveProfile(0.7, 0.8, 0.2, 0.9, 0.3, 0.8, datetime.now())),
        ]

        # Fit clustering
        cluster.fit(profiles)

        # Predict cluster for new profile
        new_profile = BigFiveProfile(0.75, 0.75, 0.25, 0.85, 0.25, 0.8, datetime.now())
        cluster_id = cluster.predict_cluster(new_profile)

        assert 0 <= cluster_id < 3

        print(f"✓ Clustering working (assigned to cluster {cluster_id})\n")
        return True
    except Exception as e:
        print(f"✗ Clustering failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("="*70)
    print(" "*15 + "PERSONALIZED RLHF SYSTEM TEST")
    print("="*70 + "\n")

    tests = [
        ("Imports", test_imports),
        ("Personality Assessment", test_personality_assessment),
        ("Reward Exploration", test_reward_exploration),
        ("Personalized Reward Model", test_personalized_reward_model),
        ("Visualization", test_visualization),
        ("Clustering", test_clustering),
    ]

    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}\n")
            results.append((test_name, False))

    # Summary
    print("="*70)
    print(" "*25 + "TEST SUMMARY")
    print("="*70 + "\n")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:10s} {test_name}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
        return 0
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
