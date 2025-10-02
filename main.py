"""
Psychology Agent - Main Application Entry
Demonstrates complete conversation flow + LLM analysis + RLHF feedback collection
"""
import asyncio
from datetime import datetime
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from agent.conversation_manager import ConversationManager
from llm_analysis.behavior_analyzer import BehaviorAnalyzer
from data_collection.search_log_processor import (
    SearchLogProcessor,
    generate_mock_search_data,
)
from rlhf.feedback_collector import collect_rating_cli


class PsychologyAgentApp:
    """Psychology Agent Application"""

    def __init__(self, user_id: str = "demo_user"):
        self.user_id = user_id
        self.conversation_manager = None
        self.behavior_analyzer = BehaviorAnalyzer()
        self.search_processor = SearchLogProcessor()

    async def start_session(self):
        """Start a new session"""
        print("\n" + "=" * 60)
        print("🧠 Psychology Agent - Mental Health Assistant")
        print("=" * 60)
        print("\nWelcome! I am your mental health assistant.")
        print("I can listen to your feelings and provide support and advice.")
        print("\nType 'analyze' to view behavior analysis")
        print("Type 'quit' or 'exit' to end the session\n")

        # Create session
        self.conversation_manager = ConversationManager(user_id=self.user_id)

        # Start conversation loop
        await self.conversation_loop()

    async def conversation_loop(self):
        """Main conversation loop"""
        turn_count = 0

        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()

                if not user_input:
                    continue

                # Command processing
                if user_input.lower() in ['quit', 'exit']:
                    await self.end_session()
                    break

                if user_input.lower() == 'analyze':
                    await self.show_behavior_analysis()
                    continue

                # Process message
                print("\nAssistant: ", end="", flush=True)
                response = await self.conversation_manager.process_message(user_input)
                print(response)

                turn_count += 1

                # Collect feedback every 3 turns
                if turn_count % 3 == 0:
                    collect_rating_cli(user_input, response, self.user_id)

            except KeyboardInterrupt:
                print("\n\nInterrupt detected, ending session...")
                await self.end_session()
                break
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()

    async def show_behavior_analysis(self):
        """Display behavior analysis"""
        print("\n" + "=" * 60)
        print("📊 Behavior Pattern Analysis")
        print("=" * 60)

        # Generate mock search data
        mock_searches = generate_mock_search_data()
        processed_searches = self.search_processor.process_search_history(mock_searches)

        # Extract search texts
        search_texts = [s.anonymized_query for s in processed_searches]

        # Get conversation summary
        conversation_summary = self.conversation_manager.get_conversation_history(last_n=5)

        # Simulate app usage data
        app_usage = {
            'screen_time': 6.5,
            'social_media_time': 2.3,
            'sleep_tracking': 6.2,
            'exercise': 25,
        }

        # Analyze
        print("\nAnalyzing...")
        pattern = await self.behavior_analyzer.analyze_recent_activity(
            user_id=self.user_id,
            search_history=search_texts[:5],
            app_usage=app_usage,
            conversation_summary=conversation_summary,
            days=7,
        )

        # Display results
        print(f"\nEmotional State: {pattern.emotional_state} (Confidence: {pattern.emotion_confidence:.2f})")
        print(f"Identified Themes: {', '.join(pattern.identified_themes) if pattern.identified_themes else 'None'}")
        print(f"Risk Factors: {', '.join(pattern.risk_factors) if pattern.risk_factors else 'None'}")
        print(f"Protective Factors: {', '.join(pattern.protective_factors) if pattern.protective_factors else 'None'}")

        if pattern.behavior_changes:
            print("\nBehavior Changes:")
            for key, value in pattern.behavior_changes.items():
                print(f"  - {key}: {value}")

        # Generate personalized insights
        print("\nGenerating personalized insights...")
        insights = await self.behavior_analyzer.generate_personalized_insights(pattern)
        print("\n💡 Personalized Insights:")
        print(insights)

    async def end_session(self):
        """End session"""
        print("\nGenerating session summary...")

        # Request user satisfaction rating
        try:
            satisfaction = input("\nPlease rate this session (1-5): ").strip()
            if satisfaction.isdigit():
                satisfaction = int(satisfaction)
            else:
                satisfaction = None
        except:
            satisfaction = None

        # End session
        summary = await self.conversation_manager.end_session(satisfaction)

        print("\n" + "=" * 60)
        print("📝 Session Summary")
        print("=" * 60)
        print(summary)
        print("\nThank you for using! Hope my support was helpful.")
        print("If you need professional help, please contact a mental health counselor.")
        print("=" * 60 + "\n")


async def demo_basic_conversation():
    """Demonstrate basic conversation functionality"""
    print("\n🎯 Demo Mode: Basic Conversation")
    print("-" * 60)

    app = PsychologyAgentApp(user_id="demo_basic")
    manager = ConversationManager(user_id="demo_basic")

    # Simulate conversation
    test_messages = [
        "I've been under a lot of work stress lately, feeling a bit anxious",
        "Yes, working overtime until late every day, and my sleep is not good",
        "Are there any ways to relieve this?",
    ]

    for msg in test_messages:
        print(f"\nUser: {msg}")
        response = await manager.process_message(msg)
        print(f"Assistant: {response}")
        await asyncio.sleep(1)


async def demo_behavior_analysis():
    """Demonstrate behavior analysis functionality"""
    print("\n🎯 Demo Mode: Behavior Analysis")
    print("-" * 60)

    analyzer = BehaviorAnalyzer()
    processor = SearchLogProcessor()

    # Process mock search data
    mock_searches = generate_mock_search_data()
    processed = processor.process_search_history(mock_searches)
    print(f"\n✓ Processed {len(processed)} search records")

    summary = processor.generate_summary(processed)
    print(f"Search category distribution: {summary['categories']}")
    print(f"Sentiment trends: {summary['sentiments']}")

    # Behavior analysis
    print("\nPerforming behavior analysis...")
    pattern = await analyzer.analyze_recent_activity(
        user_id="demo_analysis",
        search_history=[s.anonymized_query for s in processed[:5]],
        app_usage={'screen_time': 8, 'sleep_tracking': 5.5},
    )

    print(f"\nAnalysis results:")
    print(f"  Emotion: {pattern.emotional_state} ({pattern.emotion_confidence:.2f})")
    print(f"  Themes: {pattern.identified_themes}")
    print(f"  Risks: {pattern.risk_factors}")


async def demo_crisis_detection():
    """Demonstrate crisis detection functionality"""
    print("\n🎯 Demo Mode: Crisis Detection")
    print("-" * 60)

    from safety.crisis_detection import CrisisDetector

    detector = CrisisDetector()

    test_cases = [
        ("I've been under a bit of stress lately", "Low-risk scenario"),
        ("Life feels meaningless, very painful", "Medium-risk scenario"),
        ("I really don't want to live anymore, want to end it all", "High-risk scenario"),
    ]

    for message, description in test_cases:
        print(f"\nTest: {description}")
        print(f"Message: \"{message}\"")

        assessment = await detector.assess_risk(message, "demo_crisis")

        print(f"Risk Level: {assessment['risk_level']}")
        print(f"Confidence: {assessment.get('confidence', 0):.2f}")
        print(f"Detected Signals: {assessment.get('signals', [])}")

        if assessment['risk_level'] in ['high', 'medium']:
            response = await detector.generate_crisis_response(assessment)
            print(f"\nSystem Response:\n{response[:200]}...")


def main():
    """Main function"""
    print("\nWelcome to Psychology Agent!")
    print("\nPlease select a mode:")
    print("1. Interactive Conversation (Full Experience)")
    print("2. Demo: Basic Conversation")
    print("3. Demo: Behavior Analysis")
    print("4. Demo: Crisis Detection")

    try:
        choice = input("\nPlease select (1-4): ").strip()

        if choice == '1':
            app = PsychologyAgentApp()
            asyncio.run(app.start_session())
        elif choice == '2':
            asyncio.run(demo_basic_conversation())
        elif choice == '3':
            asyncio.run(demo_behavior_analysis())
        elif choice == '4':
            asyncio.run(demo_crisis_detection())
        else:
            print("Invalid selection")

    except KeyboardInterrupt:
        print("\n\nExited")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
