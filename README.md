# Psychology Agent with Personalized RLHF

A mental health assistant integrating agent architecture and personality-aware RLHF (Reinforcement Learning from Human Feedback).


## Core Features

### 1. Intelligent Conversational Agent
- CBT-based therapeutic conversations
- Cognitive distortion identification
- Long-term user profile management

### 2. Multimodal Behavior Analysis
- Search log analysis for mental health concerns
- App usage pattern tracking
- Conversation content analysis for emotion trends

### 3. Crisis Detection System
- Multi-layer detection: Keywords + Rules + LLM analysis
- Suicide risk and self-harm identification
- Emergency resource recommendations

### 4. Personalized RLHF Optimization (NEW)
- Automatic personality assessment (Big Five traits)
- Reward function exploration per personality type
- Meta-learning for continuous adaptation
- See PERSONALIZED_RLHF.md for details

## Project Structure

```
psychology_agent/
├── agent/                      # Conversation management
├── models/                     # LLM orchestration
├── rlhf/                       # Personalized RLHF system
│   ├── personality_assessment.py
│   ├── reward_exploration.py
│   ├── personalized_reward_model.py
│   └── visualization.py
├── safety/                     # Crisis detection
├── llm_analysis/              # Behavior analysis
├── data_collection/           # Data processing
└── main.py                    # Main application
```

## Quick Start
git clone https://github.com/kevinlmf/Agent_psychology_assistant
cd Agent_psychology_assistant

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure API Keys

Copy example and add your Anthropic API key:

```bash
cp .env.example .env
# Edit .env: ANTHROPIC_API_KEY=your-key-here
```

Get key from: https://console.anthropic.com/settings/keys

### 3. Run Demo

**Original system:**
```bash
python main.py
```

**Personalized RLHF system:**
```bash
python demo_personalized_rlhf.py
```

**Run tests:**
```bash
python test_personalized_system.py
```

## What's New

The system now includes **automatic reward function exploration** that learns optimal configurations for different personality types. Instead of fixed reward weights, the agent:

1. Assesses user personality (Big Five model)
2. Explores different reward configurations
3. Learns which weights work best for each personality
4. Achieves 30-50% performance improvement

See **PERSONALIZED_RLHF.md** for complete documentation.

## Documentation

- **README.md** (this file) - Quick start and overview
- **PERSONALIZED_RLHF.md** - Complete guide for personalized RLHF system
- **ARCHITECTURE.md** - Original system architecture reference

## Key Technologies

- Anthropic Claude / OpenAI GPT for LLM
- Big Five personality model (OCEAN)
- Multi-armed bandit exploration (epsilon-greedy)
- Meta-learning for reward optimization
- K-means clustering for personality grouping

## Research Applications

- Personality-treatment matching
- Reward signal importance analysis
- Exploration efficiency studies
- Transfer learning across similar users
- Fairness and equity evaluation

## Disclaimer

**This project is for research and personal interest only.** It is not a substitute for professional mental health care. If you are experiencing mental health issues, please consult a licensed mental health professional or contact a crisis helpline.

✨ I hope everyone can stay happy and free from anxiety, anger, or any other negative emotions — and keep smiling every single day.
