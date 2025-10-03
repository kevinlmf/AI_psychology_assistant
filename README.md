# Psychology Agent + RLHF

A LLM-based mental health assistant integrating **Agent architecture** and **RLHF (Reinforcement Learning from Human Feedback)**, continuously optimized through multimodal data analysis and human feedback.

## Disclaimer

**This project is for research and personal interest only.** It is not a substitute for professional mental health care. If you are experiencing mental health issues, please consult a licensed mental health professional or contact a crisis helpline.

✨ I hope everyone can stay happy and free from anxiety, anger, or any other negative emotions every single day.

## Core Features

### 1. **Intelligent Conversational Agent**
- Therapeutic conversations based on CBT (Cognitive Behavioral Therapy) principles
- Automatic identification of cognitive distortions with gentle guidance
- Long-term user profile management with personalized responses

### 2. **Multimodal Behavior Analysis**
- **Search Log Analysis**: Identify mental health concerns
- **App Usage Patterns**: Track changes in sleep, social interactions, and activities
- **Conversation Content Analysis**: Emotion trends and theme identification

### 3. **Crisis Detection System**
- Multi-layer detection mechanism: Keywords + Rules + LM deep analysis
- Automatic identification of suicide risk and self-harm behavior
- Emergency resource recommendations and human handoff

### 4. **RLHF Optimization**
- Multimodal reward function:
  - User rating feedback (explicit)
  - Behavioral indicators (conversation continuation, session length)
  - Clinical indicators (emotion improvement, risk reduction)
  - Safety checks
- Preference comparison data collection
- Support for continuous iterative optimization

## Project Structure

```
psychology_agent/
├── agent/                      # Agent core
│   ├── conversation_manager.py # Conversation manager
│   └── memory_system.py        # User memory system
├── models/                     # LM wrapper
│   ├── lm_configs.py           # Model configuration and routing
│   └── lm_orchestrator.py      # Unified LM API interface
├── lm_analysis/                # LM analysis module
│   └── behavior_analyzer.py    # Behavior pattern analysis
├── data_collection/            # Data collection
│   └── search_log_processor.py # Search log processing
├── safety/                     # Safety module
│   └── crisis_detection.py     # Crisis detection
├── rlhf/                       # RLHF training
│   ├── reward_model.py         # Reward model
│   └── feedback_collector.py   # Feedback collector
├── data/                       # Data storage
│   ├── user_profiles/          # User profiles
│   ├── validated_dialogues/    # Annotated dialogues
│   └── rlhf/                   # RLHF training data
└── main.py                     # Main application entry
```

## Quick Start
git clone https://github.com/kevinlmf/AI_psychology_assistant
cd AI_psychology_assistant

### 1. Install Dependencies

```bash
cd psychology_agent
pip install -r requirements.txt
```

### 2. Configure API Keys

**Important: Never commit your `.env` file to git!**

Copy the example file and add your API keys:

```bash
cp .env.example .env
```

Then edit `.env` and add your Anthropic API key:

```bash
# Open .env file and replace with your actual key
ANTHROPIC_API_KEY=sk-ant-api03-your-actual-key-here
```

Get your API key from: https://console.anthropic.com/settings/keys

**Note**: The project uses Anthropic Claude by default. OpenAI key is optional.

### 3. Run Demo

```bash
python main.py
```

Select mode:
- **Mode 1**: Interactive conversation (full experience)
- **Mode 2**: Demo basic conversation features
- **Mode 3**: Demo behavior analysis
- **Mode 4**: Demo crisis detection

### 4. Example Conversation

```
You: I've been under a lot of work stress lately, feeling a bit anxious
```

### 5.Future Work

This is only a draft of the conversation.  And a lot improvment will be made in the future. 
