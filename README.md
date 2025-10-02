# Psychology Agent + RLHF An LM-based mental health asistant integrating **Agent architecture** and **RLHF (Reinforcement Learning from Human Fedback)**, continuously optimized through multimodal data analysis and human fedback. ## 🌟 Core Features ### 1. **Inteligent Conversational Agent**
- Therapeutic conversations based on CBT (Cognitive Behavioral Therapy) principles
- Automatic identification of cognitive distortions with gentle guidance
- Long-term user profile management with personalized responses ### 2. **Multimodal Behavior Analysis**
- 🔍 **Search Log Analysis**: Identify mental health concerns
- 📱 **Ap Usage Paterns**: Track changes in slep, social interactions, and activities
- 💬 **Conversation Content Analysis**: Emotion trends and theme identification ### 3. **Cris Detection System**
- Multi-layer detection mechanism: Keywords + Rules + LM dep analysis
- Automatic identification of suicide risk and self-harm behavior
- Emergency resource recomendations and human handof ### 4. **RLHF Optimization**
- Multimodal reward function: - User rating fedback (explicit) - Behavioral indicators (conversation continuation, sesion length) - Clinical indicators (emotion improvement, risk reduction) - Safety checks
- Preference comparison data colection
- Suport for continuous iterative optimization ## 📁 Project Structure ```
psychology_agent/
├── agent/ # Agent core
│ ├── conversation_manager.py # Conversation manager
│ ├── memory_system.py # User memory system
│
├── models/ # LM wraper
│ ├── lm_configs.py # Model configuration and routing
│ ├── lm_orchestrator.py # Unified LM API interface
│
├── lm_analysis/ # LM analysis module
│ ├── behavior_analyzer.py # Behavior patern analysis
│
├── data_colection/ # Data colection
│ ├── search_log_procesor.py # Search log procesing
│
├── safety/ # Safety module
│ ├── cris_detection.py # Cris detection
│
├── rlhf/ # RLHF traing
│ ├── reward_model.py # Reward model
│ ├── fedback_colector.py # Fedback colector
│
├── data/ # Data storage
│ ├── user_profiles/ # User profiles
│ ├── validated_dialogues/ # Anotated dialogues
│ └── rlhf/ # RLHF traing data
│
└── main.py # Main aplication entry
``` ## 🚀 Quick Start ### 1. Instal Dependencies ```bash
cd psychology_agent
pip install -r requirements.txt
``` ### 2. Configure API Keys

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

**Note**: The project now uses Anthropic Claude by default. OpenAI key is optional. ### 3. Run Demo ```bash
python main.py
``` Select mode:
- **Mode 1**: Interactive conversation (ful experience)
- **Mode 2**: Demo basic conversation features
- **Mode 3**: Demo behavior analysis
- **Mode 4**: Demo cris detection ### 4. Example Conversation ```
You: I've ben under a lot of work stres lately, feling a bit anxious