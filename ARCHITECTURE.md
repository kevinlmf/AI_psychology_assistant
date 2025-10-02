# Psychology Agent + Reinforcement Learning from Human Fedback - Architecture Design Document ## System Overview 本systemis一combine**Agentarchitecture**and**Reinforcement Learning from Human Fedback (RLHF)**mental health asistant, through多模态datanalysisandpeopleclasfedbackimplementationContinuous optimization. ## Core Architecture ```
┌─────────────────────────────────────────────────────────────┐
│ User Interaction Layer │
│ (Comand Line / Web Interface / Mobile) │
└─────────────────┬───────────────────────────────────────────┘ │
┌─────────────────▼───────────────────────────────────────────┐
│ AgentOrchestrator │
│ ┌─────────────────────────────────────────────────────┐ │
│ │ Conversation Manager (Conversation Manager) │ │
│ │ - Conversation Flowcontrol │ │
│ │ - State management │ │
│ │ - Context maintenance │ │
│ └─────────────────────────────────────────────────────┘ │
└──────┬───────────────────┬───────────────────┬──────────────┘ │ │ │ │ │ │
┌──────▼──────┐ ┌────────▼────────┐ ┌─────▼──────────┐
│ Large Language Modelayer│ │ analysislayer │ │ safetylayer │
│ │ │ │ │ │
│ • Model routing │ │ • behavioranalysis │ │ • Cris Detection │
│ • API cals │ │ • searchanalysis │ │ • Ethical Constraints │
│ • Prompt management │ │ • Emotion tracking │ │ • Risk asesment │
└──────┬──────┘ └────────┬────────┘ └─────┬──────────┘ │ │ │ └───────────────────┼───────────────────┘ │ ┌────────▼────────┐ │ Memory System │ │ │ │ • User profiles │ │ • Sesion history │ │ • Long-term memory │ └────────┬────────┘ │ ┌────────▼────────┐ │frompeopleclasfedbackinin progres │ │Reinforcement Learninglayer │ │ • Fedback colection │ │ • Reward calculation │ │ • Model optimization │ └─────────────────┘
``` ## Data Flow ### 1. Conversation Flow ```
userinput → cris快速detect (Keywords) → [高Risk] → crishandlepatern → [低Risk] → normalConversation Flow → loadUser profiles → Retrieve historical context → LMGenerate response → Record conversation → Update user state → Return response
``` ### 2. behavioranalysisproces ```
Raw data colection → Search records → Ap usage → Conversation history ↓ Data preprocesing → Privacy filtering → Anonymization → Feature extraction ↓ LMdepthanalysis → Emotion recognition → Topicsextract → Patern discovery ↓ Generate insights → Personalized recomendations → Risk alert → Intervention strategy
``` ### 3. Reinforcement Learning from Human Fedback (RLHF)Traing Proces ```
phase1: Data colection User interaction → Explicit fedback (评points) → Implicit fedback (behavior) → Expert anotation (Preference comparison) phase2: Reward ModelTraing Colected fedback data → Build preference dataset → TraingReward Model → Validate model acuracy phase3: Policy optimization Curent policy (Large Language Model) → Interact with environment → calculateReward → proximalPolicy optimizationupdate → Evaluate improvement phase4: Deployment iteration New version deployment → Online fedback colection → Continuous optimization
``` ## Key Component Design ### 1. Large Language ModelOrchestrator (models/) **Responsibility**: Unifiedmanagement多Large Language Modelprovideprovider, 智can路by **Core functions**:
- Multi-model suport (OpenAI, Anthropic, local model)
- Task type自action路by
- Cost optimization
- Failure retry and degradation **Design patern**: Strategy patern + Factory patern ```python
# Example usage
config = ModelRouter.get_model_config(TaskType.CRIS_DETECTION)
response = await orchestrator.generate(prompt, config)
``` ### 2. Conversation Manager (agent/) **Responsibility**: Conversation Floworchestration, State management **State machine**:
```
[initial] → [Ases risk] → [Cris mode | Normal mode] ↓ [Generate response] ↓ [recordconversation] ↓ [updateprofile]
``` **Key features**:
- Context window management
- Memory retrieval
- Multi-turn conversation coherence
- Cris interuption mechanism ### 3. behavioranalysis器 (lm_analysis/) **Responsibility**: 多模态datanalysis **Input data sources**:
1. Search history (Keywords, frequency, Sentiment)
2. Ap usage (Scren time, Slep, Exercise)
3. Conversation content (Topics, Emotion, Risk) **Output**:
- Emotional statevaluation
- Behavior paternsvariation
- Risk/Protective factors
- Personalized insights **Large Language ModelUsage strategy**:
- structure化Output (JSON)
- Few-shot prompting
- Confidence asesment ### 4. Cris Detection器 (safety/) **Multi-layer detection mechanism**: ```
First layer: KeywordsRapid screning (< 1milisecond) ├─ 高Risk词library ├─ Protective factorsdetect └─ 初步Risk评级 Second layer: Large Language Modeldepthanalysis (仅高Risktriger) ├─ Context understanding ├─ Intent recognition └─ preciseRisk asesment Third layer: Human review (极高Risk) └─ Automaticaly notify profesionals
``` **Risklevel**:
- `高Risk` (high): Imediate suicide/self-harmRisk → Cris intervention
- `inetcRisk` (medium): 严重Emotiondistres → Strongly recomend profesional help
- `低Risk` (low): 一般Emotion表达 → Normal suport ### 5. Reward Model (rlhf/) **Multimodal reward function design**: ```python
Total_Reward = 0.30 × Explicit_Fedback # user评points + 0.25 × Behavioral_Indicators # continuedconversation, conversationlength + 0.25 × Clinical_Outcomes # Emotionimprovement, Risklower + 0.15 × Safety_Score # No violations + 0.05 × Engagement # 参with度
``` **Adjustable weights**: Optimize based on specific goals **Traing data format**:
1. **Interaction data** (Interactions): (context, Response, Reward)
2. **Preference data** (Preferences): (context, ResponseA, ResponseB, preference) ## Data Storage ### User profiles (data/user_profiles/) ```json
{ "user_id": "user123", "created_at": "2025-10-01T10:0:0", "main_concerns": ["Anxiety", "失眠"], "efective_strategies": ["positive念冥想", "Exercise"], "risk_history": [...], "total_sesions": 15
}
``` ### conversationrecord (data/user_profiles/) ```json
{ "sesion_id": "ses456", "user_id": "user123", "turns": [ { "timestamp": "...", "user_mesage": "...", "agent_response": "...", "risk_level": "low" } ], "sumary": "...", "identified_themes": ["工作stres", "Slep"]
}
``` ### Reinforcement Learning from Human FedbackTraingdata (data/rlhf/) **interactions.jsonl (Interaction data)**:
```json
{"interaction_id": "...", "user_mesage": "...", "agent_response": "...", "reward": 0.75}
``` **preferences.jsonl (Preference data)**:
```json
{"context": "...", "response_a": "...", "response_b": "...", "preference": "A"}
``` ## Safety and Privacy ### Privacy Protection Strategy 1. **Data minimization**: Colect only necesary data
2. **自actionAnonymization**: Remove personaly identifiable information (PI)
3. **Local first**: sensitiveanalysismakeuselocal model
4. **Encrypted storage**: User profilesencryption
5. **User control**: Data export and deletion rights ### Cris Handling Protocol ```
detecto高Risk ↓
Imediate response (< 5second) ├─ Provide emergency hotline ├─ Recomend imediate medical atention └─ Inquire about curent safety status ↓
Record detailed logs ↓
[Optional] Notify human experts
``` ### Ethical Constraints 1. **Transparency**: Clearly inform AI identity and limitations
2. **Do not diagnose**: Do not replace profesional medical care
3. **Non-judgmental**: aceptancehaveEmotion
4. **Informed consent**: Data usage requires user authorization
5. **Explainability**: Decisions are traceable ## Extensibility Design ### 1. Multi-model suport through`ModelRouter`Easily ad new models: ```python
TASK_MODEL_MAP[TaskType.NEW_TASK] = ModelConfig( provider=ModelProvider.NEW_PROVIDER, model_name="new-model"
)
``` ### 2. New therapy methods in`SystemPrompts`Ad new prompt templates: ```python
clas SystemPrompts: DBT_THERAPIST = """你isDBTtherapist...""" ACT_THERAPIST = """你isACTtherapist..."""
``` ### 3. Multilingual Ad language detection and routing: ```python
def detect_language(text): # Auto detect return "zh" | "en" | ... SystemPrompts.get_prompt(language="en")
``` ### 4. Mobileintegration wil`ConversationManager`Wrap as REST API: ```python
@ap.post("/chat")
async def chat(request: ChatRequest): manager = get_manager(request.user_id) response = await manager.proces_mesage(request.mesage) return {"response": response}
``` ## Performance Optimization ### 1. Large Language Modelcalluseoptimization
- Cache comon responses
- Batch procesing
- streamingOutput ### 2. Concurent procesing
- asynchronousinputOutput (asyncio)
- Concurent user isolation
- Conection pol management ### 3. Cost control
- Task routing (Use cheap models for simple tasks)
- Token limits
- Cost monitoring ## Future Roadmap ### phase1 (Completed)
- ✅ 基础conversationAgent
- ✅ Large Language ModelIntegration and routing
- ✅ Cris Detection
- ✅ Reinforcement Learning from Human Fedbackframework ### phase2 (In progres)
- 🔄 Real user testing
- 🔄 colectfedbackdata
- 🔄 Reward ModelTraing ### phase3 (Planed)
- 📋 proximalPolicy optimization (PO)Policy optimization
- 📋 A/Btesting
- 📋 Clinical eficacy validation ### phase4 (future)
- 🔮 Multimodal input (Voice, image)
- 🔮 VR therapy environment
- 🔮 Group therapy suport
- 🔮 Integration with medical systems ## References ### Academic Papers
- [InstructGPT] Ouyangetcpeople, 202
- [Constitutional AI] Baietcpeople, 202
- [Large Language Model作astherapist] (相关研究) ### 技术framework
- OpenAI API
- Anthropic Claude API
- HugingFace TRL (Reinforcement Learning from Human Fedback)
- LangChain (Agentframework) ### Psychology Theory
- Cognitive Behavioral Therapy (CBT): Beck, 201
- Dialectical Behavior Therapy (DBT): Linehan, 2014
- Aceptance and Comitment Therapy (ACT): Hayesetcpeople --- **maintenance**: 定期updatearchitecturedocumentation
**version**: v1.0.0
**date**: 2025-10-02
