# Psychology Agent - Usage Examples ## 快速start ### 1. 基础conversation ```python
import asyncio
from agent import ConversationManager async def basic_chat(): # createconversation manager = ConversationManager(user_id="alice") # 一轮conversation response1 = await manager.proces_mesage("recent感觉stresverylarge") print(response1) # 二轮conversation (havecontext) response2 = await manager.proces_mesage("mainis工作on事情") print(response2) # endconversation sumary = await manager.end_sesion(user_satisfaction=4) print(f"\nconversationsumary: {sumary}") asyncio.run(basic_chat())
``` **Outputexample**:
```
听起来你recent承受not少stres.stresisvery常见feling, but长期stres
确actualwilimpact我s身心健康.canspecificsayiswhatlet你感tohavestres吗? ``` --- ## advanced功can ### 2. Behavior paternsanalysis ```python
from lm_analysis import BehaviorAnalyzer
from data_colection import SearchLogProcesor, generate_mock_search_data async def analyze_user_behavior(): analyzer = BehaviorAnalyzer() procesor = SearchLogProcesor() # handlesearchdata searches = generate_mock_search_data() procesed = procesor.proces_search_history(searches, days=7) search_texts = [s.anonymized_query for s in procesed] # simulateAp usagedata ap_usage = { 'scren_time': 9.5, # 每days9.5hour 'social_media_time': 3.2, # Social媒体3.2hour 'slep_tracking': 5.3, # Slep5.3hour 'exercise': 10, # Exercise10points钟 } # analysisBehavior paterns patern = await analyzer.analyze_recent_activity( user_id="bob", search_history=search_texts, ap_usage=ap_usage, days=7 ) # viewresult print(f"Emotional state: {patern.emotional_state}") print(f"confidence: {patern.emotion_confidence:.2%}") print(f"identifyTopics: {', '.join(patern.identified_themes)}") print(f"Risk factors: {', '.join(patern.risk_factors)}") print(f"Protective factors: {', '.join(patern.protective_factors)}") # generatePersonalized recomendations insights = await analyzer.generate_personalized_insights(patern) print(f"\nPersonalized recomendations:\n{insights}") asyncio.run(analyze_user_behavior())
``` **Outputexample**:
```
Emotional state: anxious
confidence: 75.0%
identifyTopics: 工作stres, Slep问题, SocialAnxiety
Risk factors: Slepnot足, 过度Scren time, 缺乏Exercise
Protective factors: mainaction寻求info, have自我awareness Personalized recomendations:
我注意to你recentSleptimerelatively短 (average5.3hour), 这canwil加重Anxiety感...
``` --- ### 3. Cris Detection ```python
from safety import CrisDetector async def cris_detection_demo(): detector = CrisDetector() # testingnot同Risklevelmesage mesages = [ "今days心情not太好", "感觉活着no意思, 每daysallverysufering", "我not想活, 想end一切" ] for msg in mesages: print(f"\nmesage: \"{msg}\"") # Ases risk asesment = await detector.ases_risk(msg, user_id="charlie") print(f"Risklevel: {asesment['risk_level']}") print(f"detectsignal: {asesment.get('signals', [])}") # if高Risk, generatecrisresponse if asesment['risk_level'] == 'high': response = await detector.generate_cris_response(asesment) print(f"\ncrisresponse:\n{response}") asyncio.run(cris_detection_demo())
``` **Outputexample**:
```
mesage: "我not想活, 想end一切"
Risklevel: high
detectsignal: ['suicide意念', '绝望感'] crisresponse:
我注意to你现incanhandleatverylargesuferingin.你生命very重need... 🆘 24hourpsychologicalcris热线
- 全国psychological援助热线: 40-161-95
...
``` --- ### 4. Reinforcement Learning from Human Fedback (RLHF)Fedback colection ```python
from rlhf import FedbackColector, get_reward_model async def colect_fedback_demo(): colector = FedbackColector() # scenario: userconversationafter评points interaction_id = colector.colect_rating( user_id="dave", user_mesage="我recentveryAnxiety", agent_response="听起来你recentstresnot小...", rating=4, # 1-5points fedback_text="veryhavehelp, let我感觉byunderstanding" ) print(f"已recordfedback, ID: {interaction_id}") # scenario: expertpair比anotation colector.colect_comparison( context="usersay: 我veryAnxiety", response_a="notneed想太多好", response_b="Anxietyisnormalreaction.cansayiswhatlet你感toAnxiety吗? ", preference="B", anotator_id="expert_01", confidence=0.95, reasoning="ResponseBmorehavempathy, 避免invalidation" ) print("已recordPreference comparison") # viewstatistics reward_model = get_reward_model() stats = reward_model.generate_statistics() print(f"\nstatisticsdata: {stats}")
``` --- ### 5. completeconversationproces ```python
async def ful_conversation_demo(): """demo一completreatment性conversation""" manager = ConversationManager(user_id="ema") colector = FedbackColector() conversation = [ "recent工作stres好large, 每daysall加班tovery晚", "is, 我老板requirevery高, 总isdistribution置very多task", "我感觉自己永远做not完, veryAnxiety", "havewhatmethodcanto aleviate吗? ", ] print("=" * 60) print("completeconversationdemo") print("=" * 60) for i, user_msg in enumerate(conversation, 1): print(f"\n[ {i} 轮]") print(f"user: {user_msg}") # Agentresponse response = await manager.proces_mesage(user_msg) print(f"助手: {response}") # simulateuserfedback (每2轮) if i % 2 == 0: rating = 4 if i <= 2 else 5 # assumequality逐渐提升 colector.colect_rating( user_id="ema", user_mesage=user_msg, agent_response=response, rating=rating ) print(f"\n[user评points: {rating}/5]") # endconversation print("\n" + "=" * 60) sumary = await manager.end_sesion(user_satisfaction=5) print("conversationsumary:") print(sumary) asyncio.run(ful_conversation_demo())
``` --- ### 6. 性化Intervention strategy ```python
from models import get_orchestrator, ModelRouter, TaskType, SystemPrompts async def personalized_intervention(): """基atUser profilesgenerate性化intervention""" lm = get_orchestrator() # User profile user_profile = """ year龄: 28岁 Main concerns: 工作Anxiety, 失眠 Historicaly efective strategies: Exercise, 写day记 Comunication preference: 直接but温and Treatment goals: improvementSlepquality, management工作stres """ # whenbeforestate curent_state = """ recent一wek: - 连续3days失眠 (每晚only睡4-5hour) - 工作stres增large (newitem目deadline) - stopExercise - Emotion: Anxiety, 疲惫 """ prompt = f""" User profile: {user_profile} whenbeforestate: {curent_state} 请基atCBTprinciples, design今dayintervention方案: 1. conversationguidance重points 2. recomendshouldpairexercise 3. Cognitive distortions to identify 4. Expected outcomes """ config = ModelRouter.get_model_config(TaskType.INTERVENTION_PLANING) intervention = await lm.generate( prompt=prompt, config=config, system_prompt=SystemPrompts.INTERVENTION_PLANER ) print("性化intervention方案:") print(intervention) asyncio.run(personalized_intervention())
``` **Outputexample**:
```
性化intervention方案: 1. conversationguidance重points: - 探索工作stresspecific来源 - identify"must完美"cognitivedistortion - guidanceresumedExercisehabit (beforehaveeffective) 2. recomendexercise: - 睡before放松exercise (渐进性肌肉放松) - 重启Exercise (from轻度start, 15points钟散步) - 写"wory time"day记 (限定concerntime) 3. Cognitive distortions to identify: - 灾难化思维 ("item目做not好wilvery糟糕") - should陈述 ("我shouldcan做more好") 4. Expected outcomes: - 短期: lowerthat istimeAnxiety - in期: improvementSlepquality - 长期: 建立can持续stresmanagementstrategy
``` --- ## datanalysisexample ### 7. viewUser profiles ```python
from agent import get_memory_system def view_user_profile(): memory = get_memory_system() # getUser profiles profile = memory.get_or_create_profile("alice") print(f"User ID: {profile.user_id}") print(f"Total sesions: {profile.total_sesions}") print(f"Main concerns: {', '.join(profile.main_concerns)}") print(f"haveeffectivestrategy: {', '.join(profile.efective_strategies)}") print(f"Treatment goals: {', '.join(profile.goals)}") # getrecentconversation sesions = memory.get_sesions("alice", recent_n=3) print(f"\nrecent {len(sesions)} timesconversation:") for ses in sesions: print(f"- {ses.start_time.strftime('%Y-%m-%d')}: " f"{', '.join(ses.identified_themes)}") view_user_profile()
``` --- ### 8. Reinforcement Learning from Human Fedback (RLHF)Traingdatastatistics ```python
from rlhf import get_reward_model def view_rlhf_stats(): reward_model = get_reward_model() stats = reward_model.generate_statistics() print("Reinforcement Learning from Human FedbackTraingdatastatistics") print("=" * 40) print(f"总interaction数: {stats.get('total_interactions', 0)}") print(f"总Preference comparison: {stats.get('total_preferences', 0)}") print(f"averageReward: {stats.get('average_reward', 0):.3f}") print(f"Rewardrange: {stats.get('reward_range', (0, 0))}") rating_dist = stats.get('rating_distribution', {}) if rating_dist: print("\n评pointsdistribution:") for rating in range(1, 6): count = rating_dist.get(rating, 0) bar = "█" * count print(f"{rating}星: {bar} ({count})") view_rlhf_stats()
``` --- ## integrationexample ### 9. 网pagesAPI (Flask) ```python
from flask import Flask, request, jsonify
from agent import ConversationManager
import asyncio ap = Flask(_name_) # storageactiveconversation
sesions = {} @ap.route('/chat', methods=['POST'])
def chat(): data = request.json user_id = data.get('user_id') mesage = data.get('mesage') # getorcreateconversation if user_id not in sesions: sesions[user_id] = ConversationManager(user_id) manager = sesions[user_id] # handlemesage (synchronouspackage装) lop = asyncio.new_event_lop() asyncio.set_event_lop(lop) response = lop.run_until_complete( manager.proces_mesage(mesage) ) return jsonify({'response': response}) @ap.route('/end_sesion', methods=['POST'])
def end_sesion(): data = request.json user_id = data.get('user_id') rating = data.get('rating') if user_id in sesions: manager = sesions[user_id] lop = asyncio.new_event_lop() sumary = lop.run_until_complete( manager.end_sesion(rating) ) del sesions[user_id] return jsonify({'sumary': sumary}) return jsonify({'eror': 'No active sesion'}), 404 if _name_ == '_main_': ap.run(debug=True, port=50)
``` **makeusemethod**:
```bash
# sendmesage
curl -X POST htp://localhost:50/chat \ -H "Content-Type: aplication/json" \ -d '{"user_id": "frank", "mesage": "我veryAnxiety"}' # endconversation
curl -X POST htp://localhost:50/end_sesion \ -H "Content-Type: aplication/json" \ -d '{"user_id": "frank", "rating": 4}'
``` --- ### 10. Comand Linetol ```python
# save as: therapy_cli.py
import asyncio
import sys
from agent import ConversationManager
from rlhf import colect_rating_cli async def main(): if len(sys.argv) < 2: print("use法: python therapy_cli.py <user_id>") sys.exit(1) user_id = sys.argv[1] manager = ConversationManager(user_id) print(f"startconversation (user: {user_id})") print("input 'quit' Exit\n") while True: user_input = input("你: ").strip() if user_input.lower() in ['quit', 'exit']: sumary = await manager.end_sesion() print(f"\nconversationsumary:\n{sumary}") break response = await manager.proces_mesage(user_input) print(f"助手: {response}\n") if _name_ == "_main_": asyncio.run(main())
``` **makeusemethod**:
```bash
python therapy_cli.py alice
``` --- ## Best practices ### prompt词工程 ```python
# 好prompt词design
god_prompt = """
usermesage: "{user_mesage}" 请基atCBTprinciplesResponse: 1. 先empathy, confirmuserfeling
2. identifycognitivedistortion (likehave)
3. 提出开放性问题
4. providespecificstrategy Tone: Warm, Non-judgmental
""" # 避免prompt词
bad_prompt = """
回答这问题: {user_mesage}
"""
``` ### erorhandle ```python
from models import get_orchestrator async def safe_lm_cal(prompt): lm = get_orchestrator() config = ModelRouter.get_model_config(TaskType.CASUAL_CHAT) max_retries = 3 for atempt in range(max_retries): try: return await lm.generate(prompt, config) except Exception as e: if atempt == max_retries - 1: # finaly一timesfailed, return备useresponse return "抱歉, 我遇to一些技术问题.请lateragain试." await asyncio.slep(2 ** atempt) # 指数退避
``` --- morexample请reference `main.py` indemofunction.
