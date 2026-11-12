import json
from pathlib import Path
from algorithms import train_taxi,td_lambda_errors,TAXI_ENV_ID
from nfq import train_nfq
out=Path('docs/results/feedback_experiment.json')
data={'environment':TAXI_ENV_ID,'seed':2,'mcts':{},'nfq':{}}
for method,episodes in [('q_learning',80),('mcts',40)]:
 r=train_taxi(method,episodes=episodes,seed=2,simulations=5,depth=5,evaluation_episodes=20)
 data['mcts'][method]={'rewards':r.rewards.tolist(),'steps':r.steps.tolist(),'evaluation_reward':r.evaluation_reward,'success_rate':r.success_rate,'seconds':r.duration}
 out.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
 print(method,r.evaluation_reward,r.duration,flush=True)
for epsilon in (0.1,0.3,0.5):
 r=train_nfq(epochs=10,episodes_per_epoch=10,epsilon=epsilon,seed=2,evaluation_episodes=20)
 data['nfq'][str(epsilon)]={'rewards':r.rewards.tolist(),'losses':r.losses.tolist(),'evaluation_reward':r.evaluation_reward,'success_rate':r.success_rate,'seconds':r.duration}
 out.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
 print('NFQ',epsilon,r.evaluation_reward,r.duration,flush=True)
