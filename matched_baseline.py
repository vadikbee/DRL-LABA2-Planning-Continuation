import json
from pathlib import Path
from algorithms import train_taxi
p=Path('docs/results/feedback_experiment.json');d=json.loads(p.read_text(encoding='utf8'))
r=train_taxi('q_learning',episodes=40,seed=2,evaluation_episodes=20)
d['mcts']['q_learning_40']={'rewards':r.rewards.tolist(),'steps':r.steps.tolist(),'evaluation_reward':r.evaluation_reward,'success_rate':r.success_rate,'seconds':r.duration}
p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf8')
print('Q 40:',r.evaluation_reward,r.success_rate,r.duration)
