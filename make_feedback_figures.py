import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from algorithms import td_lambda_errors
p=Path('docs/results/feedback_experiment.json'); d=json.loads(p.read_text(encoding='utf-8'))
out=Path('docs/figures'); out.mkdir(exist_ok=True)
fig,ax=plt.subplots(figsize=(10,5.5),layout='constrained')
for k,label,color in [('q_learning_40','Q-learning, 40 эпизодов','#0072b2'),('mcts','ПДМК, 40 эпизодов','#d55e00')]:
 y=np.asarray(d['mcts'][k]['rewards']); ax.plot(np.arange(1,len(y)+1),y,color=color,alpha=.22); n=5; ax.plot(np.arange(n,len(y)+1),np.convolve(y,np.ones(n)/n,'valid'),label=label,color=color)
ax.set(title='Taxi: награда при обучении, seed=2',xlabel='Эпизод',ylabel='Награда за эпизод');ax.legend();ax.grid(alpha=.25);fig.savefig(out/'mcts_learning.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(10,5.5),layout='constrained')
for eps,v in d['nfq'].items(): ax.plot(range(1,11),v['rewards'],marker='o',label=f'ε = {eps}')
ax.set(title='NFQ: награда при сборе переходов, seed=2',xlabel='Эпоха (10 эпизодов)',ylabel='Средняя награда в эпохе');ax.legend();ax.grid(alpha=.25);fig.savefig(out/'nfq_epsilon.png',dpi=160);plt.close(fig)
ls=(0.0,0.8,1.0); alphas=(0.0,0.1,0.2,0.3,0.4)
d['td_lambda']={}
fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained',sharey=True)
for ax,method,title in zip(axes,['forward','backward'],['Прямое представление','Обратное представление']):
 data=td_lambda_errors(method,ls,alphas,runs=5,episodes=5,seed=2); d['td_lambda'][method]={str(k):v.tolist() for k,v in data.items()}
 for lam,y in data.items(): ax.plot(alphas,y,marker='o',label=f'λ={lam}')
 ax.set(title=title,xlabel='α',ylabel='Средняя RMS-ошибка');ax.grid(alpha=.25);ax.legend()
fig.suptitle('19 состояний: 5 запусков, 5 эпизодов');fig.savefig(out/'td_lambda.png',dpi=160);plt.close(fig)
d['td_lambda_parameters']={'lambdas':list(ls),'alphas':list(alphas),'runs':5,'episodes':5,'seed':2,'cap_rms':5}
p.write_text(json.dumps(d,ensure_ascii=False,indent=2),encoding='utf-8')
