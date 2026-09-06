import pathlib,json,csv,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
root=pathlib.Path(__file__).resolve().parent
results=[json.loads(p.read_text()) for p in sorted(root.glob('*/result.json'))]
fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
summary=[]
for d in results:
 h=d['history']; last=h[-1]; x=[r['evaluations'] for r in h]
 worst=[min(r['r2_full']+r['r2_extrapolation']) for r in h]
 axes[0].semilogy(x,np.maximum(1-np.array(worst),1e-10),label=d['name'])
 axes[1].semilogy(x,[r['physics'] for r in h],label=d['name'])
 first=next((r['evaluations'] for r in h if min(r['r2_full']+r['r2_extrapolation'])>=.999 and r['physics']<=1e-5),None)
 summary.append(dict(name=d['name'],status=d['status'],updates=last['evaluations'],first_target=first,worst_r2=worst[-1],physics=last['physics'],seconds=d['elapsed_seconds']))
axes[0].axhline(.001,color='black',ls=':');axes[1].axhline(1e-5,color='black',ls=':')
axes[0].set(ylabel='1 − worst R² (lower is better)',xlabel='Optimizer updates',title='Full and extrapolation accuracy')
axes[1].set(ylabel='Normalized physics residual',xlabel='Optimizer updates',title='Physics accuracy')
axes[0].legend(fontsize=8);fig.savefig(root/'convergence.png',dpi=180);plt.close(fig)
with (root/'summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
winner=root/'fourier_seed0'; s=np.load(winner/'snapshots.npz')
t=s['t']; ref=s['reference']; pred=s['predictions']; epochs=s['epochs']
fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True,layout='constrained'); lines=[]
for i,ax in enumerate(axes):
 ax.plot(t,ref[:,i],color='black',lw=1,label='Reference');line,=ax.plot(t,pred[-1,:,i],color='tab:blue',ls='--',label='FPINN');lines.append(line)
 ax.scatter(t[:300:10],ref[:300:10,i],s=12,color='tab:orange',label='Training data')
 ax.axvline(t[300],color='gray',ls=':',label='Extrapolation begins');ax.set_ylabel(f'Angle {i+1} (rad)');ax.legend(ncol=4,fontsize=8)
axes[-1].set_xlabel('Time (s)');fig.suptitle(f'FPINN — {epochs[-1]:,} updates');fig.savefig(root/'fourier_prediction.png',dpi=180)
def update(k):
 for i,line in enumerate(lines):line.set_ydata(pred[k,:,i])
 fig.suptitle(f'FPINN — {epochs[k]:,} updates');return lines
FuncAnimation(fig,update,frames=len(epochs),interval=200).save(root/'fourier_training.gif',writer=PillowWriter(fps=5));plt.close(fig)
d=next(d for d in results if d['name']=='fourier_seed0')
fig,ax=plt.subplots(figsize=(9,4),layout='constrained')
for key in ['total','data','physics','initial','energy']:
 ax.semilogy([r['evaluations'] for r in d['history']],[r['losses'][key] for r in d['history']],label=key)
ax.set(xlabel='Optimizer updates',ylabel='Loss',title='FPINN training losses (sampled every 1,000 updates)');ax.legend();fig.savefig(root/'fourier_loss.png',dpi=180);plt.close(fig)
lines=['# Parameter-only convergence test','', 'These are fresh CPU reruns. Earlier test artifacts were unavailable, so conclusions below use only the saved results in this folder. Production scripts were not edited.','', '## Method','', 'FPINN uses the existing fpinn2.py algorithm: four real spectral outputs, trainable Fourier-bin coefficients plus the original MLP correction, FFT reconstruction and spectral derivatives. TPINN uses tpinn2_ver2.py: time input, two angle outputs and second derivatives from autograd. The current tpinn2.py has a four-state model, so it was not used for this requested two-output comparison.','', 'Both use the same 30 observations (rows 0:300:10). R² is evaluated separately for each angle over the entire reference and the extrapolation region (rows 300 onward). Success requires all four R² values ≥ 0.999 and mean((f1/10)²+(f2/10)²) ≤ 1e-5 at three consecutive checks, spaced 1,000 optimizer updates apart. This is an evaluation stop, not a changed training algorithm. Epoch counts below mean optimizer updates; the original loops number their first update as epoch zero.','', '## Results','', '| Run | Status | Updates | First target | Worst R² | Physics |','|---|---|---:|---:|---:|---:|']
for r in summary:lines.append(f"| {r['name']} | {r['status']} | {r['updates']:,} | {r['first_target'] or 'not reached'} | {r['worst_r2']:.8f} | {r['physics']:.5g} |")
lines+=['','## Tested Fourier candidate','', '```python','FOURIER_PERIOD_FACTOR = 2','MAX_ANGULAR_FREQUENCY = 20.0','LAMBDA_PHYSICS = 1000.0','WARMUP_EPOCHS = 1000','PHYSICS_RAMP_EPOCHS = 4000','LEARNING_RATE_NETWORK = 2e-4','LEARNING_RATE_SPECTRUM = 2e-4','WEIGHT_DECAY = 1e-8','LAMBDA_DATA = 1000.0','LAMBDA_INITIAL = 500.0','# Original MultiStepLR, milestones: [10000, 30000, 60000], gamma=0.3','```','', 'The Fourier baseline uses the same settings except period factor 4, cutoff 12 and physics weight 10. Therefore the within-Fourier improvement isolates these three parameter changes together. The full candidate above differs in additional parameters from the current production file. TPINN uses width 128, three hidden tanh layers, 1,024 physics points, AdamW learning rate 2e-4, weight decay 1e-8, data weight 1000, initial weight 500, physics weight 0.1, energy weight 0.001. All original loss equations and optimizer algorithms are retained.','', '## Limits and artifacts','', 'This is a candidate versus one TPINN parameter baseline, not proof that every optimized TPINN needs more epochs. A budget-exhausted run has no measured convergence epoch. Runtime values in summary.csv include observation overhead and concurrent CPU contention; they are not an A5000 speed comparison. HPC scripts and wider GPU models were not tested.','', 'Each run includes source_snapshot.py, result.json (parameters and source hash), training.log, prediction.csv and snapshots.npz. run_trial.py applies parameter overrides and adds an observation callback to a copy of the source in memory. No production file is imported under its main name or edited.','', 'Plots: [convergence](convergence.png), [final fit](fourier_prediction.png), [training animation](fourier_training.gif), [loss](fourier_loss.png).']
(root/'RESULTS.md').write_text('\n'.join(lines)+'\n')
for d in results:
 src=root.parent/('fpinn2.py' if d['model']=='fourier' else 'tpinn2_ver2.py')
 assert hashlib.sha256(src.read_bytes()).hexdigest()==d['source_sha256'],src
print(json.dumps(summary,indent=2))
