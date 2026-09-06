import pathlib,json,csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
root=pathlib.Path(__file__).resolve().parent
runs=[json.loads(p.read_text()) for p in sorted(root.glob('*/result.json'))]
rows=[]
for d in runs:
 h=d['history'];last=h[-1]
 rows.append(dict(name=d['name'],status=d['status'],updates=last['evaluations'],data_stop=d['params']['DATA_STOP'],data_step=d['params']['DATA_STEP'],r2_full=min(last['r2_full']),r2_extrapolation=min(last['r2_extrapolation']),physics=last['physics']))
with (root/'summary.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
best=max(rows,key=lambda r:min(r['r2_full'],r['r2_extrapolation']))['name']
fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True,layout='constrained')
for name in dict.fromkeys(['current_sparse','previous_winner',best]):
 x=np.loadtxt(root/name/'prediction.csv',delimiter=',',skiprows=1)
 for j,ax in enumerate(axes):ax.plot(x[:,0],x[:,j+3],lw=1,label=name)
for j,ax in enumerate(axes):
 ax.plot(x[:,0],x[:,j+1],color='black',ls='--',lw=1,label='Reference');ax.set_ylabel(f'Angle {j+1} (rad)');ax.legend(fontsize=8)
axes[-1].set_xlabel('Time (s)');fig.savefig(root/'fit_comparison.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
for d in runs:ax.plot([r['evaluations'] for r in d['history']],[min(r['r2_full']+r['r2_extrapolation']) for r in d['history']],label=d['name'])
ax.set(ylim=(-.6,1.05),xlabel='Optimizer updates',ylabel='Worst R²',title='New initial angles: parameter screen');ax.axhline(.999,color='black',ls=':');ax.legend(fontsize=7,ncol=3);fig.savefig(root/'convergence.png',dpi=160);plt.close(fig)
s=np.load(root/best/'snapshots.npz');fig,axes=plt.subplots(2,1,figsize=(9,5),layout='constrained');lines=[]
for j,ax in enumerate(axes):
 ax.plot(s['t'],s['reference'][:,j],color='black',label='Reference');line,=ax.plot(s['t'],s['predictions'][-1,:,j],label=best);lines.append(line);ax.legend();ax.set_ylabel(f'Angle {j+1}')
def update(k):
 for j,line in enumerate(lines):line.set_ydata(s['predictions'][k,:,j])
 fig.suptitle(f"{best}: {s['epochs'][k]:,} updates")
FuncAnimation(fig,update,frames=len(s['epochs'])).save(root/'best_training.gif',writer=PillowWriter(fps=5));plt.close(fig)
fig,ax=plt.subplots(figsize=(9,4),layout='constrained');d=next(d for d in runs if d['name']==best)
for k in ['total','data','physics','initial','energy']:ax.semilogy([r['evaluations'] for r in d['history']],[r['losses'][k] for r in d['history']],label=k)
ax.legend();ax.set(xlabel='Optimizer updates',ylabel='Loss',title=best);fig.savefig(root/'best_loss.png',dpi=160);plt.close(fig)
text=['# Updated dataset: parameter tests','','The updated initial angles are −22.5° and +18°; the old data used −22.5° and −18°. The file has 2,002 rows over 0–20.01 seconds with uniform 0.01-second spacing. input.dat preserves the tested file; old_input.dat reconstructs the old reference from the previously saved prediction CSV.','','The production FPINN now uses DATA_STEP=30 (10 measurements in the first 3 seconds), whereas the previous successful benchmark used DATA_STEP=10 (30 measurements). The current learning rate, warmup, ramp and schedule also differ. `current_sparse` reproduces those current parameters; `current_script` restores 30 measurements only; `previous_winner` reproduces the full old winning configuration.','','All tests retain the existing fpinn2.py algorithm. Only parameter values and benchmark observations change. Production scripts and Outputs were not edited. Each result.json records parameter overrides, source and input hashes, and all 1,000-update observations; source_snapshot.py captures the remaining defaults. Training logs and predictions are in each run folder.','','Success requires both angle R² values ≥0.999 over the full record and the held-out region, plus normalized physics residual ≤1e-5, for three consecutive observations. For most runs the held-out region starts at row 300 (3 s); window10s runs start it at row 1000 (10 s). Those window runs still use 30 data points, spaced every 34 rows, but change the forecasting task and are not directly comparable.','','| Run | Updates | Worst full R² | Worst held-out R² | Physics | Status |','|---|---:|---:|---:|---:|---|']
for r in rows:text.append(f"| {r['name']} | {r['updates']:,} | {r['r2_full']:.6f} | {r['r2_extrapolation']:.6f} | {r['physics']:.4g} | {r['status']} |")
text+=['','A failed 30,000-update screen does not prove the configuration can never converge. These results do show that the old winning settings are not robust to the initial-condition change. CPU runs were concurrent, so their timings are not GPU benchmarks. No TPINN-versus-FPINN convergence advantage is established on this new data.','','Figures: [fit comparison](fit_comparison.png), [convergence](convergence.png), [best screened run animation](best_training.gif), [losses](best_loss.png). “Best” means highest final worst R² among screened runs; it does not imply successful convergence.']
(root/'RESULTS.md').write_text('\n'.join(text)+'\n');print(json.dumps(rows,indent=2))
