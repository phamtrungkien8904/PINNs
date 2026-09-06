"""Observe original training loops without editing production source files."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import ast, json, pathlib, types, sys, time, hashlib, contextlib
import numpy as np
import torch
ROOT=pathlib.Path(__file__).resolve().parents[1]
os.chdir(ROOT)
name=sys.argv[1]; kind=sys.argv[2]; seed=int(sys.argv[3]); budget=int(sys.argv[4])
out=ROOT/'parameter_tests'/'new_angles'/name; out.mkdir(exist_ok=True)
source=ROOT/('fpinn2.py' if kind=='fourier' else 'tpinn2_ver2.py')
raw=source.read_text(); (out/'source_snapshot.py').write_text(raw)
tree=ast.parse(raw)
class Observe(ast.NodeTransformer):
 def visit_For(self,node):
  self.generic_visit(node)
  if isinstance(node.target,ast.Name) and node.target.id=='epoch':
   node.body += ast.parse('_observe(locals())').body
  return node
# Learning-rate schedule values are parameters; preserve the scheduler itself.
if kind=='fourier':
 for node in ast.walk(tree):
  if isinstance(node,ast.keyword) and node.arg=='milestones': node.value=ast.parse(os.environ.get('TRIAL_MILESTONES','[10000,30000,60000]'),mode='eval').body
if kind=='time':
 for node in ast.walk(tree):
  if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and node.func.attr=='Linear':
   for arg in node.args:
    if isinstance(arg,ast.Constant) and arg.value==64:arg.value=128
module=types.ModuleType(name); module.__file__=str(source)
exec(compile(ast.fix_missing_locations(Observe().visit(tree)),str(source),'exec'),module.__dict__)
torch.set_num_threads(1)
params=dict(SEED=seed,EPOCHS=budget-1,DATA_STOP=300,DATA_STEP=10,WEIGHT_DECAY=1e-8,LAMBDA_DATA=1000.,LAMBDA_INITIAL=500.,PRINT_EVERY=1000,SNAPSHOT_EVERY=1000000)
if kind=='fourier':
 params.update(WARMUP_EPOCHS=1000,PHYSICS_RAMP_EPOCHS=4000,LEARNING_RATE_NETWORK=2e-4,LEARNING_RATE_SPECTRUM=2e-4,LAMBDA_PHYSICS=10.,FOURIER_PERIOD_FACTOR=4,MAX_ANGULAR_FREQUENCY=12.)
 if name!='fourier_baseline': params.update(FOURIER_PERIOD_FACTOR=2,MAX_ANGULAR_FREQUENCY=20.,LAMBDA_PHYSICS=1000.)
else: params.update(NETWORK_WIDTH=128,PHYSICS_POINTS=1024,LEARNING_RATE=2e-4,LAMBDA_PHYSICS=.1,LAMBDA_ENERGY=.001)
params.update(json.loads(os.environ.get('TRIAL_PARAMS','{}')))
module.__dict__.update(params); module.OUTPUT_DIR=out
for key in list(module.__dict__):
 if key.startswith('save_'): module.__dict__[key]=lambda *a,**kw:None
torch.manual_seed(seed); np.random.seed(seed)
rows=[]; snapshots=[]; consecutive=0; status='budget_exhausted'; start=time.perf_counter()
class Reached(Exception):pass
def observe(v):
 global consecutive
 step=v['epoch']+1
 if step%1000 and step!=budget:return
 model=v['model']
 if kind=='fourier':
  with torch.no_grad():
   spectrum,pred=module.reconstruct(model,v['omega_input'],v['total_modes'],v['n_fourier'],v['n_time'])
   vel,acc=module.spectral_derivatives(spectrum,v['omega'],v['n_fourier'],v['n_time'])
   f1,f2=module.explicit_physics_residuals(pred,vel,acc)
  t=v['t']
 else:
  t=v['t_ref']; tt=torch.tensor(t,dtype=torch.float32)[:,None].requires_grad_(True)
  pred,_,_,_,f1,f2=module.explicit_physics_residuals(model,tt)
 physics=torch.mean((f1/10)**2+(f2/10)**2).item()
 pred=pred.detach().numpy(); ref=np.column_stack((v['theta1_ref'],v['theta2_ref']))
 def r2(a,b):return (1-((a-b)**2).sum(axis=0)/((a-a.mean(axis=0))**2).sum(axis=0)).tolist()
 full=r2(ref,pred); extra=r2(ref[params['DATA_STOP']:],pred[params['DATA_STOP']:]); good=min(full+extra)>=.999 and physics<=1e-5
 consecutive=consecutive+1 if good else 0
 row=dict(evaluations=step,seconds=time.perf_counter()-start,r2_full=full,r2_extrapolation=extra,physics=physics,losses={k:v[k+'_loss'].item() for k in ['total','data','physics','initial','energy']})
 rows.append(row); snapshots.append(pred.copy())
 (out/'progress.json').write_text(json.dumps(row,indent=2))
 np.savetxt(out/'prediction.csv',np.column_stack((t,ref,pred)),delimiter=',',header='time,theta1,theta2,pred1,pred2',comments='')
 np.savez(out/'snapshots.npz',t=t,reference=ref,predictions=np.array(snapshots),epochs=[r['evaluations'] for r in rows])
 if consecutive>=3:raise Reached()
module._observe=observe
try:
 with (out/'training.log').open('w') as log,contextlib.redirect_stdout(log):module.main()
except Reached:status='converged'
finally:
 (out/'result.json').write_text(json.dumps(dict(name=name,model=kind,params=params,scheduler_milestones=json.loads(os.environ.get('TRIAL_MILESTONES','[10000,30000,60000]')) if kind=='fourier' else None,data_sha256=hashlib.sha256(module.DATA_FILE.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(raw.encode()).hexdigest(),status=status,history=rows,elapsed_seconds=time.perf_counter()-start),indent=2))
print(name,status,rows[-1] if rows else 'no observations')
