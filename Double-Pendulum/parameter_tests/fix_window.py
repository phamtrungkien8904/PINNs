import subprocess,sys,os,json,concurrent.futures
jobs=[('fix_window10_dense',{'DATA_STOP':1000,'DATA_STEP':10}),('fix_window15_dense',{'DATA_STOP':1500,'DATA_STEP':10}),('fix_window15_sparse',{'DATA_STOP':1500,'DATA_STEP':30}),('fix_unclipped',{'DATA_STEP':30,'GRADIENT_CLIP':1000.})]
def run(j):
 name,params=j
 p=subprocess.run([sys.executable,'parameter_tests/run_new_trial.py',name,'fourier','0','40000'],env=dict(os.environ,TRIAL_PARAMS=json.dumps(params),TRIAL_MILESTONES='[20000,30000,40000]'),capture_output=True,text=True)
 print(p.stdout,p.stderr,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(run,jobs))
