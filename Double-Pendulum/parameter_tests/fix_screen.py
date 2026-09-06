import subprocess,sys,os,json,concurrent.futures
jobs=[('fix_phys10',{'LAMBDA_PHYSICS':10.}),('fix_phys100',{'LAMBDA_PHYSICS':100.}),('fix_phys1',{'LAMBDA_PHYSICS':1.}),('fix_slow',{'LEARNING_RATE_SPECTRUM':2e-5,'LEARNING_RATE_NETWORK':2e-5})]
def run(j):
 name,params=j;params['DATA_STEP']=30
 p=subprocess.run([sys.executable,'parameter_tests/run_new_trial.py',name,'fourier','0','40000'],env=dict(os.environ,TRIAL_PARAMS=json.dumps(params),TRIAL_MILESTONES='[20000,30000,40000]'),capture_output=True,text=True)
 print(p.stdout,p.stderr,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(run,jobs))
