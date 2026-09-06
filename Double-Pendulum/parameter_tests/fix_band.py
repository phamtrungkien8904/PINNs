import subprocess,sys,os,json,concurrent.futures
jobs=[('fix_window10_f40',{'DATA_STOP':1000,'DATA_STEP':10,'MAX_ANGULAR_FREQUENCY':40.}),('fix_window15_f40',{'DATA_STOP':1500,'DATA_STEP':10,'MAX_ANGULAR_FREQUENCY':40.}),('fix_window18_f40',{'DATA_STOP':1800,'DATA_STEP':10,'MAX_ANGULAR_FREQUENCY':40.})]
def run(j):
 name,params=j
 p=subprocess.run([sys.executable,'parameter_tests/run_new_trial.py',name,'fourier','0','40000'],env=dict(os.environ,TRIAL_PARAMS=json.dumps(params),TRIAL_MILESTONES='[20000,30000,40000]'),capture_output=True,text=True)
 print(p.stdout,p.stderr,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(run,jobs))
