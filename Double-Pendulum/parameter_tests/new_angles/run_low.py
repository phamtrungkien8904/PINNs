import subprocess,sys,os,json,concurrent.futures
jobs=[('cutoff12',dict(MAX_ANGULAR_FREQUENCY=12.)),('cutoff16',dict(MAX_ANGULAR_FREQUENCY=16.)),('cutoff8',dict(MAX_ANGULAR_FREQUENCY=8.))]
def run(j):
 name,params=j
 p=subprocess.run([sys.executable,'parameter_tests/run_new_trial.py',name,'fourier','0','30000'],env=dict(os.environ,TRIAL_PARAMS=json.dumps(params)),capture_output=True,text=True)
 print(p.stdout,p.stderr,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(run,jobs))
