import subprocess,sys,os,json,concurrent.futures
jobs=[('physics10000',dict(LAMBDA_PHYSICS=10000.)),('cutoff50',dict(MAX_ANGULAR_FREQUENCY=50.)),('cutoff50_physics10000',dict(MAX_ANGULAR_FREQUENCY=50.,LAMBDA_PHYSICS=10000.))]
def run(j):
 name,params=j
 p=subprocess.run([sys.executable,'parameter_tests/run_new_trial.py',name,'fourier','0','30000'],env=dict(os.environ,TRIAL_PARAMS=json.dumps(params)),capture_output=True,text=True)
 print(p.stdout,p.stderr,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(run,jobs))
