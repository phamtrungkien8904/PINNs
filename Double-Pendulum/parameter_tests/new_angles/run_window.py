import subprocess,sys,os,json,concurrent.futures
jobs=[('window10s',dict(DATA_STOP=1000,DATA_STEP=34,MAX_ANGULAR_FREQUENCY=30.)),('window10s_energy',dict(DATA_STOP=1000,DATA_STEP=34,MAX_ANGULAR_FREQUENCY=30.,LAMBDA_ENERGY=1.))]
def run(j):
 name,params=j
 p=subprocess.run([sys.executable,'parameter_tests/run_new_trial.py',name,'fourier','0','30000'],env=dict(os.environ,TRIAL_PARAMS=json.dumps(params)),capture_output=True,text=True)
 print(p.stdout,p.stderr,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(run,jobs))
