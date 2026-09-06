import subprocess,sys,os,json,concurrent.futures
jobs=[('energy1',dict(LAMBDA_ENERGY=1.,MAX_ANGULAR_FREQUENCY=30.)),('energy10',dict(LAMBDA_ENERGY=10.,MAX_ANGULAR_FREQUENCY=30.)),('spectrum_fast',dict(LEARNING_RATE_SPECTRUM=0.001,MAX_ANGULAR_FREQUENCY=30.))]
def run(j):
 name,params=j
 p=subprocess.run([sys.executable,'parameter_tests/run_new_trial.py',name,'fourier','0','30000'],env=dict(os.environ,TRIAL_PARAMS=json.dumps(params)),capture_output=True,text=True)
 print(p.stdout,p.stderr,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(run,jobs))
