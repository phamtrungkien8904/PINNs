import subprocess,sys,os,json,concurrent.futures
jobs=[('early_physics',dict(WARMUP_EPOCHS=0,PHYSICS_RAMP_EPOCHS=1000)),('ridge_small',dict(INITIALIZATION_RIDGE=1e-6)),('window10s_cutoff12',dict(DATA_STOP=1000,DATA_STEP=34,MAX_ANGULAR_FREQUENCY=12.))]
def run(j):
 name,params=j
 p=subprocess.run([sys.executable,'parameter_tests/run_new_trial.py',name,'fourier','0','30000'],env=dict(os.environ,TRIAL_PARAMS=json.dumps(params)),capture_output=True,text=True)
 print(p.stdout,p.stderr,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(run,jobs))
