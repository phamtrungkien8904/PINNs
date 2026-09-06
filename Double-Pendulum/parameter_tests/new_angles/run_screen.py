import subprocess,sys,os,json,concurrent.futures
jobs=[('previous_winner',{},[10000,30000,60000]),('current_script',dict(WARMUP_EPOCHS=2000,PHYSICS_RAMP_EPOCHS=8000,LEARNING_RATE_NETWORK=1e-4,WEIGHT_DECAY=1e-7),[40000,70000,90000]),('cutoff30',dict(MAX_ANGULAR_FREQUENCY=30.),[10000,30000,60000]),('period4_cutoff30',dict(FOURIER_PERIOD_FACTOR=4,MAX_ANGULAR_FREQUENCY=30.),[10000,30000,60000])]
def run(j):
 name,params,miles=j
 p=subprocess.run([sys.executable,'parameter_tests/run_new_trial.py',name,'fourier','0','30000'],env=dict(os.environ,TRIAL_PARAMS=json.dumps(params),TRIAL_MILESTONES=json.dumps(miles)),capture_output=True,text=True)
 print(p.stdout,p.stderr,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(run,jobs))
