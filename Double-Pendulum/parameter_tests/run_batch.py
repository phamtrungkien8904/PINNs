import concurrent.futures,subprocess,sys,pathlib
root=pathlib.Path(__file__).resolve().parent
jobs=[('fourier_seed0','fourier',0,25000),('fourier_seed1','fourier',1,25000),('fourier_seed2','fourier',2,25000),('fourier_baseline','fourier',0,20000),('time_baseline','time',0,25000)]
def run(job):
 p=subprocess.run([sys.executable,str(root/'run_trial.py'),*map(str,job)],capture_output=True,text=True)
 print(p.stdout,p.stderr if p.returncode else '',flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(run,jobs))
