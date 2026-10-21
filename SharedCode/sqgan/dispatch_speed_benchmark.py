"""Wait for a free Kaggle batch GPU slot; never stop another session."""
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'Results/SQGAN/speed_benchmark'; OUT.mkdir(parents=True,exist_ok=True)
FOLDER=ROOT/'Notebooks/project_notebooks/Training/SQGAN_speed_benchmark'
KERNEL='hmmkk20/sqgan-speed-benchmark-20261020'

def run(*args):
    return subprocess.run([sys.executable,'-m','kaggle',*args],capture_output=True,text=True,
                          encoding='utf-8',errors='replace',timeout=90)

def record(status,**extra):
    data={'status':status,'kernel':KERNEL,'updated_utc':datetime.now(timezone.utc).isoformat(),
          'production_run_modified':False,**extra}
    # Direct write tolerates transient OneDrive replace locks via retries.
    for attempt in range(5):
        try:
            (OUT/'dispatch_status.json').write_text(json.dumps(data,indent=2)); break
        except PermissionError:
            time.sleep(1)
    print(json.dumps(data),flush=True)

def main():
    deadline=time.time()+24*3600
    submitted=False
    record('WAITING_FOR_GPU_SLOT')
    while time.time()<deadline:
        if (OUT/'STOP').exists():
            record('MONITOR_STOPPED'); return
        try:
            if not submitted:
                # Avoid submitting while both known GPU workloads are active.
                states=[run('kernels','status',k).stdout for k in (
                    'hmmkk20/sqgan-div2k-150ep-continuation-02',
                    'hmmkk20/ntscc-plus-full-div2k-150ep-20261020')]
                if all('RUNNING' in s or 'QUEUED' in s for s in states):
                    time.sleep(60); continue
                result=run('kernels','push','-p',str(FOLDER),'--accelerator','NvidiaTeslaT4')
                output=result.stdout+'\n'+result.stderr
                if 'successfully pushed' in output:
                    submitted=True; record('SUBMITTED')
                elif 'Maximum batch GPU session count' in output:
                    record('WAITING_FOR_GPU_SLOT')
                else:
                    record('SUBMISSION_FAILED',detail=output[-1800:]); return
            else:
                result=run('kernels','status',KERNEL)
                if 'COMPLETE' in result.stdout:
                    download=run('kernels','output',KERNEL,'-p',str(OUT/'gpu_output'),
                                 '--file-pattern',r'(benchmark_report|rank_[0-9]+).json$')
                    if download.returncode:
                        record('DOWNLOAD_FAILED',detail=download.stderr[-1000:]); return
                    reports=list((OUT/'gpu_output').rglob('benchmark_report.json'))
                    if not reports:
                        record('REPORT_MISSING'); return
                    report=json.loads(reports[0].read_text())
                    assert report['source_unchanged'] and not report['production_settings_changed']
                    record('MEASUREMENTS_READY_FOR_REVIEW',report=str(reports[0])); return
                if 'ERROR' in result.stdout or 'CANCEL' in result.stdout:
                    record('BENCHMARK_FAILED',detail=result.stdout); return
        except Exception as exc:
            record('CHECK_RETRY',detail=repr(exc))
        time.sleep(60)
    record('WAIT_EXPIRED_NO_PRODUCTION_CHANGE')

if __name__=='__main__':
    main()
