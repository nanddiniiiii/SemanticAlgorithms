"""Submit the frozen radio pilot once after the B codec job becomes terminal."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'Results/stage1_operations/radio_pilot_queue'
KERNEL='satyapaladuguplus/custom-mixture-native-swin'


def main():
    FOLDER.mkdir(parents=True,exist_ok=True)
    claim=FOLDER/'queue.claim'
    with claim.open('x') as stream:stream.write(datetime.now(timezone.utc).isoformat())
    kernel=ROOT/'Kaggle/radio_environment_pilot/kernel.py'
    metadata=kernel.parent/'kernel-metadata.json'
    digest=hashlib.sha256(kernel.read_bytes()).hexdigest()
    meta_digest=hashlib.sha256(metadata.read_bytes()).hexdigest()
    started=time.monotonic();last=None
    def call(*args,timeout=120):
        return subprocess.run([sys.executable,str(ROOT/'scripts/kaggle_account.py'),'B',*args],
            cwd=ROOT,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout)
    while time.monotonic()-started<4*3600:
        try:
            result=call('kernels','status',KERNEL)
            text=result.stdout+result.stderr
            state=next((s for s in ('COMPLETE','ERROR','CANCELLED','RUNNING','QUEUED')
                        if 'KernelWorkerStatus.'+s in text),'UNKNOWN')
            if state!=last:
                event={'timestamp':datetime.now(timezone.utc).isoformat(),'upstream':KERNEL,
                    'state':state,'kernel_sha256':digest}
                with (FOLDER/'events.jsonl').open('a') as stream:stream.write(json.dumps(event)+'\n')
                last=state
            if result.returncode==0 and state in ('COMPLETE','ERROR','CANCELLED'):
                if hashlib.sha256(kernel.read_bytes()).hexdigest()!=digest or hashlib.sha256(metadata.read_bytes()).hexdigest()!=meta_digest:
                    raise RuntimeError('Frozen radio candidate changed; refusing submission')
                # Claim submission before network request. An ambiguous failure is never retried.
                with (FOLDER/'submission.claim').open('x') as stream:stream.write(digest)
                submitted=call('kernels','push','-p',
                    '/mnt/d/Bunker/BaseCamp/WirelessCommunication/SemanticCommunication/SemanticLossless/Semantic_Lossless/Kaggle/radio_environment_pilot',
                    '--accelerator','NvidiaTeslaT4','--timeout','3600',timeout=180)
                (FOLDER/'submission.log').write_text(submitted.stdout+submitted.stderr,encoding='utf-8')
                (FOLDER/'status.json').write_text(json.dumps({'status':'submitted' if submitted.returncode==0 else 'submission_failed_no_retry',
                    'returncode':submitted.returncode,'kernel_sha256':digest,'upstream_terminal':state},indent=2))
                return
        except Exception as error:
            with (FOLDER/'errors.jsonl').open('a') as stream:
                stream.write(json.dumps({'timestamp':datetime.now(timezone.utc).isoformat(),'error':str(error)})+'\n')
            if (FOLDER/'submission.claim').exists():return
        time.sleep(60)
    (FOLDER/'status.json').write_text(json.dumps({'status':'queue_timeout_no_submission'}))


if __name__=='__main__':main()
