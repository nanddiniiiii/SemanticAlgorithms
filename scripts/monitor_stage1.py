"""Watch submitted Stage1 jobs and retrieve results; never launch/retry training.

Uses isolated account identities through kaggle_account.py. This monitor records
queue/failure states honestly and can continue as a hidden local helper process.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
JOBS = [
    ('A', 'satyapaladugu/stage1-shared-codebook-classifier', 'shared_classifier'),
    ('B', 'satyapaladuguplus/stage1-native-swin-evaluation', 'swin_native'),
]


def call(account, *command, timeout=120):
    return subprocess.run([sys.executable, str(ROOT/'scripts/kaggle_account.py'), account, *command],
        cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=timeout)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--interval', type=int, default=60)
    parser.add_argument('--max-hours', type=float, default=4)
    parser.add_argument('--once', action='store_true')
    args = parser.parse_args()
    folder = ROOT/'Results/stage1_monitor'; folder.mkdir(parents=True, exist_ok=True)
    started = time.monotonic(); terminal = set()
    while time.monotonic()-started < args.max_hours*3600:
        states = []
        for account, kernel, name in JOBS:
            marker = folder/(name+'_downloaded.json')
            if marker.exists(): terminal.add(name); continue
            record = {'timestamp': datetime.now(timezone.utc).isoformat(), 'account': account, 'kernel': kernel}
            try:
                result = call(account, 'kernels', 'status', kernel)
                output = result.stdout+'\n'+result.stderr
                record.update(status_output=output.strip(), returncode=result.returncode)
                state = 'complete' if 'KernelWorkerStatus.COMPLETE' in output else 'error' if 'KernelWorkerStatus.ERROR' in output else None
                if result.returncode == 0 and state:
                    target = ROOT/'Results/stage1_downloads'/name
                    remote_target = '/mnt/'+target.drive[0].lower()+target.as_posix()[2:] if account == 'B' else str(target)
                    downloaded = call(account, 'kernels', 'output', kernel, '-p', remote_target,
                        '--file-pattern', '.*(json|jsonl|csv|pt|png|log)$', '--force', timeout=900)
                    record.update(download_returncode=downloaded.returncode,
                        download_output=(downloaded.stdout+'\n'+downloaded.stderr)[-6000:])
                    if downloaded.returncode == 0:
                        merged = subprocess.run([sys.executable, str(ROOT/'scripts/import_stage1_outputs.py')],
                            cwd=ROOT, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=180)
                        record.update(import_returncode=merged.returncode, import_output=(merged.stdout+'\n'+merged.stderr)[-3000:])
                        if merged.returncode != 0:
                            raise RuntimeError('Output download succeeded but canonical import failed; will retry')
                        marker.write_text(json.dumps({'kernel': kernel, 'terminal_status': state,
                            'download_directory': str(target), 'timestamp': record['timestamp']}, indent=2), encoding='utf-8')
                        terminal.add(name)
            except Exception as error:
                record['monitor_error'] = str(error)
            states.append(record)
            with (folder/'events.jsonl').open('a', encoding='utf-8') as stream: stream.write(json.dumps(record)+'\n')
        status = {'timestamp': datetime.now(timezone.utc).isoformat(), 'jobs': states,
            'downloaded_jobs': sorted(terminal), 'training_launched_by_monitor': False}
        temporary = folder/'status.json.tmp'; temporary.write_text(json.dumps(status, indent=2), encoding='utf-8')
        temporary.replace(folder/'status.json')
        print(json.dumps(status), flush=True)
        if args.once or len(terminal) == len(JOBS): return
        time.sleep(max(10, args.interval))


if __name__ == '__main__':
    main()
