"""Serial person1 chunks using the existing stable notebook and receipts."""
import argparse
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import time
from .protocol import ORDER
from src.runtime import write_json
from src.persistence import file_lock


def main():
    p=argparse.ArgumentParser();p.add_argument('--username',required=True);p.add_argument('--interval',type=int,default=1800)
    p.add_argument('--training-seconds',type=int,default=14400);a=p.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+',a.username):raise ValueError('Invalid Kaggle username')
    a.username=a.username.lower()
    root=Path(__file__).resolve().parents[2];folder=root/'Results/person1_campaign'/a.username;folder.mkdir(parents=True,exist_ok=True)
    base=root/'Results/team_downloads/1'/(a.username+'__semantic-notebook-1')
    state_path=folder/'progress.json'
    state=json.loads(state_path.read_text()) if state_path.is_file() else dict(username=a.username,models={n:dict(status='pending') for n in ORDER},handled=[])
    if state.get('username')!=a.username:raise ValueError('Campaign state belongs to another Kaggle account')
    from src.launch import notebook_revision
    reviewed=notebook_revision(json.loads((root/'Notebooks/1.ipynb').read_text()))
    if state.get('reviewed_source') not in (None,reviewed) and any(
        entry['status'] in ('partial','training_completed') for entry in state['models'].values()):
        raise ValueError('Reviewed source changed after training progress; inspect contracts before continuation')
    state['reviewed_source']=reviewed
    for entry in state['models'].values():entry['pilot_passed']=entry.get('pilot_source')==reviewed
    with file_lock(folder/'campaign.lock') as owned:
        if not owned:raise RuntimeError('A person1 campaign is already running')
        while True:
            current=notebook_revision(json.loads((root/'Notebooks/1.ipynb').read_text()))
            if current!=reviewed:
                raise RuntimeError('Notebook source changed during campaign; stop and revalidate before submitting')
            receipts=sorted(base.glob('*/submission.json'),key=lambda p:p.stat().st_mtime)
            active=[]
            for receipt in receipts:
                destination=receipt.parent
                collected=destination/'collector_state.json'
                if not collected.is_file() or json.loads(collected.read_text()).get('status')!='retrieved':
                    active.append(destination);continue
                if destination.name in state['handled']:continue
                submission=json.loads(receipt.read_text());name=submission.get('person1_model')
                if name not in ORDER:continue
                summaries=[path for path in (destination/'published_output').rglob('training_summary.json') if path.parent.name==name]
                if submission.get('person1_mode')=='pilot':
                    passed=list((destination/'published_output').rglob('pilot_passed.json'))
                    if not passed:raise RuntimeError('Person1 pilot failed; inspect retrieved evidence before retry')
                    state['models'][name]['pilot_source']=submission['source_revision']
                    state['models'][name]['pilot_passed']=submission['source_revision']==reviewed
                elif summaries:
                    summary=json.loads(summaries[0].read_text());entry=state['models'][name]
                    if summary['status']=='training_completed':
                        packaged=summaries[0].parent/'package_result.json'
                        if not packaged.is_file() or not json.loads(packaged.read_text()).get('produced'):
                            raise RuntimeError('Training finished but evaluation/package failed; inspect '+str(destination))
                    entry.update(status=summary['status'],completed_epochs=summary['completed_epochs'],receipt=str(destination))
                    if summary['status']=='partial':
                        checkpoint=destination/('checkpoint_dataset-'+name+'.json')
                        if not checkpoint.is_file():raise RuntimeError('Partial chunk has no verified continuation input')
                        source=json.loads(checkpoint.read_text())
                        if source.get('status')!='ready':raise RuntimeError('Private continuation input not ready')
                        entry['dataset']=source['dataset']
                else:
                    raise RuntimeError('Training failed before valid progress; inspect '+str(destination))
                state['handled'].append(destination.name);write_json(state_path,state)
            if active:
                command=[sys.executable,str(root/'scripts/1.py'),'--username',a.username,'--resume',str(active[-1]),'--interval',str(a.interval),'--max-hours','24']
            else:
                selected=next((n for n in ORDER if not state['models'][n].get('pilot_passed')),None)
                mode='pilot'
                if selected is None:
                    selected=next((n for n in ORDER if state['models'][n]['status']!='training_completed'),None)
                    mode='train'
                if selected is None:
                    state['status']='training_finished';write_json(state_path,state)
                    from .handoff import create
                    create(root);return
                command=[sys.executable,str(root/'scripts/1.py'),'--username',a.username,'--model',selected,'--mode',mode,
                         '--training-seconds',str(a.training_seconds),'--interval',str(a.interval),'--max-hours','24','--create-missing']
                if receipts:command.append('--new-run')
                if mode=='train' and state['models'][selected].get('dataset'):command+=['--continue-from',state['models'][selected]['dataset']]
            state.update(status='running',command=command,updated_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
            write_json(state_path,state);print('Campaign command:',command,flush=True)
            result=subprocess.run(command,cwd=root)
            if result.returncode:
                state.update(status='needs_investigation',returncode=result.returncode)
                write_json(state_path,state);raise RuntimeError('Collector/submission failed; campaign stopped with receipts preserved')


if __name__=='__main__':main()
