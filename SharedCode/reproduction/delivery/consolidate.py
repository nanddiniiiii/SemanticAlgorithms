"""Collect complete measured grids with explicit evidence limitations; never rank."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path


def consolidate(paths, output, contract):
    contract_path=Path(contract)
    policy=json.loads(contract_path.read_text())
    contract_sha=hashlib.sha256(contract_path.read_bytes()).hexdigest()
    rows=[]; reports=[]
    expected=set(itertools.product([f'kodim{i:02d}' for i in range(1,25)],policy['evaluation']['snr_db'],range(10)))
    for source in paths:
        path=Path(source)
        receipt=json.loads((path/'measurement_receipt.json').read_text())
        records=[json.loads(line) for line in (path/'measured_results.jsonl').read_text().splitlines() if line.strip()]
        identity=receipt.get('identity',{})
        blockers=[]
        if receipt.get('status')!='complete' or receipt.get('evaluation_complete') is not True: blockers.append('evaluation not complete')
        epochs=receipt.get('complete_training_epochs',0)
        if not 50<=epochs<=150: blockers.append('completed training epoch count outside contract')
        if receipt.get('validation_pass') is not True: blockers.append('validation has not passed')
        if not receipt.get('checkpoint_sha256') or len(receipt['checkpoint_sha256'])!=64: blockers.append('checkpoint checksum absent')
        if not receipt.get('stream_ledger') or receipt.get('all_transmitted_streams_counted') is not True: blockers.append('stream ledger missing')
        if identity.get('contract_sha256')!=contract_sha: blockers.append('contract hash differs')
        grid={(r.get('image'),r.get('snr_db'),r.get('trial')) for r in records}
        if grid!=expected or len(records)!=len(expected): blockers.append('missing or duplicate 24x5x10 measurements')
        for r in records:
            numeric=['SSIM','MS_SSIM','LPIPS','MSE','complex_channel_uses','CBR','encoder_seconds','decoder_seconds','channel_seconds','end_to_end_seconds']
            if r.get('measured') is not True or any(type(r.get(k)) not in (int,float) or not math.isfinite(r[k]) for k in numeric):
                blockers.append('invalid measured metrics'); break
            if (type(r['complex_channel_uses']) is not int or r['complex_channel_uses']<=0
                    or r['CBR']<=0 or not 0<=r['MSE']<=1
                    or not 0<=r['SSIM']<=1 or not 0<=r['MS_SSIM']<=1 or r['LPIPS']<0
                    or any(r[k]<0 for k in ['encoder_seconds','decoder_seconds','channel_seconds','end_to_end_seconds'])):
                blockers.append('measured metric outside physical range'); break
            if r.get('native_full_image') is not True or r.get('all_transmitted_streams_counted') is not True:
                blockers.append('geometry or stream accounting incomplete'); break
            if r.get('source_commit')!=identity.get('source_commit') or r.get('model')!=receipt.get('model') or r.get('channel')!=receipt.get('channel'):
                blockers.append('measurement identity differs'); break
            if r.get('PSNR') is None and r['MSE']!=0: blockers.append('PSNR absent for nonzero MSE'); break
            if r.get('PSNR') is not None and (type(r['PSNR']) not in (int,float) or not math.isfinite(r['PSNR'])): blockers.append('invalid PSNR'); break
            if r['MSE']>0 and not math.isclose(r['PSNR'],-10*math.log10(r['MSE']),abs_tol=1e-4,rel_tol=1e-6):
                blockers.append('PSNR and MSE inconsistent'); break
            if r['MSE']==0 and r.get('PSNR') is not None:
                blockers.append('zero MSE must represent infinite PSNR as null'); break
            h,w=r.get('original_height'),r.get('original_width')
            if (h,w) not in {(512,768),(768,512)} or abs(r['CBR']-r['complex_channel_uses']/(3*h*w))>1e-12:
                blockers.append('realized channel budget inconsistent'); break
        limitations=['Checkpoint checksum and completed-training evidence require independent review',
                     'Model provenance and all-stream transport ledger require independent review',
                     'Partial forward FLOPs and absent training FLOPs cannot support total-cost ranking',
                     'Realized CBR values must be matched before any rate-quality ranking']
        reports.append({'source':str(path),'measured_grid_valid':not blockers,'blocking_errors':sorted(set(blockers)),
                        'ranking_eligible':False,'limitations':limitations,'runtime_receipt':receipt})
        if not blockers: rows.extend(records)
    output=Path(output)
    if output.exists(): raise FileExistsError('Preserve existing consolidated artifacts')
    output.mkdir(parents=True)
    (output/'evidence_report.json').write_text(json.dumps(reports,indent=2))
    if not rows: raise ValueError('No complete validated measured grids; no comparison produced')
    (output/'partial_measured_results_unranked.jsonl').write_text(''.join(json.dumps(r,allow_nan=False)+'\n' for r in rows))
    return len(rows)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inputs',nargs='+',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--contract',required=True)
    args=parser.parse_args()
    print(json.dumps({'event':'consolidated_unranked_partial_evidence','rows':consolidate(args.inputs,args.output,args.contract),'ranking_eligible':False}))
