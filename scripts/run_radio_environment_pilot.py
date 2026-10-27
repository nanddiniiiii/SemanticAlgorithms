"""Bounded Sionna PHY and ns-3 installation/correctness checks on Kaggle."""
import argparse
import importlib.metadata
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();checks={}
    def record(name,value):
        checks[name]=value
        (args.output/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
        print(json.dumps({'stage':name,**value}),flush=True)
    try:
        import torch
        from sionna.phy.channel import AWGN
        from sionna.phy.mapping import Mapper,Demapper
        from sionna.phy.fec.ldpc import LDPC5GEncoder,LDPC5GDecoder
        if not torch.cuda.is_available():raise RuntimeError('GPU pilot requires actual CUDA')
        record('environment',{'python':sys.version,'interpreter':sys.executable,
            'torch':torch.__version__,'cuda':torch.version.cuda,
            'sionna_no_rt':importlib.metadata.version('sionna-no-rt'),
            'gpus':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
        torch.manual_seed(22)
        for device in [f'cuda:{i}' for i in range(min(2,torch.cuda.device_count()))]:
            x=torch.ones(262144,dtype=torch.complex64,device=device)
            y=AWGN(device=device)(x,0.1);noise=y-x
            power=float(noise.abs().square().mean())
            snr=10*math.log10(1/power)
            assert abs(snr-10)<0.15
            v=torch.randn(32,64,device=device,requires_grad=True)
            z=torch.complex(v,torch.zeros_like(v));loss=AWGN(device=device)(z,0.1).abs().square().mean()
            loss.backward();assert torch.isfinite(v.grad).all() and v.grad.abs().sum()>0
            record('awgn_'+device.replace(':','_'),{'status':'passed','complex_noise_power':power,
                'measured_esn0_db':snr,'target_esn0_db':10,'finite_nonzero_gradient':True})
        device='cuda:0';mapper=Mapper('qam',2,device=device)
        demapper=Demapper('app','qam',2,device=device)
        bits=torch.randint(0,2,(128,1024),device=device).float()
        symbols=mapper(bits);clean=demapper(symbols,1e-4)>0
        assert torch.equal(clean,bits.bool())
        measured=[]
        for esn0 in (0,5,10):
            no=10**(-esn0/10)
            llr=demapper(AWGN(device=device)(symbols,no),no)
            ber=float(((llr>0)!=bits.bool()).float().mean())
            theory=0.5*math.erfc(math.sqrt(10**(esn0/10)/2))
            assert abs(ber-theory)<max(0.005,0.12*theory)
            measured.append({'esn0_db':esn0,'ber':ber,'qpsk_awgn_theory':theory})
        record('qpsk',{'status':'passed','noiseless_exact':True,'measurements':measured})
        encoder=LDPC5GEncoder(256,512,device=device)
        decoder=LDPC5GDecoder(encoder,hard_out=True,num_iter=10,device=device)
        payload=torch.randint(0,2,(16,256),device=device).float()
        coded=encoder(payload);tx=mapper(coded)
        recovered=decoder(demapper(tx,1e-4))
        assert torch.equal(recovered,payload)
        record('ldpc_noiseless',{'status':'passed','payload_bits':256,'coded_bits':512,
            'qpsk_data_symbols':256,'blocks':16,'exact_blocks':16,
            'scope':'noiseless digital chain; no pilots/headers/OFDM accounted'})
    except Exception as error:
        record('sionna_failure',{'status':'failed','error':str(error),'traceback':traceback.format_exc()})
    # Independent ns-3 source/build smoke: success is not coupled ray tracing.
    try:
        source=args.output/'ns3_source'
        commands=[['git','clone','--depth','1','--branch','ns-3.47',
            'https://github.com/nsnam/ns-3-dev-git.git',str(source)],
            [sys.executable,'ns3','configure','--enable-examples',
             '--enable-modules=core,network,internet,point-to-point,applications'],
            [sys.executable,'ns3','build','first','-j','2'],
            [sys.executable,'ns3','run','first','--no-build']]
        # ns-3 runner checks USER, while Kaggle containers often run as root.
        import os
        environment=os.environ.copy();environment['USER']='kaggle'
        for index,command in enumerate(commands):
            log=args.output/f'ns3_{index}.log'
            print(json.dumps({'stage':'ns3','step':index,'command':command}),flush=True)
            with log.open('w') as stream:
                subprocess.run(command,cwd=None if index==0 else source,
                    env=environment,stdout=stream,stderr=subprocess.STDOUT,
                    timeout=180 if index==0 else 600,check=True)
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
        assert revision=='e2c9e30c6ebdfd534aa7e30f6324b5674d138b9f'
        record('ns3',{'status':'passed','version':'3.47','commit':revision,
            'scope':'native first UDP example; no Sionna-RT coupling certified'})
    except Exception as error:
        record('ns3_failure',{'status':'failed','error':str(error),'traceback':traceback.format_exc()})
    (args.output/'status.json').write_text(json.dumps({'status':'completed' if
        not any('failure' in key for key in checks) else 'partial_or_failed',
        'elapsed_seconds':time.monotonic()-started,
        'ns3sionna_coupled':False,'codec_training_started':False},indent=2)+'\n')


if __name__=='__main__':main()
