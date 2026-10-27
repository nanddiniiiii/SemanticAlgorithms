"""Measure active-symbol SNR in the retained port without changing its code."""
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import torch
from Infrastructure.Artifacts import LatentArtifact
from Infrastructure.Channels.awgn import AWGNChannel
from src.artifacts import write_json


def main():
    torch.set_num_threads(2)
    measurements=[]
    for rate in [32,64,96,128,192]:
        mask=torch.zeros(1,4096,192);mask[:,:,:rate]=1
        latent=LatentArtifact(tensor=torch.ones_like(mask),rate_mask=mask,source_model='synthetic_power_probe')
        torch.manual_seed(42)
        result=AWGNChannel().transmit(latent,10)
        signal=latent.tensor*mask
        active_noise=(result.received-signal)[mask.bool()]
        snr=10*math.log10(1/float(active_noise.square().mean()))
        expected=10+10*math.log10(192/rate)
        assert abs(snr-expected)<.1
        measurements.append({'feature_rate':rate,'nominal_snr_db':10,
            'measured_active_coordinate_snr_db':snr,'expected_dense_padding_gain_db':10*math.log10(192/rate)})
    report={'scope':'synthetic audit of retained port, not native author model; no source modified',
        'finding':'normalization averages inactive zeros while channel-use count omits them',
        'measurements':measurements}
    write_json(ROOT/'Results/implementation_checks/swin_active_power_audit.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
