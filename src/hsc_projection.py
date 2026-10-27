"""HSC paper projection equations; float32 RGB extension, not author codec release.

The transmitter uses a local reconstruction, never the noisy receiver output.
Both A and AX must travel to the receiver; their bytes are charged explicitly.
"""
import io
import numpy as np


def encode_projection(source,local_reconstruction,rank):
    x=np.asarray(source,dtype=np.float64)
    local=np.asarray(local_reconstruction,dtype=np.float64)
    if x.ndim!=2 or x.shape!=local.shape or not np.isfinite(x).all() or not np.isfinite(local).all():
        raise ValueError('Need finite equally shaped 2D source/local matrices')
    if not isinstance(rank,int) or not 0<=rank<=x.shape[0]:
        raise ValueError('rank outside matrix height')
    error=x-local
    _,vectors=np.linalg.eigh(error@error.T)
    basis=vectors[:,-rank:].T if rank else np.empty((0,x.shape[0]))
    projection=basis@x
    buffer=io.BytesIO()
    np.savez(buffer,basis=basis.astype(np.float32),projection=projection.astype(np.float32),
             shape=np.array(x.shape,dtype=np.int64))
    payload=buffer.getvalue()
    return payload,{'bytes':len(payload),'bits':8*len(payload),'rank':rank,
        'basis_values':basis.size,'projection_values':projection.size,
        'coefficient_precision':'float32','transport':'npz container; no FEC/modem',
        'receiver_realized_noise_used_by_encoder':False}


def decode_projection(payload,receiver_reconstruction):
    receiver=np.asarray(receiver_reconstruction,dtype=np.float64)
    with np.load(io.BytesIO(payload),allow_pickle=False) as record:
        basis=record['basis'].astype(np.float64)
        projection=record['projection'].astype(np.float64)
        shape=tuple(int(v) for v in record['shape'])
    if receiver.shape!=shape or basis.shape[1]!=shape[0] or projection.shape!=(basis.shape[0],shape[1]):
        raise ValueError('Projection packet and receiver dimensions disagree')
    if not all(np.isfinite(v).all() for v in (receiver,basis,projection)):
        raise ValueError('Nonfinite projection input')
    inverse=np.linalg.pinv(basis)
    return inverse@projection+(np.eye(shape[0])-inverse@basis)@receiver
