import unittest
import numpy as np
from src.hsc_projection import encode_projection,decode_projection


class HSCProjectionTests(unittest.TestCase):
    def test_error_matches_discarded_eigenvalues_and_packet_counts_basis(self):
        rng=np.random.default_rng(22)
        x=rng.random((8,12));local=x+rng.normal(0,0.1,x.shape)
        for rank in (0,3,8):
            packet,info=encode_projection(x,local,rank)
            actual=decode_projection(packet,local)
            eigenvalues=np.linalg.eigvalsh((x-local)@(x-local).T)
            expected=sum(eigenvalues[:8-rank])
            self.assertAlmostEqual(float(np.square(x-actual).sum()),expected,places=6)
            self.assertEqual(info['bits'],8*len(packet))
            self.assertEqual(info['basis_values'],rank*8)

    def test_received_base_changes_output_without_encoder_oracle(self):
        x=np.eye(4);local=np.zeros_like(x)
        packet,_=encode_projection(x,local,1)
        first=decode_projection(packet,local)
        second=decode_projection(packet,np.ones_like(x))
        self.assertFalse(np.allclose(first,second))
