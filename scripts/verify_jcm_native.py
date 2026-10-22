"""Compare the adapter with an independent native instance, same weights/RNG.

Synthetic numerical equivalence only; no quality or training convergence claim.
Run with the project Torch environment after `prepare --fetch`.
"""
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import torch
from semantic_lossless.artifacts import write_json
from semantic_lossless.codecs import build_codec, upstream_import
from semantic_lossless.config import Config
from semantic_lossless.sources import source_path
import importlib


def main():
    torch.set_num_threads(2)
    config = Config('jcm', dataset='cifar10', crop_size=32,
                    model_options={'modulation': '16qam', 'tradeoff_lambda': 200})
    adapter = build_codec(config).eval()
    with upstream_import(source_path('jcm'), ('network', 'modules')):
        native = importlib.import_module('network').JCM(copy.copy(adapter.native), torch.device('cpu'))
    native.load_state_dict(adapter.core.state_dict(), strict=True)
    native.eval()
    torch.manual_seed(42)
    image = torch.rand(2, 3, 32, 32)
    labels = torch.tensor([2, 7])
    rng = torch.get_rng_state()
    expected = native(image * 2 - 1)
    torch.set_rng_state(rng)
    actual = adapter(image)
    torch.testing.assert_close(actual['native_reconstruction'], expected[4], rtol=0, atol=0)
    torch.testing.assert_close(actual['logits'], expected[3], rtol=0, atol=0)
    expected_loss = (torch.nn.functional.cross_entropy(expected[3], labels)
                     + 200 * torch.nn.functional.mse_loss(expected[4], image * 2 - 1))
    actual_loss = adapter.loss(actual, {'image': image, 'label': labels})
    torch.testing.assert_close(actual_loss, expected_loss, rtol=0, atol=0)
    expected_loss.backward()
    actual_loss.backward()
    for (name, parameter), (other_name, other) in zip(native.named_parameters(), adapter.core.named_parameters()):
        assert name == other_name
        if parameter.grad is None:
            assert other.grad is None
        else:
            torch.testing.assert_close(parameter.grad, other.grad, rtol=0, atol=0)
    shapes = {'input': list(image.shape), 'encoder': list(native.encoder(image * 2 - 1).shape),
              'code': list(expected[0].shape), 'normalized_channel_input': list(expected[1].shape),
              'received': list(expected[2].shape), 'class_logits': list(expected[3].shape),
              'reconstruction': list(expected[4].shape)}
    assert shapes['encoder'] == [2, 256, 4, 4]
    assert shapes['code'] == [2, 256]
    assert shapes['reconstruction'] == [2, 3, 32, 32]
    receipt = {'status': 'passed', 'scope': 'CPU synthetic native-forward, objective and gradient equivalence; not paper reproduction',
               'model': adapter.metadata, 'torch': torch.__version__, 'shapes': shapes,
               'forward_max_absolute_error': 0, 'loss_absolute_error': 0,
               'parameter_gradients': 'exact equality', 'rng': 'identical Torch state before each forward',
               'excluded': ['dataset preprocessing', 'training convergence', 'GPU equivalence', 'published metrics']}
    destination = ROOT / 'Results/implementation_checks/jcm_native_equivalence.json'
    write_json(destination, receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == '__main__':
    main()
