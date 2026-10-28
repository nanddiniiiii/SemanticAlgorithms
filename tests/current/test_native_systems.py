import math
import pytest


def test_legacy_fft_matches_explicit_dft_and_inverse_with_gradient():
    import torch
    from src.native_runtime import legacy_fft,legacy_ifft
    value=torch.randn(2,4,2,dtype=torch.float64,requires_grad=True)
    k=torch.arange(4,dtype=torch.float64)
    matrix=torch.exp(-2j*math.pi*k[:,None]*k[None,:]/4)
    result=legacy_fft(value)
    expected=torch.view_as_complex(value.contiguous())@matrix.T
    torch.testing.assert_close(torch.view_as_complex(result),expected,atol=1e-12,rtol=1e-12)
    torch.testing.assert_close(legacy_ifft(result),value,atol=1e-12,rtol=1e-12)
    result.square().sum().backward()
    assert torch.isfinite(value.grad).all()


@pytest.mark.parametrize('name,crop',[('deepjscc_mimo',32),('ofdm_guided_jscc',256),('ssc',256)])
def test_author_system_cpu_optimizer_and_strict_checkpoint(name,crop,tmp_path):
    import torch
    from src.models import build,load_training_state
    from src.sources import source_path
    if not (source_path(name)/'source_receipt.json').is_file():pytest.skip('Pinned author source not cached')
    previous=torch.get_num_threads();torch.set_num_threads(2)
    try:
        options={'algorithm':name,'crop':crop,'device':'cpu','data_parallel':False,'from_scratch':True}
        model=build(name,options,'cpu').train();image=torch.rand(1,3,crop,crop)
        optimizer=model.optimizer();optimizer.zero_grad()
        output=model(image);loss=model.loss(output,{'image':image})
        assert output['reconstruction'].shape==image.shape and torch.isfinite(loss)
        loss.backward();optimizer.step()
        checkpoint=tmp_path/'last.pt'
        torch.save({'state_dict':model.state_dict(),'optimizer':optimizer.state_dict()},checkpoint)
        state=torch.load(checkpoint,map_location='cpu',weights_only=True)
        load_training_state(model,state['state_dict'])
        if name=='ssc':assert output['source_bits']>20
    finally:torch.set_num_threads(previous)


def test_shared_codebook_parameter_ema_survives_alias_reload():
    import torch
    from src.ssc_training import update_parameter_ema
    embedding=torch.nn.Embedding(4,2)
    model=torch.nn.ModuleList([embedding]*5)
    initial=embedding.weight.detach().clone()
    state={key:value.clone() for key,value in model.state_dict().items()}
    with torch.no_grad():embedding.weight.add_(1)
    update_parameter_ema(model,state,decay=.9)
    model.load_state_dict(state,strict=True)
    torch.testing.assert_close(embedding.weight,initial+.1)
