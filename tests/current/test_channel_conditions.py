import pytest

from Infrastructure.conditions import ChannelCondition, ChannelSetup


def test_train_and_test_conditions_are_independent():
    setup = ChannelSetup(train=ChannelCondition('awgn', 10),
                         test=ChannelCondition('rayleigh_perfect_csi', 5))
    assert setup.for_stage('train').channel == 'awgn'
    assert setup.for_stage('test').channel == 'rayleigh_perfect_csi'
    assert setup.for_stage('test').snr_db == 5
    assert setup.for_stage('train').build().name == 'awgn'
    assert setup.for_stage('test').build().name == 'rayleigh_perfect_csi'


def test_unknown_channel_stage_and_nonfinite_snr_fail():
    with pytest.raises(ValueError):
        ChannelCondition('rician', 10)
    with pytest.raises(ValueError):
        ChannelCondition('awgn', float('nan'))
    with pytest.raises(ValueError):
        ChannelSetup().for_stage('unknown')


def test_deepjscc_selects_train_and_test_channels(monkeypatch):
    torch=pytest.importorskip('torch')
    from types import SimpleNamespace
    from Infrastructure.Channels.awgn import AWGNChannel
    from Infrastructure.Channels.rayleigh import RayleighChannel
    from src.models import build
    from src.sources import source_path
    if not (source_path('deepjscc')/'source_receipt.json').is_file():
        pytest.skip('Pinned DeepJSCC source not present')
    calls=[]
    def record(self,latent,snr_db,**kwargs):
        calls.append((self.name,snr_db))
        return SimpleNamespace(received=latent.tensor)
    monkeypatch.setattr(AWGNChannel,'transmit',record)
    monkeypatch.setattr(RayleighChannel,'transmit',record)
    model=build('deepjscc',{'from_scratch':True,'latent_channels':4,
        'train_channel':'awgn','train_snr_db':10,
        'test_channel':'rayleigh_perfect_csi','test_snr_db':5})
    image=torch.rand(1,3,32,32)
    model.train();model(image)
    model.eval();model(image)
    assert calls==[('awgn',10.0),('rayleigh_perfect_csi',5.0)]


def test_other_adapters_do_not_silently_change_channel():
    pytest.importorskip('torch')
    from src.models import build
    with pytest.raises(ValueError,match='does not yet support'):
        build('jcm',{'test_channel':'rayleigh_perfect_csi'})


def test_awgn_noise_power_and_channel_use_accounting():
    torch=pytest.importorskip('torch')
    from Infrastructure.Artifacts import LatentArtifact
    torch.manual_seed(23)
    latent=torch.ones(1,200000)
    result=ChannelCondition('awgn',10).build().transmit(
        LatentArtifact(latent,None,'fixture'),10)
    # Each real component has unit signal power; at Es/N0=10 dB its
    # noise variance is 0.1 after the original latent scale is restored.
    assert (result.received-latent).square().mean().item()==pytest.approx(.1,rel=.03)
    assert result.channel_uses_per_image==[100000]


@pytest.mark.parametrize('coherence',['symbol','image'])
def test_rayleigh_matches_explicit_complex_reference(coherence):
    torch=pytest.importorskip('torch')
    from Infrastructure.Artifacts import LatentArtifact
    from Infrastructure.Channels.channel_utils import real_to_complex_per_image,normalize_complex_per_image,complex_to_real_per_image
    image=torch.arange(1,17,dtype=torch.float32).reshape(2,8)
    symbols,shape=real_to_complex_per_image(image)
    normalized,scale=normalize_complex_per_image(symbols)
    torch.manual_seed(712)
    fading_shape=normalized.real if coherence=='symbol' else normalized.real[:,:1]
    h=torch.complex(torch.randn_like(fading_shape),torch.randn_like(fading_shape))/2**.5
    noise=torch.complex(torch.randn_like(normalized.real),torch.randn_like(normalized.imag))*(.1/2)**.5
    expected=complex_to_real_per_image(((h*normalized+noise)/h)*scale,shape)
    torch.manual_seed(712)
    actual=ChannelCondition('rayleigh_perfect_csi',10,coherence).build().transmit(LatentArtifact(image,None,'test'),10)
    torch.testing.assert_close(actual.received,expected)
    assert actual.metadata['coherence']==coherence


def test_masked_channel_power_uses_active_symbols_and_rejects_unpaired_masks():
    torch=pytest.importorskip('torch')
    from Infrastructure.Artifacts import LatentArtifact
    values=torch.ones(1,200000)
    mask=torch.zeros_like(values);mask[:,:50000]=1;mask[:,100000:150000]=1
    torch.manual_seed(992)
    channel=ChannelCondition().build()
    result=channel.transmit(LatentArtifact(values,mask,'fixture'),10)
    active=(result.received-values)[mask.bool()]
    assert active.square().mean().item()==pytest.approx(.1,rel=.03)
    assert result.channel_uses_per_image==[50000]
    assert torch.all(result.received[~mask.bool()]==0)
    mask[0,0]=0
    with pytest.raises(ValueError,match='matching'):
        channel.transmit(LatentArtifact(values,mask,'fixture'),10)


def test_direct_channel_empty_and_nonfinite_snr_rejected():
    torch=pytest.importorskip('torch')
    from Infrastructure.Artifacts import LatentArtifact
    channel=ChannelCondition().build()
    with pytest.raises(ValueError,match='nonempty'):
        channel.transmit(LatentArtifact(torch.empty(0,8),None,'test'),10)
    with pytest.raises(ValueError,match='finite'):
        channel.transmit(LatentArtifact(torch.ones(1,8),None,'test'),float('nan'))


def test_notebook_environment_profiles_and_native_restrictions():
    import json
    from pathlib import Path
    from src.notebook_support import environment_options
    environment=json.loads((Path(__file__).parents[2]/'configs/environment.json').read_text())
    options=environment_options('adjscc','manifest',environment,'awgn_to_rayleigh')
    assert options['train_channel']=='awgn' and options['test_channel']=='rayleigh_perfect_csi'
    options=environment_options('deepjscc','manifest',environment,overrides={'snr_db':7})
    assert options['train_snr_db']==options['test_snr_db']==7
    with pytest.raises(ValueError,match='does not yet support'):
        environment_options('mdjcm','manifest',environment,'rayleigh_symbol')
    assert environment_options('swinjscc','manifest',environment,'rayleigh_block')['channel_backend']=='infrastructure'


def test_adjscc_stage_conditions_preserve_normalized_decoder_input(monkeypatch):
    torch=pytest.importorskip('torch')
    from types import SimpleNamespace
    from src.models import build
    from Infrastructure.Channels.awgn import AWGNChannel
    from Infrastructure.Channels.rayleigh import RayleighChannel
    from src.sources import source_path
    if not (source_path('diffcom')/'source_receipt.json').is_file():pytest.skip('Pinned ADJSCC source not present')
    calls=[]
    def no_noise(self,latent,snr_db,**options):
        calls.append((self.name,snr_db,options['restore_scale']))
        assert (2*latent.tensor.square().mean(1)).tolist()==pytest.approx([1.0],rel=1e-4)
        return SimpleNamespace(received=latent.tensor)
    monkeypatch.setattr(AWGNChannel,'transmit',no_noise)
    monkeypatch.setattr(RayleighChannel,'transmit',no_noise)
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        model=build('adjscc',{'from_scratch':True,'latent_channels':2,'train_snr_db':10,
            'test_snr_db':5,'test_channel':'rayleigh_perfect_csi'})
        image=torch.rand(1,3,32,32)
        model.train();assert model(image)['reconstruction'].shape==image.shape
        model.eval();assert model(image)['reconstruction'].shape==image.shape
        assert calls==[('awgn',10,False),('rayleigh_perfect_csi',5,False)]
    finally:torch.set_num_threads(previous)
