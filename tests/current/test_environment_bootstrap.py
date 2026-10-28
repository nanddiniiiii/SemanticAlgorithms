import subprocess
import sys
import pytest
from src import notebook_support


def test_bootstrap_avoids_ensurepip_and_targets_model_environment(tmp_path,monkeypatch):
    calls=[]
    def create(self,path):
        calls.append(('create',self.with_pip,self.system_site_packages))
        path.mkdir(parents=True)
        (path/'pyvenv.cfg').write_text('test')
    monkeypatch.setattr(notebook_support.venv.EnvBuilder,'create',create)
    commands=[]
    python=notebook_support.setup_model(tmp_path,'ssc',tmp_path/'run',command_runner=lambda command,timeout:commands.append(command))
    assert calls==[('create',False,True)]
    assert commands[0][:5]==[sys.executable,'-m','pip','--python',str(python)]


def test_bootstrap_preserves_installer_stderr(tmp_path,monkeypatch):
    destination=tmp_path/'.venv-team/ssc';destination.mkdir(parents=True)
    (destination/'pyvenv.cfg').write_text('test')
    def fail(command,stdout,stderr,timeout):
        stdout.write('specific installer failure\n');stdout.flush()
        return subprocess.CompletedProcess(command,7)
    monkeypatch.setattr(notebook_support.subprocess,'run',fail)
    with pytest.raises(subprocess.CalledProcessError):
        notebook_support.setup_model(tmp_path,'ssc',tmp_path/'run')
    assert 'specific installer failure' in (tmp_path/'run/environment_setup.log').read_text()
