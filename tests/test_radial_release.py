from dataclasses import asdict
import json

import numpy as np
import pytest

from llsa.secondary.global_radial import GlobalRadial, OutletSponge, BlendedSponge
from llsa.secondary.radial import RadialConfig, unpack_field


@pytest.mark.parametrize('fraction',[0.,.35,1.])
@pytest.mark.parametrize('damp_mean',[False,True])
def test_damping_release_is_exact_operator_homotopy(fraction,damp_mean):
    r = np.linspace(420.,470.,9)
    c = RadialConfig(n=8,harmonics=2)
    old = OutletSponge(425.,10.,10.,damp_mean)
    new = OutletSponge(445.,10.,10.,damp_mean)
    problems = [GlobalRadial(r,c,outlet_sponge=s) for s in [old,new,BlendedSponge(old,new,fraction)]]
    g = problems[0]
    x = np.random.default_rng(32).normal(size=len(r)*g.width)*.001
    q = unpack_field(x,len(r),g.H,g.s)
    pairs = [p.evaluate(x,q[0],jacobian=True) for p in problems]
    np.testing.assert_allclose(pairs[2][0],(1-fraction)*pairs[0][0]+fraction*pairs[1][0],atol=2e-14)
    delta = pairs[2][1]-(1-fraction)*pairs[0][1]-fraction*pairs[1][1]
    assert np.max(abs(delta.data),initial=0) < 1e-12
    np.testing.assert_allclose(problems[-1].sponge_strength[r<=old.start],0,atol=0)


def test_release_workflow_marks_auxiliary_fields_and_can_restart(tmp_path):
    from llsa.secondary.radial_release import main
    c = RadialConfig(n=8,harmonics=1)
    r = np.linspace(320/abs(c.ro),325/abs(c.ro),15)
    q = np.zeros((len(r),3,4*(c.n+1)),complex)
    seed = tmp_path/'laminar_seed.npz'
    np.savez(seed,radii=r,modes=q,config_json=json.dumps(asdict(c)))
    common = ['--old-start','321','--target','323','--fixed-end','325','--width','2',
              '--step','.1','--max-step','.2']
    first = tmp_path/'first'
    assert main(['--seed',str(seed),'--output',str(first),*common]) == 0
    summary = json.loads((first/'summary.json').read_text())
    assert summary['status']=='target_reached_discrete_equations'
    assert summary['stations'][-1]['release_fraction']==1
    assert summary['stations'][-1]['status']=='converged_discrete_equations'
    assert all(s['status']=='converged_auxiliary_damping' for s in summary['stations'][:-1])
    assert all(s['sponge_start']==321 for s in summary['stations'][:-1])
    # The accumulated 0.85 station may be stored as 0.8500000000000001;
    # users should be able to restart using the decimal shown in the folder.
    restart = summary['stations'][-2]
    source = first/restart['folder']/'fields.npz'
    assert main(['--seed',str(source),'--start-fraction',f'{restart["release_fraction"]:.6f}',
                 '--output',str(tmp_path/'second'),*common]) == 0
    bad = common.copy();bad[bad.index('323')]='322'
    with pytest.raises(SystemExit):
        main(['--seed',str(source),'--start-fraction',str(restart['release_fraction']),
              '--output',str(tmp_path/'bad'),*bad])
    previous = summary['stations'][1]
    trial = summary['stations'][2]
    trial_path = first/trial['folder']/'fields.npz'
    resumed = tmp_path/'resume_trial'
    assert main(['--seed',str(first/previous['folder']/'fields.npz'),
                 '--start-fraction',str(previous['release_fraction']),
                 '--trial-checkpoint',str(trial_path),'--output',str(resumed),*common,
                 '--step','.15','--phase-continuity']) == 0
    resumed_summary = json.loads((resumed/'summary.json').read_text())
    assert resumed_summary['stations'][1]['initial_guess']=='trial_checkpoint'
    assert resumed_summary['stations'][1]['release_fraction']==trial['release_fraction']
    with pytest.raises(SystemExit):
        main(['--seed',str(first/previous['folder']/'fields.npz'),
              '--start-fraction',str(previous['release_fraction']),
              '--trial-checkpoint',str(trial_path),'--output',str(tmp_path/'wrong_trial'),
              *common])  # 0.1+0.1 does not match the saved 0.25 trial.
