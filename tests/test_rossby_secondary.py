import numpy as np
import pytest
from scipy import linalg

from llsa.secondary import PrimaryWave, PierOperators, WaveSolver, FloquetProblem, secondary_spectrum, continue_ro
from llsa.secondary.discretization import rotation_rates, sample_inertial_base
from llsa.secondary.equation_audit import independent_laminar_check, floquet_full_ns_check
from llsa.secondary.time_domain import LinearTimeProblem
from llsa.secondary.validation import compare_tables


@pytest.fixture(scope='module')
def waves():
    solver=WaveSolver(PierOperators(n=24,mean_mode='parallel'),1)
    original=solver.solve(solver.bootstrap(),frequency=50.5)
    continued,_=continue_ro(original,[-.751])
    return original,continued[0]


@pytest.mark.parametrize('ro',[-1.,-.751,-.4,0.,.5,1.])
def test_inertial_bvp_independently_recovers_bek(ro):
    assert independent_laminar_check(ro)['passed']


def test_reference_rates_and_nonzero_rotating_far_field():
    # From Ro=(Omega_f-Omega_d)/Omega and the paper's implicit definition of Omega.
    for ro in [-1,-.751,0,.5,1]:
        disk,external=rotation_rates(ro)
        assert external-disk==pytest.approx(ro)
        assert external/(2-ro)+disk/(2+ro)==pytest.approx(1)
        U,V,W,dU,dV,dW=sample_inertial_base(np.array([0.,40.,100.,np.inf]),ro)
        assert V[0]==pytest.approx(disk,abs=1e-12)
        np.testing.assert_allclose(V[1:],external,atol=1e-14)
        np.testing.assert_allclose([U[0],W[0]],0,atol=1e-12)
    assert rotation_rates(-.751)==pytest.approx((1.0934995,.3424995))


def test_saved_ro_roundtrip_and_legacy_files(waves,tmp_path):
    old,new=waves
    for wave in waves:
        path=tmp_path/f'{wave.operators.ro}.npz'
        wave.save(path)
        restored=PrimaryWave.load(path)
        assert restored.operators.ro==wave.operators.ro
        np.testing.assert_array_equal(restored.modes,wave.modes)
        np.testing.assert_array_equal(restored.operators.base,wave.operators.base)
    with np.load(tmp_path/'-1.0.npz') as d:
        legacy={k:d[k] for k in d.files if k not in ('ro','schema_version','convention')}
    np.savez(tmp_path/'legacy.npz',**legacy)
    assert PrimaryWave.load(tmp_path/'legacy.npz').operators.ro==-1.


def test_rossby_branch_constraint_and_full_ns_linearization(waves):
    old,new=waves
    assert new.operators.ro==-.751
    assert new.operators.radius==old.operators.radius
    assert new.energies()[0]==pytest.approx(old.energies()[0],rel=1e-7)
    assert new.residual<1e-9
    assert floquet_full_ns_check(new,harmonics=1)['passed']
    assert compare_tables(new)['table2']['passed'] is None


def test_zero_amplitude_secondary_reduces_to_laminar_sidebands(waves):
    wave=waves[1]
    zero=PrimaryWave(wave.operators,wave.alpha,wave.omega,np.zeros_like(wave.modes),0.,[])
    problem=FloquetProblem(zero,beta_hat=20,harmonics=1)
    ah=.01-.025j
    spec=secondary_spectrum(problem,ah,target=8+1j,count=4)
    expected=[]
    for n in [-1,0,1]:
        values=linalg.eigvals(wave.operators.linear(ah+n*wave.alpha,20+n*wave.operators.beta),
                             1j*wave.operators.M)
        expected.extend(values[np.isfinite(values)]*wave.operators.radius-n*wave.omega)
    for omega in spec.omega:
        assert min(abs(np.asarray(expected)-omega))<1e-7


def test_time_solver_loads_same_ro_and_retains_far_rotation(waves):
    old,new=waves
    time=LinearTimeProblem(new,harmonics=1,n=24,zmax=120)
    assert time.ro==-.751
    assert time.V[-1]==pytest.approx(new.operators.external_rotation)
    reference=sample_inertial_base(time.grid.z,-.751)
    np.testing.assert_array_equal(time.U,reference[0])
    np.testing.assert_array_equal(time.V,reference[1])
    source=LinearTimeProblem(old,harmonics=1,n=24,zmax=120)
    assert np.max(abs(time.explicit_rhs(time.random_state(amplitude=1))-
                         source.explicit_rhs(source.random_state(amplitude=1))))>1e-3


def test_fixed_re_path_and_zero_shear_rejection(waves):
    old,_=waves
    result,_=continue_ro(old,[-.99],fixed_re=True)
    assert result[0].operators.re==pytest.approx(old.operators.re)
    assert result[0].operators.radius==pytest.approx(510/.99)
    with pytest.raises(ValueError,match='Ro=0'):
        continue_ro(old,[0])
    with pytest.raises(ValueError,match='Ro=0'):
        WaveSolver(PierOperators(ro=0,n=24),1).solve(old,energy=10)


def test_cli_never_relabels_existing_wave_ro(waves,tmp_path):
    from llsa.secondary.cli import main
    source=tmp_path/'wave.npz';waves[1].save(source)
    with pytest.raises(SystemExit) as error:
        main(['pier-spectrum','--wave',str(source),'--ro=-1','--output',str(tmp_path/'bad')])
    assert error.value.code==2
    code=main(['pier-ro-scan','--wave',str(source),'--ro-values','-0.751',
               '--output',str(tmp_path/'scan')])
    assert code==0
    assert PrimaryWave.load(tmp_path/'scan_ro_000_wave.npz').operators.ro==-.751


def test_time_solver_rejects_stale_ro_metadata(waves):
    old,new=waves
    stale=PrimaryWave(new.operators,old.alpha,old.omega,old.modes,0.,[])
    with pytest.raises(ValueError,match='Ro/model'):
        LinearTimeProblem(stale,n=24,harmonics=1)


def test_pressure_elimination_carries_general_rotation():
    from examples.pier_pressure_elimination import finite_op
    op=finite_op(24,120,ro=-.751,radius=510/.751,beta=68)
    assert op.ro==-.751
    assert op.re==pytest.approx(510)
    assert op.base[1][-1]==pytest.approx(.3424995)


def test_failed_scan_retains_last_converged_checkpoint(waves,tmp_path,monkeypatch):
    import json
    from llsa.secondary import cli
    source=tmp_path/'wave.npz';waves[0].save(source)
    def interrupted(seed,values,**kwargs):
        kwargs['callback'](seed,dict(ro=seed.operators.ro,residual=seed.residual))
        raise RuntimeError('controlled continuation failure')
    monkeypatch.setattr(cli,'continue_ro',interrupted)
    with pytest.raises(SystemExit):
        cli.main(['pier-ro-scan','--wave',str(source),'--ro-values','-0.751',
                  '--output',str(tmp_path/'partial')])
    metadata=json.loads((tmp_path/'partial.json').read_text())
    assert metadata['status']=='failed'
    assert len(metadata['trajectory'])==1
    assert PrimaryWave.load(tmp_path/'partial_latest_wave.npz').operators.ro==-1.
