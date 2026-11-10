import numpy as np
import pytest
from scipy import sparse

from llsa.secondary.global_radial import GlobalRadial, OutletSponge, BlendedSponge
from llsa.secondary.radial import RadialConfig, pack_field, unpack_field
from llsa.secondary.radial_arclength import DampingArc


@pytest.mark.parametrize('order',[0,1,2])
def test_arc_cli_keeps_auxiliary_points_separate_from_target(tmp_path,order):
    from dataclasses import asdict
    import json
    from llsa.secondary.radial_arc_cli import main
    c = RadialConfig(n=8,harmonics=1)
    r = np.linspace(320/abs(c.ro),325/abs(c.ro),15)
    q = np.zeros((len(r),3,4*(c.n+1)),complex)
    settings = dict(old_start=321.,target=323.,fixed_end=325.,width=2.,strength=10.,
                    damp_mean=False,outlet_alpha='0j')
    for name,fraction in [('previous',.1),('current',.2)]:
        np.savez(tmp_path/f'{name}.npz',radii=r,modes=q,converged=True,
            config_json=json.dumps(asdict(c)),release_json=json.dumps(dict(**settings,fraction=fraction)))
    output = tmp_path/'output'
    assert main(['--previous-seed',str(tmp_path/'previous.npz'),'--seed',str(tmp_path/'current.npz'),
        '--predictor-order',str(order),'--step','.4','--max-step','.5','--max-stations','3','--output',str(output)]) == 0
    summary = json.loads((output/'summary.json').read_text())
    assert summary['status']=='target_reached_discrete_equations'
    assert summary['stations'][-1]['release_fraction']==1.
    assert summary['stations'][-1]['status']=='converged_discrete_equations'
    assert all(s['status']=='converged_auxiliary_damping' for s in summary['stations'][:-1])
    assert summary['stations'][-2]['release_fraction']>1  # Still auxiliary, never a physical target.


@pytest.mark.parametrize('order',[0,2])
def test_arclength_crosses_a_known_simple_fold(order):
    """Embed x^2+(tau-3)*x+1=0 in the corrector's field layout.

    This tests continuation geometry, not a claim of a fold in the flow.
    The exact branch tau=3-x-1/x turns at (x,tau)=(1,1).
    """
    r = np.arange(420.,427.)
    old,new = OutletSponge(421.,2.,3.,True),OutletSponge(423.,2.,3.,True)
    g = GlobalRadial(r,RadialConfig(n=8,harmonics=1),outlet_sponge=old)
    size = len(r)*g.width
    k = 3*g.width+1
    d = (new.profile(r)[3]-old.profile(r)[3])/r[3]
    def polynomial(x,inlet,*,jacobian=False,**kwargs):
        f = x.copy()
        f[k] = d*(x[k]**2-3*x[k]+1)
        if not jacobian:
            return f
        diagonal = np.ones(size)
        diagonal[k] = d*(2*x[k]-3)
        return f,sparse.diags(diagonal,format='csc')
    g.evaluate = polynomial
    def quadratic(q,dr):
        x = pack_field(q)
        result = np.zeros(size);result[k] = d*x[k]**2
        return unpack_field(result,len(r),g.H,g.s)
    g._convection_field = quadratic
    def field(a):
        x = np.zeros(size);x[k] = a
        return unpack_field(x,len(r),g.H,g.s)
    previous,current = field(.75),field(.85)
    lp,lc = 3-.75-1/.75,3-.85-1/.85
    fractions = [lc]
    for _ in range(6):
        arc = DampingArc(g,new,current[0],previous,current,lp,lc,.12)
        if order:
            info = arc.prepare_predictor(sparse.linalg.spsolve,order)
            assert info['tangent_flow_residual_inf'] < 1e-10
            assert info['curvature_flow_residual_inf'] < 1e-10
        result = g.solve(current[0],initial=arc.physical_field(arc.predictor),
            augmentation=arc,tol=1e-11,max_iter=12)
        value = pack_field(result.modes)[k]
        fraction = arc.accepted_state[-1]
        assert abs(fraction-(3-value-1/value)) < 1e-10
        assert np.linalg.norm(arc.physical_residual(arc.accepted_state),np.inf) < 1e-10
        previous,current,lp,lc = current,result.modes,lc,fraction
        fractions.append(fraction)
    assert value > 1.2
    assert max(fractions) > fractions[-1]+.02


@pytest.mark.parametrize('pin',[None,423.])
@pytest.mark.parametrize('solver',['direct','pardiso-gmres'])
def test_arclength_jacobian_and_independent_fixed_fraction_flow(pin,solver):
    if solver.startswith('pardiso'):
        pytest.importorskip('pypardiso')
    r = np.arange(420.,427.)
    c = RadialConfig(n=8,harmonics=2)
    old,new = OutletSponge(421.,2.,3.,True),OutletSponge(423.,2.,3.,True)
    g = GlobalRadial(r,c,outlet_sponge=old)
    rng = np.random.default_rng(201)
    q = unpack_field(1e-4*rng.normal(size=len(r)*g.width),len(r),g.H,g.s)
    forcing = unpack_field(g.evaluate(pack_field(q),q[0],outlet=q[-1]),len(r),g.H,g.s)
    fixed = GlobalRadial(r,c,outlet_sponge=BlendedSponge(old,new,.1))
    current = fixed.solve(q[0],initial=q,outlet=q[-1],forcing=forcing,tol=1e-11).modes
    arc = DampingArc(g,new,q[0],q,current,0.,.1,.07,
        phase_pin_radius=pin,outlet=q[-1],forcing=forcing)
    state = arc.predictor.copy()
    f,J = arc.evaluate(state,jacobian=True)
    v = rng.normal(size=len(state))
    h = 1e-6
    fd = (arc.evaluate(state+h*v)-arc.evaluate(state-h*v))/(2*h)
    np.testing.assert_allclose(J@v,fd,rtol=2e-7,atol=2e-8)
    h2 = 1e-3
    second_fd = (arc.evaluate(state+h2*v)-2*arc.evaluate(state)+arc.evaluate(state-h2*v))/h2**2
    np.testing.assert_allclose(arc.directional_second_derivative(state,v),second_fd,rtol=2e-6,atol=1e-7)
    info = arc.prepare_predictor(sparse.linalg.spsolve,2)
    assert info['tangent_flow_residual_inf'] < 1e-8
    assert info['curvature_flow_residual_inf'] < 1e-8
    solution = g.solve(q[0],initial=arc.physical_field(state),outlet=q[-1],forcing=forcing,
        augmentation=arc,tol=1e-10,max_iter=15,phase_update=True,phase_continuity=True,
        linear_solver=solver,preconditioner_refresh=3)
    fraction = arc.accepted_state[-1]
    assert .1 < fraction < 1
    independent = GlobalRadial(r,c,outlet_sponge=BlendedSponge(old,new,fraction))
    residual = independent.evaluate(pack_field(solution.modes),q[0],outlet=q[-1],forcing=forcing)
    assert np.linalg.norm(residual,np.inf)/np.max(abs(solution.modes)) < 2e-9
