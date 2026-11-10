"""Independent phase/units checks, local-wave comparison and spectral checks."""
from fullbvp_phaseE_adapter import *
from unittest.mock import patch
from dataclasses import replace
from scipy import linalg
from llsa.secondary.discretization import pier_grid
from llsa.secondary.operators import PierOperators
from llsa.secondary.primary import PrimaryWave, WaveSolver
from llsa.secondary.dns_floquet import DNSFloquetProblem, eigen_continuity
from llsa.secondary.floquet import FloquetProblem, secondary_spectrum
from fullbvp_phaseA_spatial import Frozen


def correlation(q,v,mass):
    inner=lambda a,b: np.einsum('vi,ij,vj->',a.conj(),mass,b)
    scale=inner(q,v)/inner(q,q)
    overlap=abs(inner(q,v))/np.sqrt(inner(q,q).real*inner(v,v).real)
    error=np.sqrt(inner(v-scale*q,v-scale*q).real/inner(v,v).real)
    return dict(velocity_overlap=float(overlap),phase_and_amplitude_aligned_error=float(error))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    checks={}
    # An independent analytic carrier detects phase sign, an extra factor R,
    # and accidental conjugation. The profile is polynomial in source xi.
    r=np.linspace(558,562,81);xi=pier_grid(32).xi
    q=np.zeros((len(r),13,4,len(xi)),complex)
    q[:,7,1]=np.exp((.12+.21j)*(r[:,None]-560))*(1-xi**2)[None,:]
    q[:,5]=q[:,7].conj()
    with patch('fullbvp_phaseE_adapter.source',return_value=(r,xi,q,52.68143270236591)):
        b=frozen_base(560,n=24,H=1,zmax=30.)
    checks['manufactured_alpha_error']=abs(b.alpha-.21)
    checks['manufactured_envelope_error']=abs(b.metadata['envelope_growth']-.12)
    assert checks['manufactured_alpha_error']<1e-7
    assert checks['manufactured_envelope_error']<1e-7
    target=1-source_xi(b.operators.grid.z)**2
    checks['manufactured_profile_max_error']=float(np.max(abs(b.modes[2].reshape(4,-1)[1]-target)))
    assert checks['manufactured_profile_max_error']<1e-12
    assert b.omega==52.68143270236591

    # An existing converged PrimaryWave must give exactly the same matrix.
    old=PrimaryWave.load(ROOT/'results/pier2003/cyl_wave_n30_h1.npz')
    dns=DNSFrozenBase(old.operators,old.alpha,old.omega,old.modes.copy(),{})
    p=DNSFloquetProblem(dns,beta_hat=17.,harmonics=2)
    original=FloquetProblem(old,beta_hat=17.,harmonics=2,wall_condition='continuity')
    a=.073-.021j
    delta=p.matrix(a)-original.matrix(a)
    checks['existing_PrimaryWave_matrix_max_error']=float(np.max(abs(delta.data))) if delta.nnz else 0.
    assert checks['existing_PrimaryWave_matrix_max_error']<1e-11
    b=frozen_base(560,n=32,H=1,zmax=30.)
    p=DNSFloquetProblem(b,beta_hat=17.,harmonics=2)
    poly=p.A0+a*p.A1+a*a*p.A2
    delta=p.matrix(a)-poly
    checks['matrix_polynomial_max_error']=float(np.max(abs(delta.data))) if delta.nnz else 0.
    assert checks['matrix_polynomial_max_error']<1e-11
    # Reversing detuning and beta conjugates the real physical operator.
    pm=DNSFloquetProblem(b,beta_hat=-17.,harmonics=2)
    s=secondary_spectrum(p,.073,target=10+2j,count=8)
    w=s.omega[np.argmax(s.omega.imag)]
    sm=secondary_spectrum(pm,-.073,target=-w.conjugate(),count=8)
    checks['real_field_spectral_symmetry_error']=float(np.min(abs(sm.omega+w.conjugate())))
    assert checks['real_field_spectral_symmetry_error']<1e-6

    # The small-amplitude global wave is a spatially growing wave, not a
    # real-alpha neutral local wave. Compare both well-defined local objects.
    full,dq,xi,omega=section(560.)
    native=full[7,:3]
    spatial={}
    for kind in ('global','pier'):
        op=Frozen(560.,32,kind)
        roots,vectors,res=op.spatial(omega)
        aim=b.alpha-1j*b.metadata['envelope_growth']
        j=int(np.argmin(abs(roots-aim)))
        spatial[kind]=dict(alpha=pair(roots[j]),inferred_global_alpha=pair(aim),
            alpha_distance=float(abs(roots[j]-aim)),residual=float(res[j]),
            **correlation(vectors[:,j].reshape(4,-1)[:3],native,op.mass))
    energy=float(560.**2*np.einsum('vi,ij,vj->',native.conj(),pier_grid(32).mass,native).real)
    local=dict(status='not_attempted',R=560.,energy=energy)
    try:
        solver=WaveSolver(PierOperators(radius=560.,beta=68.,n=32,ro=-1.),harmonics=1)
        seed=solver.linear_seed(energy=energy,bracket=(.6,.7),target=70.)
        wave=solver.solve(seed,energy=energy,max_iter=25)
        wave.save(OUT/'local_primary_H1_R560.npz')
        local.update(status='converged',alpha=wave.alpha,omega=wave.omega,residual=wave.residual,
            alpha_difference=wave.alpha-b.alpha,omega_difference=wave.omega-b.omega,
            **correlation(wave.modes[2].reshape(4,-1)[:3],native,wave.operators.grid.mass))
    except Exception as exc:
        local.update(status='failed',error=str(exc))
    write_json(OUT/'validation.json',dict(checks=checks,passed=True,
        source_sha256=SOURCE_SHA,small_amplitude_frozen=b.metadata,
        spatial_comparison=spatial,local_PrimaryWave=local,
        existing_PrimaryWave=dict(path=str(ROOT/'results/pier2003/cyl_wave_n30_h1.npz'),
            radius=old.operators.radius,alpha=old.alpha,omega=old.omega,
            purpose='operator/unit equivalence; R510 is not a same-radius physical comparison'),
        interpretation='Same-radius real-alpha PrimaryWave fixes energy and solves neutral omega; the BVP fixes omega and has complex effective spatial alpha. Agreement of these physically different solutions is not imposed.'))
    print(json.dumps(dict(checks=checks,local_PrimaryWave=local,spatial_comparison=spatial),ensure_ascii=False))


if __name__=='__main__':main()
