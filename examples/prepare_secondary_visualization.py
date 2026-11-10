"""Export the actual N=60 primary and one secondary mode for offline visualization."""
import hashlib
import json
from pathlib import Path

import numpy as np

from llsa.secondary import PrimaryWave, FloquetProblem, secondary_spectrum
from llsa.secondary.time_domain import interpolate_primary

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/secondary_visualization'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT/'results/pier2003/wave_n60_h5.npz'
    wave = PrimaryWave.load(source)
    ah = -.00163221046-.03698086207j
    problem = FloquetProblem(wave, beta_hat=20, harmonics=5, wall_condition='continuity')
    spectrum = secondary_spectrum(problem, ah, target=8.60669+.85618j)
    k = spectrum.nearest(8.60669+.85618j)
    vector, omega = spectrum.vectors[:, k], spectrum.omega[k]
    secondary = PrimaryWave(wave.operators, wave.alpha, wave.omega,
                            vector.reshape(11, -1), 0., [])
    z = np.array([.4, .8, 1.2, 1.6, 2., 2.8, 4., 6.])
    primary = interpolate_primary(wave, z)[:, :3]*problem.radius
    mode = interpolate_primary(secondary, z)[:, :3]*problem.radius
    ns = np.arange(-5, 6)
    ref = 3
    phase = np.arange(16384)*2*np.pi/16384
    wphase = np.exp(1j*phase[:, None]*ns)@primary[:, 2, ref]
    origin = phase[np.argmax(wphase.real)]
    primary *= np.exp(1j*ns*origin)[:, None, None]
    mode *= np.exp(1j*ns*origin)[:, None, None]
    # A common translation of primary and secondary; arbitrary eigenvector phase
    # then chosen so the secondary is positive at x=theta=t=0, z=1.6.
    eigenphase = -np.angle(mode[:, 2, ref].sum())
    mode *= np.exp(1j*eigenphase)
    primary_rms = float(np.sqrt(np.sum(abs(primary[:, 2, ref])**2)))
    mode_rms = float(np.sqrt(.5*np.sum(abs(mode[:, 2, ref])**2)))
    mode *= primary_rms/mode_rms
    shares = problem.radius**2*np.einsum('hvi,ij,hvj->h',
             vector.reshape(11,4,-1)[:,:3].conj(), wave.operators.grid.mass,
             vector.reshape(11,4,-1)[:,:3]).real
    shares /= shares.sum()
    data = dict(radius=problem.radius, beta=68, beta_hat=20, alpha=wave.alpha,
                omega=wave.omega, alpha_hat=[ah.real,ah.imag], omega_hat=[omega.real,omega.imag],
                source=str(source.relative_to(ROOT)), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                n=60,H=5,K=5,mean_mode=wave.operators.mean_mode,secondary_wall='continuity',
                z=z.tolist(), harmonics=ns.tolist(),primary_real=primary.real.tolist(),
                primary_imag=primary.imag.tolist(),secondary_real=mode.real.tolist(),
                secondary_imag=mode.imag.tolist(),sideband_energy_fraction=shares.tolist(),
                primary_w_rms_reference=primary_rms,reference_z=1.6,
                scalar='normal velocity deviation from laminar W(z)',
                amplitude_definition='a = rms_theta(w_secondary) / rms_theta(w_primary) at x=0,z=1.6,t=0',
                normalization='secondary scaled once by positive factor, for all components and heights',
                phase_origin=float(origin),eigenvector_phase=float(eigenphase),
                eigen_residual=float(spectrum.residuals[k]),
                limitation='Frozen R=510 local parallel-mean primary; linear secondary superposition; no secondary nonlinear saturation.',
                repeat=dict(primary_degrees=360/68, floquet_factor_degrees=360/20,
                            adjacent_primary_phase_degrees=360*20/68, composite_degrees=90,
                            primary_cells_per_composite_period=17))
    # Direct numerical reconstruction checks with complete retained harmonics.
    angles=np.arange(8192)*2*np.pi/8192
    e1=np.exp(1j*angles[:,None]*68*ns)
    e2=np.exp(1j*angles[:,None]*(20+68*ns))
    p=(e1@primary[:,2,ref]).real
    q=(e2@mode[:,2,ref]).real
    ratio=float(np.sqrt(np.mean(q*q)/np.mean(p*p)))
    assert abs(ratio-1)<1e-12
    assert np.max(abs((e1@primary[:,2,ref]).imag))<1e-10
    shift=np.exp(1j*(20+68*ns)*2*np.pi/68)
    cell_error=float(np.max(abs(shift-np.exp(1j*2*np.pi*20/68))))
    repeat_error=float(np.max(abs(np.exp(1j*(20+68*ns)*np.pi/2)-1)))
    assert max(cell_error,repeat_error)<1e-12
    data['reconstruction_checks']=dict(reference_rms_ratio=ratio,cell_phase_error=cell_error,
                                      quarter_circle_repeat_error=repeat_error)
    (OUT/'mode_data.json').write_text(json.dumps(data,ensure_ascii=False,separators=(',',':'))+'\n')
    np.savez_compressed(OUT/'mode_coefficients.npz',primary=primary,secondary=mode,z=z,
                        ns=ns,alpha_hat=ah,omega_hat=omega,raw_vector=vector)
    print(json.dumps({k:data[k] for k in ['omega_hat','eigen_residual','repeat','reconstruction_checks']},indent=2))
    print('Sidebands:',list(zip((20+68*ns).tolist(),np.round(shares,6).tolist())))


if __name__=='__main__':
    main()
