"""Full-resolution Ro=-1 regression, then finite-amplitude Ro continuation.

Uses the preserved N=60,H=5 reference wave; all new results go to ro_extension/.
The branch constraint is fixed E1, Re and beta, not a natural-selection claim.
"""
import json
from pathlib import Path
import numpy as np

from llsa.secondary import (PrimaryWave, WaveSolver, FloquetProblem, secondary_saddle,
                            secondary_spectrum, continue_ro)
from llsa.secondary.equation_audit import independent_laminar_check, floquet_full_ns_check

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/ro_extension'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    reference=PrimaryWave.load(ROOT/'results/pier2003/wave_n60_h5.npz')
    seed=PrimaryWave(reference.operators,reference.alpha,reference.omega,
                     reference.modes*1.001,1.,[])
    regression=WaveSolver(reference.operators,5).solve(seed,frequency=50.5)
    regression.save(OUT/'regression_ro_minus1_wave.npz')
    saddle=secondary_saddle(FloquetProblem(regression,beta_hat=20,harmonics=5,wall_condition='continuity'),
                            -.00163221046-.03698086207j,8.606689722678+.856178080687j)
    regression_result=dict(alpha=regression.alpha,energies=regression.energies().tolist(),
        alpha_difference=regression.alpha-reference.alpha,
        energy_max_difference=float(max(abs(regression.energies()-reference.energies()))),
        alpha_hat=[saddle.alpha_hat.real,saddle.alpha_hat.imag],
        omega_hat=[saddle.omega_hat.real,saddle.omega_hat.imag],group_velocity=abs(saddle.group_velocity),
        omega_difference=abs(saddle.omega_hat-(8.606689722678+.856178080687j)))
    regression_result['passed']=bool(abs(regression_result['alpha_difference'])<1e-7 and
        regression_result['energy_max_difference']<.001 and regression_result['omega_difference']<1e-5 and saddle.converged)
    print('Ro=-1 regression:',json.dumps(regression_result),flush=True)
    def record(wave,row):
        print('Ro step:',json.dumps(row),flush=True)
        wave.save(OUT/'latest_continuation_wave.npz')
    waves,path=continue_ro(regression,[-.95,-.9,-.85,-.8,-.751],fixed_re=True,callback=record)
    records=[]
    temporal_target=9.02669363981+2.99170079257j
    previous_ro=-1
    from llsa.secondary.discretization import rotation_rates
    for wave in waves:
        ro=wave.operators.ro
        name=f'ro_{ro:.3f}_wave.npz'
        wave.save(OUT/name)
        temporal_target += 20*(rotation_rates(ro)[1]-rotation_rates(previous_ro)[1])
        problem=FloquetProblem(wave,beta_hat=20,harmonics=5,wall_condition='continuity')
        spec=secondary_spectrum(problem,0,target=temporal_target,count=10)
        temporal_target=complex(spec.omega[spec.nearest(temporal_target)])
        row=dict(ro=ro,radius=wave.operators.radius,re=wave.operators.re,wave=name,alpha=wave.alpha,
                 omega=wave.omega,energies=wave.energies().tolist(),residual=wave.residual,
                 secondary_alpha_hat=[0.,0.],tracked_temporal_omega=[temporal_target.real,temporal_target.imag],
                 near_target_spectrum=[[v.real,v.imag] for v in spec.omega])
        records.append(row);previous_ro=ro
        print('Completed:',json.dumps(row),flush=True)
    audit=floquet_full_ns_check(waves[-1],harmonics=5)
    data=dict(regression=regression_result,branch_constraint='fixed E1, Re=510, beta=68; R=510/abs(Ro); alpha and omega solved',
              branch=records,trajectory=path,full_ns_audit=audit,
              independent_baseflow=[independent_laminar_check(r) for r in [-1,-.751,-.4,0,.5,1]],
              pier_table3_verified=False)
    (OUT/'validation.json').write_text(json.dumps(data,indent=2)+'\n')
    if not regression_result['passed'] or not audit['passed']:
        raise RuntimeError('Regression or full NS audit failed; inspect validation.json')
    print('Saved',OUT/'validation.json')


if __name__=='__main__':
    main()
