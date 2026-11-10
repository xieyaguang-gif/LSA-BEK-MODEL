#!/usr/bin/env python3
"""Independent same-grid check of the negative-Ro frequency/frame transformation."""
import hashlib,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMEXPR_MAX_THREADS','VECLIB_MAXIMUM_THREADS','BLIS_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from llsa import LocalProblem,Parameters,solve_bek
from llsa.secondary.discretization import rotation_rates
if '--help' in sys.argv:
 print(__doc__);raise SystemExit(0)
OUT=ROOT/'results/ro_scan_20261101'
rows=[]
for ro in [-1.,-.9,-.751,-.3,-.1]:
 re=500.;beta=68;alpha=.218+.122j;raw=.035+.002j;a=abs(ro);R=re/a;disk,external=rotation_rates(ro);physical_alpha=alpha.conjugate();Omega=beta*disk-re*raw.conjugate()
 p=LocalProblem(solve_bek(ro,zmax=40,tol=1e-10,samples=16001),Parameters(re=re,ro=ro,beta=beta,n=32,zmax=40,stretch=.1,boundary='matlab'));m=p.grid.size;f=p.flow;S=p.grid.S[1:-1];D=p.grid.D[1:-1];D2=p.grid.D2[1:-1];U,V,W,dU,dV,dW=[getattr(f,k)[1:-1,None] for k in ['U','V','W','dU','dV','dW']];k=beta/R
 blocks=np.zeros((m,4,m,4),complex);adv=(1j*physical_alpha*ro*U+1j*k*(disk+ro*V)+(physical_alpha**2+k*k)/R-1j*Omega/R)*S+ro*W/R*D-D2/R;cor=2*(disk+ro*V)/R*S
 for v in range(3):blocks[1:-1,v,:,v]=adv+ro/R*(dW if v==2 else U)*S
 blocks[1:-1,0,:,1]=-cor;blocks[1:-1,1,:,0]=cor;blocks[1:-1,0,:,2]=ro*dU*S;blocks[1:-1,1,:,2]=ro*dV*S;blocks[1:-1,0,:,3]=1j*physical_alpha*S;blocks[1:-1,1,:,3]=1j*k*S;blocks[1:-1,2,:,3]=D;blocks[1:-1,3,:,0]=(1j*physical_alpha+1/R)*S;blocks[1:-1,3,:,1]=1j*k*S;blocks[1:-1,3,:,2]=D
 rawmatrix=p.matrix(alpha)-raw*p.B;row_scale=np.tile([a,a,-a,-1],m);inverse_variables=np.tile([1,1,-1,-1/a],m);transformed=row_scale[:,None]*rawmatrix.conjugate()*inverse_variables[None,:];interior=np.arange(4*m).reshape(m,4)[1:-1].ravel();error=float(np.max(abs(transformed[interior]-blocks.reshape(4*m,4*m)[interior])));rows.append(dict(ro=ro,max_interior_operator_error=error,Omega_BEK_over_Omega_d=1/disk,passed=error<1e-11))
report=dict(verifier_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),passed=all(r['passed'] for r in rows),records=rows,derivation='q_inertial=diag(1,1,-1,-absRo)*conj(q_raw); row scaling diag(absRo,absRo,-absRo,-1). This yields alpha_inertial=conj(alpha_raw), Omega_inertial_BEK=beta*disk-Re*conj(omega_raw). Comparison uses the exact same local spectral grid and boundary exclusion, independently assembled inertial interior equations.',scope='The full cylindrical frozen global operator has additional radial scale/curvature terms and a different wall-normal closure. Its saddle is separately solved for S1; the mapped S0 frequency is not a global eigenfrequency.')
(OUT/'frequency_mapping_validation.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(frequency_mapping_passed=report['passed'],max_error=max(r['max_interior_operator_error'] for r in rows))))
