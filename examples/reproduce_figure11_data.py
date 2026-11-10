"""Scan the Ro=-0.751 primary absolute saddle used for a Figure 11 remake."""
import json
from pathlib import Path
import numpy as np
from scipy import linalg, optimize
from llsa import LocalProblem, Parameters, solve_bek, temporal
RO=-.751; N=64; ZMAX=40.; STRETCH=.1
FLOW=solve_bek(RO,zmax=ZMAX,tol=1e-9,samples=8001)
def evaluate(x,beta,target=.04):
    p=LocalProblem(FLOW,Parameters(re=x[2],ro=RO,beta=beta,n=N,zmax=ZMAX,stretch=STRETCH,boundary='matlab'))
    alpha=complex(x[0],x[1]); spec=temporal(p,alpha,left=True); k=spec.select(target)
    omega=spec.values[k]; q=spec.vectors[:,k]; left=spec.left[:,k]
    deriv=np.vdot(left,(p.A1+2*alpha*p.A2)@q)/np.vdot(left,p.B@q)
    return np.array([deriv.real,deriv.imag,omega.imag]),complex(omega)
def main():
    out=Path('results/figure11_ro_minus0p751'); out.mkdir(parents=True,exist_ok=True)
    beta_rows=[]; seed=np.array([.2613,.1489,409.7])
    for beta in np.arange(60.,111.,2.):
        sol=optimize.root(lambda x:evaluate(x,beta)[0],seed,tol=2e-7,options={'maxfev':35}); err,om=evaluate(sol.x,beta)
        row={'beta':float(beta),'success':bool(sol.success and max(abs(err))<2e-6),'re':float(sol.x[2]),'alpha':[float(sol.x[0]),float(sol.x[1])],'omega':[float(om.real),float(om.imag)],'residual':err.tolist()}
        beta_rows.append(row)
        if row['success']: seed=sol.x
    freq_rows=[]; seed=np.array([.2613,.1489,409.7])
    target=.04
    for re in np.linspace(400.,1200.,81):
        sol=optimize.root(lambda x:evaluate([x[0],x[1],re],85,target)[0][:2],seed[:2],tol=2e-7,options={'maxfev':30}); err,om=evaluate([sol.x[0],sol.x[1],re],85,target)
        row={'re':float(re),'success':bool(sol.success and max(abs(err[:2]))<2e-6),'alpha':[float(sol.x[0]),float(sol.x[1])],'omega':[float(om.real),float(om.imag)],'saddle_residual':err.tolist()}
        freq_rows.append(row)
        if row['success']: seed=np.array([sol.x[0],sol.x[1],re]); target=om
    data={'ro':RO,'n':N,'zmax':ZMAX,'stretch':STRETCH,'boundary':'matlab','convention':'solver omega; paper frequency plotted as -Re(omega)','beta_scan':beta_rows,'frequency_scan_beta85':freq_rows,'paper_reference':{'re':409.6,'beta':85,'omega_real':-.038617}}
    (out/'figure11_data.json').write_text(json.dumps(data,indent=2)+'\n')
    print(json.dumps({'beta_success':sum(x['success'] for x in beta_rows),'frequency_success':sum(x['success'] for x in freq_rows),'output':str(out/'figure11_data.json')},indent=2))
if __name__=='__main__': main()
