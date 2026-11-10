#!/usr/bin/env python3
"""Prepared autonomous B4 correction at a genuine first-pass upper root.
Not started by the audit. Fixed physical E1; no pseudo-arclength mean norm.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse,fcntl,json,time,traceback,gc
from pathlib import Path
import numpy as np
from scipy.interpolate import BarycentricInterpolator,interp1d
from scipy.optimize import brentq
from numpy.polynomial.legendre import leggauss
import fullbvp_phaseB_solver as solver
from fullbvp_phaseB_solver import ROOT,Mapping,transfer,unpack,common,write
from fullbvp_phaseB_report import diagnostic,compare_fields

class TightCorrector(solver.Corrector):
    def metrics(self,y):
        r=super().metrics(y)
        r['accepted']=bool(r['accepted'] and r['relative_field_inf']<1e-10
            and r['residual_over_fundamental_feedback']<1e-5
            and max(abs(r['phase']),abs(r['branch_constraint']))<1e-10)
        return r

def read(path):return json.loads(Path(path).read_text())
def mean_finite_height(base,other,zmax):
    b=np.load(ROOT/base['fields']);d=np.load(ROOT/other['fields'])
    r=b['radii'];r=r[r<=d['radii'][-1]]
    qq=b['field'].reshape(len(b['radii']),2*int(b['H'])+1,4,len(b['xi']))[:len(r),int(b['H']),:3]
    vv=d['field'].reshape(len(d['radii']),2*int(d['H'])+1,4,len(d['xi']))[:,int(d['H']),:3]
    vv=interp1d(d['radii'],vv,axis=0,kind='cubic')(r)
    z,w=leggauss(160);z=(z+1)*zmax/2;w=w*zmax/2
    x=np.array([brentq(lambda x:.6*x+.4*(x**3+.5*(1-x*x))-(a-2)/(a+2),-1,1) for a in z])
    u=BarycentricInterpolator(b['xi'],qq,axis=2)(x);v=BarycentricInterpolator(d['xi'],vv,axis=2)(x)
    rw=common.radial_weights(r)
    norm=lambda a:float(np.einsum('rvi,i,r->',abs(a)**2,w,rw,optimize=True))
    return dict(zmax=zmax,relative_velocity_L2=float(np.sqrt(norm(v-u)/norm(u))),
        common_R_range=[float(r[0]),float(r[-1])],definition='integral 0<=z<=zmax of |mean q velocities|² dz dR; no infinite-height mean norm')

def run(out,budget):
    out=Path(out).resolve();job=out/'upper_refinement_audit';job.mkdir(parents=True,exist_ok=True)
    lock=(job/'numerical.lock').open('a')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print(json.dumps(dict(status='busy',reason='Another B4 job holds the exclusive lock')))
        return 75
    for marker in ['B4_DONE.json','B4_FAILED.json']:
        old=job/marker
        if old.exists():old.rename(job/(old.stem+'_previous_'+str(time.time_ns())+old.suffix))
    raw=read(out/'campaign.json');source=next(p for p in raw['points'] if p['id']=='branch_0026_try0');A=source['A1']
    state=dict(status='running',source=source['id'],A1=A,matching='same physical E1',points=[],comparisons=[],budget_per_point_seconds=budget,
        CPU_threads=1,GPU=False,scope='strict upper reference, N40, dr~.25, H4 separately; no continuation or spectrum')
    write(job/'B4_SUMMARY.json',state)
    solver.Corrector=TightCorrector
    baseline=None
    for label,n,H,M in [('baseline_strict',32,3,800),('N40',40,3,800),('dr025',32,3,round((source['Rout']-400)/.25)),('H4',32,4,800)]:
        mapping=Mapping(n,H,M);src=baseline or source
        y=transfer(ROOT/src['fields'],mapping);p,*_=mapping.at(y[-1],y[-2])
        qref=unpack(y[:-2],mapping.nr,H,mapping.s)[:,H+1];qref/=common.mass_norm(p,qref)
        folder=job/label
        if folder.exists() and not (folder/'summary.json').exists():
            cached=[p.parent for p in job.glob(label+'_restart*/summary.json') if read(p).get('accepted')]
            if cached:folder=sorted(cached,key=lambda p:int(p.name.rsplit('_restart',1)[1]))[-1]
            else:
                retry=1
                while (job/(label+f'_restart{retry}')).exists():retry+=1
                folder=job/(label+f'_restart{retry}')
        if (folder/'summary.json').exists():
            rec=read(folder/'summary.json')
            if rec['accepted']:y=transfer(folder/'fields.npz',mapping)
        else:y,rec=solver.correct(mapping,qref,y,'energy',A,folder,budget,max_newton=10)
        good=rec['accepted'] and rec.get('highest_to_fundamental',1) <= .01 and rec.get('relative_field_inf',1)<1e-10
        item=dict(label=label,status='accepted' if good else 'failed_or_unresolved',summary=str(folder/'summary.json'))
        if good:
            point={k:rec[k] for k in ['Rout','omega','A1','energies','highest_to_fundamental','residual_inf','relative_field_inf','residual_over_fundamental_feedback','phase','branch_constraint','n','H','intervals']}
            point.update(id='B4_'+label,role='upper_mesh_refinement',fields=str((folder/'fields.npz').relative_to(ROOT)),
                summary=str((folder/'summary.json').relative_to(ROOT)),a=rec['Rout']-100,dr=(rec['Rout']-400)/M,accepted=True,resolved=True)
            diag=diagnostic(point,job)
            diag['physical_mean_energy_all_height']=None
            diag['legacy_harmonic_energy_shares_with_mean_invalid_physical_interpretation']=diag.pop('harmonic_energy_shares_with_conjugates',None)
            diag['legacy_mean_norm_invalid_physical_interpretation']=diag['energies'][0]
            write(job/'diagnostics'/(point['id']+'.json'),diag)
            if label=='baseline_strict':baseline=point
            else:
                bd=next(p for p in state['points'] if p['id']=='B4_baseline_strict')
                diff=compare_fields(baseline,point);legacy=diff.pop('mean_relative_mass')
                item.update(delta_Rout=point['Rout']-baseline['Rout'],delta_omega=point['omega']-baseline['omega'],
                    delta_front=None if diag['front_fig8'] is None or bd['front_fig8'] is None else diag['front_fig8']-bd['front_fig8'],
                    delta_gq=diag['g_q_570_600']-bd['g_q_570_600'],
                    delta_sponge_R3=diag['sponge_fraction_cylindrical_R3']-bd['sponge_fraction_cylindrical_R3'],
                    highest_to_fundamental=diag['highest_to_fundamental'],**diff,
                    legacy_mean_relative_mass_invalid_physical_interpretation=legacy,
                    finite_height_mean_comparisons=[mean_finite_height(baseline,point,z) for z in [20,40]])
            state['points'].append(diag)
        if label!='baseline_strict':state['comparisons'].append(item)
        state['last_attempt']=item;write(job/'B4_SUMMARY.json',state);gc.collect()
        if label=='baseline_strict' and not good:
            state['status']='baseline_strict_correction_failed';break
    if state['status']=='running':state['status']='completed' if len(state['points'])==4 else 'completed_with_unresolved_checks'
    write(job/'B4_SUMMARY.json',state)
    from fullbvp_phaseB_audit import main
    main(out)
    write(job/'B4_DONE.json',dict(status=state['status'],source=source['id'],accepted_points=len(state['points']),
        production_visual_review='required after report/figure refresh'))
    print(json.dumps(dict(status=state['status'],accepted_points=len(state['points']))))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',default=str(ROOT/'results/fullbvp_global_20261030/phaseB'));parser.add_argument('--budget',type=float,default=1200)
    a=parser.parse_args()
    try:exitcode=run(a.out,a.budget)
    except BaseException as exc:
        job=Path(a.out).resolve()/'upper_refinement_audit';job.mkdir(parents=True,exist_ok=True)
        write(job/'B4_FAILED.json',dict(exception=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc()))
        raise
    if exitcode:raise SystemExit(exitcode)
