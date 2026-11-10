#!/usr/bin/env python3
"""Finite independent branch survey and legacy-candidate audit for S0b."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMEXPR_MAX_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[key]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import argparse,hashlib,json,resource,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import roscan_s0b_v2 as m
import numpy as np
SOURCE=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

def same(a,b):
    return abs(a['beta']-b['beta'])<1e-3 and abs(a['Re_ca']/b['Re_ca']-1)<1e-5 and abs(complex(*a['alpha'])-complex(*b['alpha']))<1e-4

def run(ro,budget):
    _,hard=resource.getrlimit(resource.RLIMIT_AS);resource.setrlimit(resource.RLIMIT_AS,(8_000_000_000,hard))
    start=time.monotonic();deadline=start+budget;dest=m.RUN/m.old.tag(ro);primary=json.loads((dest/'branch_continued_physical_branch.json').read_text())
    results={'ro':ro,'primary_branch':primary['status'],'coverage':[],'legacy':[],'extra_branches':[],'unresolved':[],'search_complete':False,'source_sha256':SOURCE,'numerical_source_sha256':m.SOURCE}
    known=[primary['continuous_stationary']]
    def classify(rec,label):
        calc=m.Calculator(ro,128,40.,deadline)
        fine,_=calc.critical(rec['beta'],(*rec['alpha'],rec['Re_ca']),complex(*rec['omega_raw']))
        # Compare with each already resolved physical family at exactly this beta.
        for family in known:
            try:
                other,_=calc.critical(fine['beta'],(*family['alpha'],family['Re_ca']),complex(*family['omega_raw']))
                if same(fine,other):return dict(kind='duplicate',matched='continued_physical_branch' if family is known[0] else 'extra_branch',candidate=m.compact(fine))
            except m.old.BudgetExceeded:raise
            except Exception:pass
        opt=m.stationary(calc,fine)
        for family in known:
            if same(opt,family):return dict(kind='duplicate_at_minimum',candidate=m.compact(fine))
        branch=m.validate_branch(ro,opt,label,deadline);results['extra_branches'].append(label)
        if 'continuous_stationary' in branch:known.append(branch['continuous_stationary'])
        else:results['unresolved'].append(dict(label=label,reason=branch.get('error','unresolved minimum')))
        return dict(kind='new_family',label=label,status=branch['status'],candidate=m.compact(fine))
    try:
        # Audit every distinct legacy minimum/seed family before new probes.
        survey=json.loads((m.old.OUT/m.old.tag(ro)/'survey.json').read_text());legacy=[]
        for b in survey.get('branch_minima',[]):
            if b.get('minimum'):legacy.append(b['minimum'])
        if survey.get('candidate_minimum'):legacy.append(survey['candidate_minimum'])
        labels={r.get('seed_label','unknown') for r in survey.get('found_branches',[])}
        for label in labels:
            family=[r for r in survey['found_branches'] if r.get('seed_label','unknown')==label]
            if family:legacy.append(min(family,key=lambda r:r['Re_ca']))
        unique=[]
        for r in legacy:
            if not any(same(r,v) for v in unique):unique.append(r)
        for j,rec in enumerate(unique):
            if time.monotonic()>deadline:raise m.old.BudgetExceeded('Legacy audit budget exhausted')
            item={'seed':m.compact(rec)}
            try:item.update(classify(rec,f'legacy_{j}'))
            except m.old.BudgetExceeded:raise
            except Exception as ex:item.update(kind='unresolved',error=str(ex));results['unresolved'].append(dict(label=f'legacy_{j}',reason=str(ex)))
            results['legacy'].append(item);m.old.dump(dest/'branch_search.json',results)
        optimum=primary['minimum'];center=optimum['beta'];re=optimum['Re_ca']
        betas=sorted(set([0,round(center*.5),center,round(center*1.5),round(center*2)]))
        # Signed-imaginary alpha seeds probe additional saddle identities, including type-II candidates.
        seeds=[(.08,-.08,-.025),(.12,.04,.005),(.35,.20,.04),(.20,.32,.065)]
        coarse=m.Calculator(ro,64,40.,deadline)
        for beta in betas:
            for j,(ar,ai,omega) in enumerate(seeds):
                if time.monotonic()>deadline:raise m.old.BudgetExceeded('Independent probe budget exhausted')
                item=dict(beta=beta,alpha_seed=[ar,ai],omega_seed=omega,n=64,Re_seed=re)
                try:
                    rec,_=coarse.critical(beta,(ar,ai,re),omega)
                    item.update(coarse_candidate=m.compact(rec),**classify(rec,f'probe_b{beta}_s{j}'))
                except m.old.BudgetExceeded:raise
                except Exception as ex:item.update(kind='failed_seed',error=f'{type(ex).__name__}: {ex}')
                results['coverage'].append(item);m.old.dump(dest/'branch_search.json',results)
        results['search_complete']=True
    except Exception as ex:results['stopped_reason']=f'{type(ex).__name__}: {ex}'
    results.update(seconds=time.monotonic()-start,stats=dict(m.STATS),peak_RSS_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    m.old.dump(dest/'branch_search.json',results)
    print(json.dumps(dict(ro=ro,search_complete=results['search_complete'],legacy=len(results['legacy']),probes=len(results['coverage']),extra=results['extra_branches'],unresolved=results['unresolved'],seconds=results['seconds'],stopped=results.get('stopped_reason'))),flush=True)
    return results['search_complete']

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ro',nargs='+',type=float,default=[-.5,-.4,-.3,-.2,-.1]);p.add_argument('--budget',type=float,default=50);p.add_argument('--workers',type=int,default=1);a=p.parse_args()
    if not 1<=a.workers<=5:p.error('workers must be 1..5')
    if a.workers==1:
        for ro in a.ro:run(ro,a.budget)
    else:
        with ProcessPoolExecutor(max_workers=a.workers) as pool:list(pool.map(run,a.ro,[a.budget]*len(a.ro)))
