#!/usr/bin/env python3
"""Resolve legacy families using integer-beta continuation rather than a stationary Newton jump."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import json,time
from concurrent.futures import ProcessPoolExecutor
import roscan_s0b_v2 as m

def run(ro):
    start=time.monotonic();deadline=start+50;dest=m.RUN/m.old.tag(ro);search=json.loads((dest/'branch_search.json').read_text());legacy=next(x for x in search['legacy'] if x.get('kind')=='unresolved');seed=legacy['seed'];calc=m.Calculator(ro,128,40.,deadline)
    points={};bracket=False;errors=[];label='legacy_integer_walk'
    def solve(beta,state):
        if beta in points:return points[beta]
        r,_=calc.critical(beta,(*state['alpha'],state['Re_ca']),complex(*state['omega_raw']));r['beta']=int(beta);points[beta]=r;m.old.dump(dest/'legacy_walk'/f'beta{beta}.json',r);return r
    first=solve(int(seed['beta']),seed);initial_pinch=m.old.pinch(calc,first) if False else None
    try:
        for direction in [-1,1]:
            previous=first;higher=0;step=max(5,round(first['beta']*.05))
            for k in range(100):
                if time.monotonic()>deadline:raise m.old.BudgetExceeded('Legacy integer-walk budget exhausted')
                beta=max(0,previous['beta']+direction*step)
                if beta==previous['beta']:break
                try:r=solve(beta,previous)
                except m.old.BudgetExceeded:raise
                except Exception as ex:
                    errors.append(dict(beta=beta,error=str(ex)))
                    if step>1:step=max(1,step//2);continue
                    break
                higher=higher+1 if r['Re_ca']>previous['Re_ca'] else 0;previous=r
                if higher>=2 or beta==0:break
        best=min(points.values(),key=lambda r:r['Re_ca']);keys=sorted(points);j=keys.index(best['beta']);lo=keys[max(0,j-1)];hi=keys[min(len(keys)-1,j+1)]
        for beta in range(lo,hi+1):solve(beta,min(points.values(),key=lambda r:abs(r['beta']-beta)))
        best=min(points.values(),key=lambda r:r['Re_ca']);neighbors=[1] if best['beta']==0 else [-1,1];bracket=all(best['beta']+d in points and points[best['beta']+d]['Re_ca']>best['Re_ca'] for d in neighbors)
        m.old.near_roots=m.spatial_near;pin=m.old.pinch(calc,best);dense=m.dense_check(calc,best)
        branch=dict(label=label,status='neutral_minimum_pinch_unresolved',minimum=m.compact(best),integer_minimum_bracketed=bracket,pinch=pin,dense_validation=dense,convergence=[],beta_coverage=sorted(points),errors=errors,seconds=time.monotonic()-start,source='examples/roscan_s0b_legacy.py')
        if pin['status']=='finite_path_passed':
            for nn,zz in [(192,40.),(128,50.)]:
                c=m.Calculator(ro,nn,zz,deadline);r,_=c.critical(best['beta'],(*best['alpha'],best['Re_ca']),complex(*best['omega_raw']));diff=abs(r['Re_ca']/best['Re_ca']-1);branch['convergence'].append(dict(n=nn,zmax=zz,relative_Re_change=diff,passed=diff<.001,result=m.compact(r)))
            if bracket and all(x['passed'] for x in branch['convergence']):branch['status']='validated_pinch_minimum'
    except Exception as ex:
        best=min(points.values(),key=lambda r:r['Re_ca']);m.old.near_roots=m.spatial_near
        # The exact original candidate still receives a finite-path classification.
        calc.deadline=time.monotonic()+12
        try:pin=m.old.pinch(calc,first)
        except Exception as pinex:pin={'status':'unresolved','error':str(pinex)}
        branch=dict(label=label,status='partial',minimum=m.compact(best),integer_minimum_bracketed=bracket,pinch={'status':'pending_at_minimum'},original_seed_pinch=pin,beta_coverage=sorted(points),errors=errors,error=str(ex),seconds=time.monotonic()-start,source='examples/roscan_s0b_legacy.py')
    m.old.dump(dest/f'branch_{label}.json',branch)
    print(json.dumps(dict(ro=ro,status=branch['status'],Re=branch['minimum']['Re_ca'],beta=branch['minimum']['beta'],bracket=bracket,pinch=branch['pinch']['status'],original_seed_pinch=branch.get('original_seed_pinch',{}).get('status'),points=len(points),seconds=branch['seconds'],error=branch.get('error'))),flush=True)

if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=2) as pool:list(pool.map(run,[-.3,-.2]))
