#!/usr/bin/env python3
"""Independent small-negative-Ro check of Lingwood's Ekman limit, not new scan rows."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
import hashlib,json,time
from pathlib import Path
import roscan_s0b_v2 as m
start=time.monotonic();deadline=start+45
state=json.loads((m.RUN/m.old.tag(-.1)/'branch_continued_physical_branch.json').read_text())['minimum'];rows=[]
for ro in [-.05,-.01,-.001]:
    calc=m.Calculator(ro,128,40.,deadline); beta=state['beta']*abs(state['ro'])/abs(ro)
    seed,_=calc.critical(beta,(*state['alpha'],state['Re_ca']),complex(*state['omega_raw']))
    opt=m.stationary(calc,seed);dense=m.dense_check(calc,opt)
    rows.append(dict(ro=ro,minimum=m.compact(opt),spanwise_wavenumber=opt['beta']*abs(ro)/opt['Re_ca'],dense_validation=dense))
    state=opt;m.old.dump(m.OUT/'ekman_limit.json',dict(rows=rows,reference_Re=198.,reference_source='Lingwood (1997), JFM 331, 405-428; DOI 10.1017/S0022112096004144',scope='Continuous-beta neutral saddle minimum, independently dense-qualified; no exact-Ro=0 calculation and no integer optimization at these auxiliary points.',seconds=time.monotonic()-start,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
print(json.dumps([dict(ro=r['ro'],Re=r['minimum']['Re_ca'],beta=r['minimum']['beta'],alpha=r['minimum']['alpha'],k=r['spanwise_wavenumber'],dense_passed=r['dense_validation']['passed']) for r in rows]))
