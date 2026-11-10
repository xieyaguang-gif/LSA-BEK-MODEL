#!/usr/bin/env python3
"""Collect all pilot attempts and compare accepted roots at common stations."""
import json
from pathlib import Path
import sys
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from llsa.secondary.operators import PierOperators

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/ro_minus1_beta68_free_bvp_20261029'


def load(name):
    with np.load(OUT/name) as data:
        return {key:data[key].copy() for key in data.files}


def comparison(first,second,rmax=540.):
    a,b=load(first),load(second)
    na,nb=a['modes'].shape[-1]//4-1,b['modes'].shape[-1]//4-1
    if na!=nb:
        raise ValueError('This comparison expects matching wall-normal grids')
    m=na+1
    M=PierOperators(n=na).grid.mass
    r=a['radii'];q=a['modes'];qq=b['modes'][:len(r)]
    if not np.array_equal(r,b['radii'][:len(r)]):
        raise ValueError('This comparison expects matching common radial nodes')
    H=(q.shape[1]-1)//2;HH=(qq.shape[1]-1)//2
    v=q[:,H+1,:3*m].reshape(len(r),3,m)
    vv=qq[:,HH+1,:3*m].reshape(len(r),3,m)
    mean=q[:,H,:3*m].reshape(len(r),3,m)
    meanb=qq[:,HH,:3*m].reshape(len(r),3,m)
    mask=r<=rmax
    def norm2(x):
        xx=x.reshape(-1,m)
        return float(np.einsum('vi,ij,vj->',xx.conj(),M,xx).real)
    phase=np.angle(np.einsum('rvi,ij,rvj->',v[mask].conj(),M,vv[mask]))
    vv=vv*np.exp(-1j*phase)
    rows=[]
    for R in [520,530,540,548]:
        i=int(np.argmin(abs(r-R)))
        ea=r[i]**2*norm2(v[i]);eb=r[i]**2*norm2(vv[i])
        rows.append(dict(R=float(r[i]),E1_first=ea,E1_second=eb,
            E1_relative_change=eb/ea-1,
            complex_velocity_profile_relative_l2=np.sqrt(norm2(vv[i]-v[i])/norm2(v[i])),
            mean_profile_relative_l2=np.sqrt(norm2(meanb[i]-mean[i])/norm2(mean[i])),
            mean_first_l2=np.sqrt(norm2(mean[i])),mean_second_l2=np.sqrt(norm2(meanb[i])),
            mean_difference_l2=np.sqrt(norm2(meanb[i]-mean[i])),
            mean_first_max=float(abs(mean[i]).max()),mean_second_max=float(abs(meanb[i]).max())))
    return dict(first=str(first),second=str(second),common_unsponged_region=[450,rmax],
        alignment_phase=float(phase),omega_change=float(b['omega']-a['omega']),
        omega_relative_change=float(b['omega']/a['omega']-1),
        common_complex_velocity_relative_l2=np.sqrt(norm2(vv[mask]-v[mask])/norm2(v[mask])),
        common_mean_relative_l2=np.sqrt(norm2(meanb[mask]-mean[mask])/norm2(mean[mask])),stations=rows)


def main():
    records=[]
    for path in sorted(OUT.rglob('*.json')):
        data=json.loads(path.read_text())
        if not isinstance(data,dict) or not {'history','status','omega'}<=data.keys():
            continue
        meta={}
        if path.with_suffix('.npz').exists():
            saved=load(path.with_suffix('.npz').relative_to(OUT))
            meta=json.loads(str(saved['metadata_json']))
            config=json.loads(str(saved['config_json']))
            meta.setdefault('n',config['n']);meta.setdefault('h',config['harmonics'])
            meta.setdefault('rmin',float(saved['radii'][0]));meta.setdefault('rmax',float(saved['radii'][-1]))
        auxiliary=bool(meta.get('auxiliary',data.get('growth',0.)!=0.))
        classification=('auxiliary_converged' if auxiliary else 'accepted_nonzero') if data['status']=='converged' else ('trivial' if data['status']=='trivial' else 'failed')
        record={k:v for k,v in data.items() if k!='history'}
        if 'tolerance' not in record:
            record['tolerance']=1e-7 if data['name']=='physical_free' else (1e-6 if data['name'].startswith('aux_relaxed') else 1e-9)
        record.update(path=str(path.relative_to(OUT)),classification=classification,auxiliary=auxiliary,
            n=meta.get('n'),h=meta.get('h'),rmin=meta.get('rmin'),rmax=meta.get('rmax'),
            sponge=meta.get('sponge'),omega_inertial=data['omega'],omega_disk_signed=data['omega']-68,
            pattern_speed_over_disk=data['omega']/68,
            final_scaled_field_residual=data['history'][-1]['scaled_residual'],
            iterations=data['history'][-1]['iteration'])
        records.append(record)
    result=dict(statement='Finite-domain unforced free-frequency branch existence demonstrated; domain/harmonic independence and temporal attraction not established.',
        total_recorded_newton_seconds=sum(x['seconds'] for x in records),attempts=records)
    accepted=[x for x in records if x['classification']=='accepted_nonzero']
    result['accepted_nonzero_count']=len(accepted)
    (OUT/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    comp=comparison('amplitude_n16_mode4/fields.npz','domain_r600_n16_h1/fields.npz')
    (OUT/'domain_common_region_comparison.json').write_text(json.dumps(comp,indent=2)+'\n')
    h2=OUT/'refine_n16_h2_resumed/fields.npz'
    if h2.exists():
        comp2=comparison('amplitude_n16_mode4/fields.npz',h2.relative_to(OUT))
        (OUT/'harmonic_common_region_comparison.json').write_text(json.dumps(comp2,indent=2)+'\n')
    third=OUT/'domain_r620_n16_h1/fields.npz'
    if third.exists():
        comp3=comparison('domain_r600_n16_h1/fields.npz',third.relative_to(OUT))
        (OUT/'domain_600_620_common_region_comparison.json').write_text(json.dumps(comp3,indent=2)+'\n')
    print(json.dumps(dict(attempts=len(records),accepted_nonzero=len(accepted),newton_seconds=result['total_recorded_newton_seconds'])))


if __name__=='__main__':
    main()
