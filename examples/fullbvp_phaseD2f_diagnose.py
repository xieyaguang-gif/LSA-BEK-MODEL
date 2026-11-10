#!/usr/bin/env python3
"""Bounded SVD diagnosis of projected accepted fields; not a stability result."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key]='1'
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy import linalg,sparse
import fullbvp_phaseD2e_coarse as old


def equilibrate(K,steps=6):
    A=K.astype(complex).tocsr(copy=True)
    rows=np.ones(A.shape[0]);cols=np.ones(A.shape[1])
    for _ in range(steps):
        r=1/np.sqrt(np.maximum(np.asarray(abs(A).max(axis=1).toarray()).ravel(),1e-300))
        A=sparse.diags(r)@A;rows*=r
        c=1/np.sqrt(np.maximum(np.asarray(abs(A).max(axis=0).toarray()).ravel(),1e-300))
        A=A@sparse.diags(c);cols*=c
    return A.tocsr(),rows,cols


def localization(x,mapping):
    a=abs(x.reshape(mapping.nr,2*mapping.H+1,4,mapping.m))**2
    total=float(a.sum());a/=max(total,1e-300)
    return dict(variables=dict(zip(('u','v','w','p'),a.sum(axis=(0,1,3)).tolist())),
                packed_harmonics=a.sum(axis=(0,2,3)).tolist(),radial=a.sum(axis=(1,2,3)).tolist(),
                z_nodes=a.sum(axis=(0,1,2)).tolist(),mean_far_w=float(a[:,0,2,-1].sum()),
                mean_pressure=float(a[:,0,3,:].sum()),
                largest_index=list(map(int,np.unravel_index(np.argmax(a),a.shape))))


def diagnose(source,out):
    out=old.output_path(out);out.mkdir(parents=True,exist_ok=False)
    rows=[];start=time.monotonic()
    for n in (8,12):
        mapping=old.base.Mapping(n,1,4)
        y=old.base.transfer(source,mapping);q=old.d2.v1.unpack(y[:-2],mapping.nr,mapping.H,mapping.s)
        temporal=old.d2.v1.Temporal(mapping,q,float(y[-1]),float(y[-2]))
        J,assembly=old.d2.assemble(temporal,old.d2.v1.Deadline(120))
        gauge=np.zeros((mapping.nr,3,4,mapping.m),complex)
        gauge[:,0,3,:]=(400/temporal.p.radii[:,None])**2;gauge=gauge.ravel();gauge/=np.linalg.norm(gauge)
        vectors=[];case=[]
        for sigma in (.05+0j,.05+2j,1.+0j):
            K=(J+sigma*sparse.diags(temporal.b)).tocsr();dense=K.toarray()
            u,s,vh=linalg.svd(dense,full_matrices=False,check_finite=False,lapack_driver='gesdd')
            v=vh[-1].conj();vectors.append(v)
            E,r,c=equilibrate(K);scaled=linalg.svdvals(E.toarray(),check_finite=False)
            record=dict(n=n,H=1,intervals=4,size=J.shape[0],sigma=old.d2.v1.zpair(sigma),
                smin=float(s[-1]),smax=float(s[0]),condition=float(s[0]/s[-1]),
                smallest_singular_values=s[-6:].tolist(),right=localization(v,mapping),left=localization(u[:,-1],mapping),
                mass_participation=float(np.linalg.norm(temporal.b*v)/max(temporal.b)),
                pressure_gauge_relative=float(np.linalg.norm(K@gauge)/s[0]),
                pressure_gauge_overlap=float(abs(np.vdot(gauge,v))),
                equilibrated_smin=float(scaled[-1]),equilibrated_smax=float(scaled[0]),
                equilibrated_condition=float(scaled[0]/scaled[-1]),
                row_scale_range=[float(r.min()),float(r.max())],column_scale_range=[float(c.min()),float(c.max())],
                stationary_field=False,source=str(source),assembly_action_error=assembly['complex_action_relative'])
            case.append(record)
            np.savez_compressed(out/f'N{n}_sigma{len(case)-1}.npz',singular_values=s,right=v,left=u[:,-1],
                                radii=temporal.p.radii,z=mapping.op.grid.z,row_scale=r,column_scale=c)
        for record,v in zip(case,vectors):record['right_overlap_with_real_shift']=float(abs(np.vdot(vectors[0],v)))
        rows.extend(case)
    result=dict(source=str(source),source_sha256=old.d2.v1.sha(source),seconds=time.monotonic()-start,cases=rows,
                sign='K=J+sigma B follows B dt q+F=0; temporal growth eigenvalues satisfy J+lambda B',
                projected_field_spectra_not_physical_stability=True)
    old.write(out/'DIAGNOSIS.json',result)
    print(json.dumps([dict(n=r['n'],sigma=r['sigma'],smin=r['smin'],condition=r['condition'],
        scaled_condition=r['equilibrated_condition'],variables=r['right']['variables'],
        far_w=r['right']['mean_far_w'],mean_p=r['right']['mean_pressure'],mass=r['mass_participation'],
        gauge_overlap=r['pressure_gauge_overlap'],overlap=r['right_overlap_with_real_shift']) for r in rows]))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();diagnose(a.source,a.out)
