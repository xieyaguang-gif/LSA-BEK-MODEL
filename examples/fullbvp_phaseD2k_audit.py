#!/usr/bin/env python3
"""Read existing 720/725 fields and certificates; never assemble a large Jacobian."""
import time
from pathlib import Path
import numpy as np
import fullbvp_phaseD2k_campaign as k


def main():
    started=time.monotonic();rows=[];points=[]
    for R in (720,725):
        seed=k.seed_bundle(R);spec=k.VARIANTS['baseline'];mapping=k.mesh(spec,R)
        y=k.transfer(seed,mapping)
        temporal=k.temporal_api.Temporal(mapping,k.base.unpack(y[:-2],mapping.nr,mapping.H,mapping.s),y[-1],y[-2],k.temporal_api.Deadline(20))
        stationarity=k.old.stationarity_gate(temporal)
        out=k.PREP/'baseline_audit'/str(R);out.mkdir(parents=True,exist_ok=True)
        physical=k.point_diagnostics(mapping,y,seed,out)
        point=Path(seed['fields']).parents[2];centers=[]
        for marker in sorted(point.glob('center_*/DONE.json')):
            center=k.read(marker)
            if center.get('complete') and all(p['accepted'] and p['original_relative']<=1e-8 for p in center['eigenpairs']):
                arrays=Path(center['arrays'])
                if not arrays.is_absolute():arrays=k.ROOT/arrays
                if k.sha(arrays)!=center['arrays_sha256']:raise RuntimeError('prior spectrum arrays changed')
                centers.append(center)
        pairs=k.candidate_pairs(centers)
        spectral=out/'spectrum';spectral.mkdir(exist_ok=True)
        for idx,pair in enumerate(pairs):
            with np.load(pair['arrays']) as data:vector=data['vectors'][:,pair['index']]
            certificate=k.temporal_api.certify(temporal.J,temporal.B,max(temporal.b),complex(*pair['value']),vector,tol=1e-8,phase=temporal.neutral())
            if not certificate['accepted']:raise RuntimeError('prior pair failed present original-operator recertification')
            pair['recertified']=certificate
            pair['localization'],rem=k.localize_vector(temporal,vector)
            path=spectral/f'mode_{idx}.npz'
            np.savez_compressed(path,field=rem,radii=temporal.p.radii,xi=mapping.op.grid.xi,H=mapping.H,a=2.)
            pair.update(shape=str(path),shape_sha256=k.sha(path))
        k.write(spectral/'spectrum.json',dict(pairs=pairs,historical_evidence_only=True))
        points.append(dict(Rout=R,directory=str(out),physical_diagnostics=physical,diagnosed=True,
                           singular_diagnostics=[],fields=seed['fields'],fields_sha256=seed['sha256'][seed['fields']]))
        rows.append(dict(Rout=R,seed=seed,stationarity=stationarity,physical=physical,
            prior_centers=len(centers),prior_pairs=pairs,
            spectrum_status='prior pairs re-certified with current original operator and localized; no new full eigenproblem solved'))
    k.write(k.PREP/'baseline_audit.json',dict(points=rows,seconds=time.monotonic()-started,production_started=False))
    k.write(k.PREP/'baseline_audit/baseline/state.json',dict(status='historical_evidence_only',points=points))
    k.summarize(k.PREP/'baseline_audit',plot=True)
    for p in rows:
        print('Rout=%s omega=%.10f front_fig8=%s max_qv1_z1p3=%.8g prior_centers=%s' %
              (p['Rout'],p['physical']['omega'],p['physical']['front_fig8'],p['physical']['main_wave_z1p3_max_before_sponge'],p['prior_centers']))


if __name__=='__main__':main()
