"""Bounded pre-production spectrum coverage and saddle diagnostics."""
from fullbvp_phaseE_campaign import *


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--absolute',action='store_true')
    args=parser.parse_args()
    configs=[]
    if not args.absolute:
        base=dict(kind='temporal',R=585.,N=32,H=6,K=3,zmax=30.,beta_hat=0.,
            alpha_hat=.2,shifts=[[float(w),12.] for w in range(-80,81,20)],count=16,amplitude=1.)
        # All requested sensitivity directions, at the same radius/detuning.
        for update in ({},dict(N=48),dict(H=4),dict(K=4),dict(zmax=40.),dict(dense=True),dict(amplitude=0.)):
            configs.append(dict(base,**update))
        # Sector-forbidden channels and additional radial samples; sparse
        # high-Im shifts reduce crowding by stable near-zero-frequency modes.
        for r in (560.,585.,600.):
            for beta in (0.,17.,68/3,34.):
                for a in (-.2,-.1,.1,.2):configs.append(dict(base,R=r,beta_hat=beta,alpha_hat=a))
            for a in (-.2,-.1,.1,.2):configs.append(dict(base,R=r,alpha_hat=a,amplitude=0.))
        stage='short_audit'
    else:
        manifest=json.loads((OUT/'short_audit_manifest.json').read_text())
        for radius in (560.,585.,600.):
            candidates=[]
            for key in manifest['jobs']:
                data=json.loads((OUT/'jobs'/key/'result.json').read_text());c=data['config']
                if c['R']==radius and c['beta_hat']==0. and c['amplitude']==1. and c['N']==32 and c['H']==6 and c['K']==3 and c['zmax']==30. and not c.get('dense',False):
                    candidates.append(data)
            best=max(candidates,key=lambda d:d['modes'][0]['growth'])
            configs.append(dict(best['config'],kind='absolute',temporal_job=best['job'],branch=0,
                seed_imag=0.,pinch_steps=45,eta_max=12.))
        # Same positive-growth branch with the primary removed, and all four
        # requested discretization changes for the R585 saddle/pinch.
        for key in manifest['jobs']:
            data=json.loads((OUT/'jobs'/key/'result.json').read_text());c=data['config']
            laminar=(c['R']==560. and c['amplitude']==0. and c['alpha_hat']==.2)
            variant=(c['R']==585. and c['alpha_hat']==.2 and c['beta_hat']==0.
                and c['amplitude']==1. and (c['N']!=32 or c['H']!=6 or c['K']!=3 or c['zmax']!=30.))
            if laminar or variant:
                configs.append(dict(c,kind='absolute',temporal_job=key,branch=0,
                    seed_imag=0.,pinch_steps=45,eta_max=12.))
        stage='short_absolute'
    configs=list({job_key(c):c for c in configs}.values())
    start=time.monotonic();keys,errors=run_jobs(configs,8)
    write_json(OUT/f'{stage}_manifest.json',dict(stage=stage,jobs=sorted(keys),errors=errors,
        requested=len(configs),workers=8,threads_per_worker=1,seconds=time.monotonic()-start))
    summarize(keys,OUT/f'{stage}_summary.json')
    print(json.dumps(dict(stage=stage,done=len(keys),failed=len(errors),seconds=time.monotonic()-start)))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
