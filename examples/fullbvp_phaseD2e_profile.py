#!/usr/bin/env python3
"""Bounded real-operator profiling; random fields are not stability results."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','BLIS_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import json
import time
from pathlib import Path
import numpy as np
from scipy import sparse
import fullbvp_phaseD2d_precision as d2


class TimedFactor:
    def __init__(self, factor):
        self.factor, self.info = factor, factor.info
        self.seconds, self.refinements = [], []

    def solve(self, rhs):
        start = time.monotonic()
        y = self.factor.solve(rhs)
        self.seconds.append(time.monotonic()-start)
        self.refinements.append(self.info.get('last_iterative_refinement_steps'))
        return y

    def close(self):
        self.factor.close()

    def summary(self):
        return dict(phase33_calls=len(self.seconds), phase33_seconds=sum(self.seconds),
                    phase33_each_seconds=self.seconds, native_refinement_steps=self.refinements,
                    triangular_solve_count='not observable: phase33 includes native iterative refinement')


def case(n,H,intervals,out):
    out.mkdir(parents=True,exist_ok=False)
    deadline=d2.v1.Deadline(120)
    mapping=d2.v1.Mapping(n=n,H=H,intervals=intervals)
    rng=np.random.default_rng(261030)
    q=d2.v1.unpack(rng.normal(size=mapping.nr*(2*H+1)*mapping.s)*1e-4,mapping.nr,H,mapping.s)
    temporal=d2.v1.Temporal(mapping,q,720.,52.7,deadline)
    J,assembly=d2.assemble(temporal,deadline)
    sigma=.05+2j
    K=J.astype(complex)+sigma*sparse.diags(temporal.b)
    factor=TimedFactor(d2.factorize(K,'pardiso',16,deadline))
    start=time.monotonic()
    try:
        inverse=d2.PrecisionInverse(temporal.J,temporal.B,sigma,factor,deadline,out/'inverse.jsonl',assembled_K=K)
        for _ in range(3):
            inverse.T@(rng.normal(size=temporal.size)+1j*rng.normal(size=temporal.size))
        elapsed=time.monotonic()-start
        rows=[json.loads(s) for s in (out/'inverse.jsonl').read_text().splitlines()]
        result=dict(n=n,H=H,intervals=intervals,size=temporal.size,assembly=assembly,
                    factor=factor.info,timing=factor.summary(),inverse_seconds=elapsed,
                    inverse_calls=inverse.calls,direct_corrections=sum(r['direct_corrections'] for r in rows),
                    gmres_calls=sum(r['krylov_used'] for r in rows),
                    original_residual_max=max(r['original_relative'] for r in rows),
                    physical_stability_result=False,field='random nonstationary operator fixture')
    finally:
        factor.close()
    d2.v1.write(out/'profile.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    out=d2.output_path(args.out)
    if not out.is_relative_to(d2.OUT/'D2e'):
        raise ValueError('profile output must be inside D2e')
    out.mkdir(parents=True,exist_ok=False)
    rows=[case(n,h,r,out/f'N{n}_H{h}_R{r}') for n,h,r in [(12,2,16),(16,2,16),(12,3,16),(12,2,32)]]
    d2.v1.write(out/'PROFILE.json',rows)
    print(json.dumps([dict(n=r['n'],H=r['H'],intervals=r['intervals'],size=r['size'],
          assembly=r['assembly']['assembly_seconds'],analysis=r['factor']['analysis_seconds'],
          factor=r['factor']['factorization_seconds'],nnz=r['factor']['factor_nnz'],
          solve=sum(r['timing']['phase33_each_seconds'])/r['timing']['phase33_calls'],
          residual=r['original_residual_max']) for r in rows]))


if __name__=='__main__':main()
