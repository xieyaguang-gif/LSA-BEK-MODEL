"""Locate the residual in the original physical equations of a saved field."""
import argparse
import json
from pathlib import Path

import numpy as np

from continue_radial_sponge import load_seed, write_json
from llsa.secondary.global_radial import GlobalRadial, OutletSponge, BlendedSponge
from llsa.secondary.radial import pack_field


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('case', type=Path)
    args = parser.parse_args()
    report = json.loads((args.case/'metadata.json').read_text())
    settings = json.loads((args.case.parent/'summary.json').read_text())['arguments']
    path = args.case/'fields.npz'
    if not path.exists():
        path = args.case/'checkpoint.npz'
    config, r, q = load_seed(path)
    _, _, original = load_seed(Path(settings['seed']),config.n,config.harmonics)
    sponge = OutletSponge(report['sponge_start']/abs(config.ro),
            settings['width']/abs(config.ro),settings['strength'],settings['damp_mean'])
    if 'release_fraction' in report:
        old = OutletSponge(settings['old_start']/abs(config.ro),sponge.width,sponge.strength,sponge.damp_mean)
        new = OutletSponge(settings['target']/abs(config.ro),sponge.width,sponge.strength,sponge.damp_mean)
        sponge = BlendedSponge(old,new,report['release_fraction'])
    problem = GlobalRadial(r, config, outlet_alpha=complex(settings['outlet_alpha']),outlet_sponge=sponge)
    residual = problem.evaluate(pack_field(q),original[0])/report['residual_scale']
    field = abs(residual.reshape(len(r),2*config.harmonics+1,4,config.n+1))
    peaks = []
    for component,label in enumerate(['radial momentum','azimuthal momentum','axial momentum','continuity / pressure closure']):
        values = field[:,:,component,:]
        i,part,z = np.unravel_index(np.argmax(values),values.shape)
        peaks.append(dict(equation=label,residual=float(values[i,part,z]),Re=float(abs(config.ro)*r[i]),
            harmonic=int((part+1)//2),part='real' if part == 0 or part%2 else 'imaginary',
            z=float(problem.ops[0].grid.z[z]),radial_endpoint=bool(i in (0,len(r)-1)),
            normal_endpoint=bool(z in (0,config.n))))
    output = dict(source=str(path),source_status=report['status'],
        release_fraction=report.get('release_fraction'),
        original_equation_residual=float(np.max(field)),peaks=peaks,
        note='Snapshot diagnostic; an unconverged checkpoint is not a physical solution.')
    write_json(args.case/'residual_diagnostic.json',output)
    print(json.dumps(output,indent=2))


if __name__ == '__main__':
    main()
