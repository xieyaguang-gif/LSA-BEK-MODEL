"""Front diagnostics for the Ro=-0.751, beta=85 H4 branch (supervisor-written).

Criteria (same definitions as the Ro=-1 Phase B report):
  fig8  : downward crossing of (ln RMS_v - local growth) in R in [550, min(650, Rout-100)]
  2beta : first upward crossing of |v_2| = 1e-6 (arrival of the 2-beta harmonic) in R in (420, Rout-100)
  2beta5: first upward crossing of |v_2|/|v_1| = 0.05 in R in (500, Rout-100)
Local growth comes from the Ro=-0.751 beta=85 Phase A curve (phaseA/beta85/A3/curve_N32_global.json).
"""
import os
for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import json, sys
from pathlib import Path
import numpy as np
from scipy.interpolate import BarycentricInterpolator, PchipInterpolator
from scipy.optimize import brentq
from llsa.secondary.global_radial import radial_differences

HERE = Path(__file__).resolve().parents[1]
PHASE_B = HERE/'results/ro0751_global_20261101/phaseB'
CURVE = HERE/'results/ro0751_global_20261101/phaseA/beta85/A3/curve_N32_global.json'


def crossings(r, f, mask):
    out = []
    for i in range(len(r)-1):
        if mask[i] and mask[i+1] and f[i]*f[i+1] < 0:
            out.append(dict(R=float(r[i]-f[i]*(r[i+1]-r[i])/(f[i+1]-f[i])),
                            direction='down' if f[i] > 0 else 'up'))
    return out


def first_down(xs):
    return next((x['R'] for x in xs if x['direction'] == 'down'), None)


def first_up(xs):
    # 2-beta arrival: |v2| first reaches the threshold as R increases (inner side).
    return next((x['R'] for x in xs if x['direction'] == 'up'), None)


def diagnose(fields_path):
    d = np.load(fields_path)
    r = d['radii']; H = int(d['H']); m = len(d['xi'])
    q = d['field'].reshape(len(r), 2*H+1, 4, m)
    transform = lambda x: .6*x+.4*(x**3+.5*(1-x*x))
    xi = brentq(lambda x: transform(x)-(1.3-2)/(1.3+2), -1, 1)
    v = BarycentricInterpolator(d['xi'], q, axis=-1)(xi)[:, H+1:, 1]
    rms = np.sqrt(2*np.sum(abs(v)**2, axis=1))
    fund = abs(v[:, 0])
    D, _ = radial_differences(r)
    grms = D@np.log(np.maximum(rms, 1e-250))
    rout = float(r[-1]); a = rout-100.
    rows = json.loads(CURVE.read_text())['rows']
    lr = np.array([x['r'] for x in rows]); lg = np.array([x['growth'] for x in rows])
    localg = PchipInterpolator(lr, lg, extrapolate=False)(r)
    valid = (r >= 550) & (r <= min(650, a)) & (r >= lr.min()) & (r <= lr.max())
    fig8 = first_down(crossings(r, grms-localg, valid))
    two = first_up(crossings(r, abs(v[:, 1])-1e-6, (r > 420) & (r < a)))
    ratio = np.where(fund > 0, abs(v[:, 1])/np.maximum(fund, 1e-250), 0.)
    two5 = first_up(crossings(r, ratio-.05, (r > 500) & (r < a) & (fund > fund.max()*1e-8)))
    return dict(Rout=rout, A1=float(fund[np.argmin(abs(r-rout+100))]) if False else None,
                front_fig8=fig8, front_2beta_1e6=two, front_2beta_5pct=two5, H=H,
                highest_to_fundamental=float(np.max(abs(q[:, -1, 1])) / max(np.max(abs(q[:, H+1, 1])), 1e-300)))


def main():
    state = json.loads((PHASE_B/'campaign_h4.json').read_text())
    out = []
    for p in state['points']:
        src = HERE/'results/ro0751_global_20261101/phaseB'/'attempts'
        path = PHASE_B/'attempts'/p['id']/'result'/'fields.npz'
        if not path.exists():
            continue
        res = diagnose(path); res.update(id=p['id'], A1=p['A1'], omega=p['omega'], Rout_state=p['Rout'])
        out.append(res)
    (PHASE_B/'fronts_h4.json').write_text(json.dumps(out, indent=2))
    for x in out:
        print(f"{x['id']:>26s} Rout={x['Rout_state']:.4f} A1={x['A1']:.4f} w={x['omega']:.4f} "
              f"fig8={x['front_fig8']} 2b={x['front_2beta_1e6']} 2b5={x['front_2beta_5pct']}")


if __name__ == '__main__':
    main()
