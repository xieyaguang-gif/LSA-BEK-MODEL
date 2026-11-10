"""Assemble the verified even-beta points from adaptive saddle continuation."""
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/pier2003'
sources=[
    'figure11_track_4_0.csv',
    'figure11_track_8_4.csv',
    'figure11_track_9_8.csv',
    'figure11_track_14_10.csv',
    'figure11_track_16_14.csv',
    'figure11_scan_even_low.csv',
    'figure11_scan_20_50.csv',
]
records={}
for name in sources:
    data=np.atleast_2d(np.loadtxt(OUT/name,comments='#',delimiter=','))
    for row in data:
        beta=float(row[0])
        if abs(beta/2-round(beta/2))<1e-10 and -1e-10<=beta<=50+1e-10 and bool(row[6]):
            records[int(round(beta))]=(row,name)
missing=[b for b in range(0,51,2) if b not in records]
if missing:
    raise RuntimeError(f'Missing converged even beta values: {missing}')
with (OUT/'figure11_current_even_0_50.csv').open('w') as f:
    f.write('beta_hat,alpha_real,alpha_imag,omega_real,omega_imag,abs_group_velocity,source\n')
    for beta in range(0,51,2):
        row,name=records[beta]
        f.write(','.join([f'{row[i]:.16g}' for i in range(6)]+[name])+'\n')
print(OUT/'figure11_current_even_0_50.csv')
