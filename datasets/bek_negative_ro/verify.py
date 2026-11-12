#!/usr/bin/env python3
"""Check the released files and key numerical identities without any solves."""
from pathlib import Path
import csv, hashlib, json
import numpy as np
P = Path(__file__).resolve().parent
manifest = json.loads((P/'manifest.json').read_text())
for name, record in manifest['files'].items():
    assert hashlib.sha256((P/name).read_bytes()).hexdigest() == record['sha256'], name
rows = list(csv.DictReader((P/'scalar_results.csv').open()))
assert len(rows) == 11 and sum(r['case_type']=='survey' for r in rows)==9
for r in rows:
    v = lambda k: float(r[k])
    assert np.isclose(v('Re_ca'),abs(v('Ro'))*v('R_ca'),rtol=1e-13)
    assert v('R_ca') < v('Rf_NL') < v('Rf_L')
    assert np.isclose(v('subcritical_width_N32'),v('Rout_neutral_branch_N32')-v('Rout_fold_N32'))
assert len(list(csv.DictReader((P/'branches.csv').open()))) == 224
checks = list(csv.DictReader((P/'resolution_checks.csv').open()))
assert len(checks)==8 and len({r['table_row'] for r in checks})==7
for r in checks:
    a,b=float(r['value_before']),float(r['value_after'])
    assert np.isclose(float(r['absolute_change']),abs(b-a),rtol=1e-12,atol=1e-14)
for folder, filename in [('wave_structure','projections.npz'),('damping_response','profiles.npz')]:
    q=P/'data'/folder/filename
    assert hashlib.sha256(q.read_bytes()).hexdigest()==json.loads((q.parent/'metadata.json').read_text())['array_sha256']
    with np.load(q,allow_pickle=False) as a:
        assert all(np.issubdtype(a[k].dtype,np.number) and np.isfinite(a[k]).all() for k in a.files)
print('Verified 11 cases, 224 branch points, 7 Table II rows and all file checksums.')
