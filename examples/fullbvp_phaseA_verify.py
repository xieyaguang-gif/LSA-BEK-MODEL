#!/usr/bin/env python3
"""Final independent result/file consistency checks and PDF preview rendering."""
import json,re,subprocess,sys,os,ctypes,argparse
from pathlib import Path
import numpy as np
from PIL import Image
from fullbvp_phaseA_global import OUT,dump

def main():
    summary=json.loads((OUT/'summary.json').read_text());report=(OUT/'REPORT.md').read_text()
    expected=[700,710,715,720,725,730,740]
    assert [x['rout'] for x in summary['A1']]==expected
    assert all(x['residual']<1e-7 and x['correlation']>=.85 for x in summary['A1']+summary['neutral'])
    assert all(abs(x['sigma'])<2e-8 for x in summary['neutral'])
    assert all(x['correlation']>=.85 for x in summary['validation']['neutral_cross_grid_correlations'])
    assert summary['A3']['nominal_r_end']['unique_intersection']
    for p in summary['A3']['pinch_summary']:
        assert p['opposite_half_planes_at_end'] and p['max_residual']<1e-8
    for x in summary['A1']:
        assert f"{x['omega']:.6f}" in report and f"{x['sigma']:.7f}" in report
    controls=summary['A4']['rows'];assert len(controls)==2
    assert abs(controls[0]['dr_actual']-controls[1]['dr_actual'])<1e-12
    assert controls[0]['a']==controls[1]['a']==620.
    preview=Path('/tmp/fullbvp_phaseA_visual_20261030');preview.mkdir(exist_ok=True)
    figures=[]
    for name in ['sigma_Rout','growth_intersection']:
        png=OUT/f'{name}.png';pdf=OUT/f'{name}.pdf'
        assert png.exists() and pdf.exists()
        with Image.open(png) as im:assert min(im.size)>900;size=list(im.size)
        target=preview/f'{name}_pdf'
        subprocess.run(['pdftoppm','-png','-r','120','-singlefile',str(pdf),str(target)],check=True,capture_output=True)
        figures.append(dict(name=name,png=str(png),pdf=str(pdf),png_size=size,pdf_render=str(target)+'.png'))
    statuses=[line.split('\t') for line in (OUT/'worker_status.tsv').read_text().splitlines()]
    assert all(row[2]=='0' for row in statuses)
    assert len(statuses)==8
    result=dict(passed=True,seven_requested_points=expected,neutral_grids=4,workers=len(statuses),
        figure_artifacts=figures,visual_review_pending=True,nominal_r_end=summary['A3']['nominal_r_end'])
    dump(OUT/'validation.json',result)
    print(json.dumps(result),flush=True)

def await_numerical():
    # A local event-driven background waiter, never a model status-polling loop.
    libc=ctypes.CDLL(None,use_errno=True);fd=libc.inotify_init1(0)
    if fd<0:raise OSError(ctypes.get_errno(),'inotify_init')
    libc.inotify_add_watch(fd,str(OUT).encode(),0x00000100|0x00000008|0x00000080)
    while not (OUT/'NUMERICAL_DONE').exists():
        if (OUT/'FAILED').exists():raise RuntimeError((OUT/'FAILED').read_text())
        os.read(fd,65536)
    os.close(fd)
    subprocess.run([sys.executable,'examples/fullbvp_phaseA_report.py'],check=True)
    main()
    (OUT/'REVIEW_READY').write_text('Latest report, spectral refinement, and validation completed; manual PNG/PDF review remains.\n')

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--await-numerical',action='store_true');args=parser.parse_args()
    await_numerical() if args.await_numerical else main()
