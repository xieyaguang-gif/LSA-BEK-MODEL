#!/usr/bin/env python3
"""Reproduce D4 provenance and growth comparisons without running a solver.

Writes only phaseD/D4. Input fields are read-only; no phaseC2 or other D task
is traversed. Python: system python3 (numpy, scipy, PyMuPDF installed).
"""
from __future__ import annotations

import os
for _key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
             'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_key] = '1'

import csv
import hashlib
import json
import subprocess
import time
from pathlib import Path

import numpy as np
import pymupdf as fitz
from scipy.interpolate import BarycentricInterpolator
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/fullbvp_global_20261030/phaseD/D4'
BASE = ROOT/'results/fullbvp_global_20261030'
LIT = ROOT/'results/appelquist2016_branch_20261029/literature'
REF = BASE/'phaseB/reference_data'
WINDOWS = [(550., 570.), (560., 580.), (570., 600.), (580., 600.)]
MANIFEST: dict[str, dict] = {}


def track(path):
    p = Path(path).resolve()
    assert 'phaseC2' not in p.parts
    if 'phaseD' in p.parts:
        assert p.is_relative_to(OUT)
    if str(p) not in MANIFEST:
        MANIFEST[str(p)] = {'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                            'bytes': p.stat().st_size}
    return p


def read(path):
    return json.loads(track(path).read_text())


def write(name, obj):
    (OUT/name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def csvwrite(name, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with (OUT/name).open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def coeff_at_z13(xi, values):
    transform = lambda x: .6*x+.4*(x**3+.5*(1-x*x))
    target = brentq(lambda x: transform(x)-(1.3-2)/(1.3+2), -1, 1)
    return BarycentricInterpolator(xi, values, axis=-1)(target)


def window_stats(r, logamp, lower, upper):
    """Endpoint interpolation is in ln(A); OLS uses actual interior vertices."""
    assert len(r) == len(logamp) and np.all(np.diff(r) > 0)
    assert lower >= r[0] and upper <= r[-1]
    mask = (r >= lower) & (r <= upper)
    rr = r[mask]
    yy = logamp[mask]
    assert len(rr) >= 1
    slope, residual = None, None
    if len(rr) >= 2:
        slope, intercept = np.polyfit(rr, yy, 1)
        residual = float(np.sqrt(np.mean((yy-slope*rr-intercept)**2)))
    endpoint = (np.interp(upper, r, logamp)-np.interp(lower, r, logamp))/(upper-lower)
    # Trapezoidal approximation to the uniform-R OLS integrals. A fixed grid
    # is declared as numerical quadrature, never independent DNS samples.
    quadrature = np.linspace(lower, upper, 1201)
    vals = np.interp(quadrature, r, logamp)
    centered = quadrature-(lower+upper)/2
    uniform_fit = np.trapezoid(centered*vals, quadrature)/np.trapezoid(centered**2, quadrature)
    return {'window_lower': lower, 'window_upper': upper,
            'n_actual_interior': int(len(rr)), 'first_actual_R': float(rr[0]),
            'last_actual_R': float(rr[-1]), 'endpoint_ln_slope': float(endpoint),
            'native_vertex_OLS_ln_slope': None if slope is None else float(slope),
            'native_vertex_OLS_residual_rms': residual,
            'uniform_R_OLS_ln_slope': float(uniform_fit),
            'uniform_R_quadrature_nodes': len(quadrature),
            'energy_ln_endpoint': float(2*endpoint),
            'energy_proxy_definition': 'A_squared_of_same_scalar_diagnostic; not full height-integrated kinetic energy',
            'log10_amplitude_endpoint': float(endpoint/np.log(10)),
            'physical_velocity_endpoint': float(endpoint+np.log(upper/lower)/(upper-lower))}


def vertices(drawing):
    points = []
    for item in drawing['items']:
        assert item[0] == 'l', 'Unexpected non-linear source vector segment'
        if not points:
            points.append(tuple(item[1]))
        points.append(tuple(item[2]))
    return np.asarray(points)


def fig8(doc):
    page = doc[13]
    drawings = page.get_drawings()
    # Actual tick strokes, not text bounding-box centers (which are offset).
    xpt = np.array([drawings[i]['rect'].x0 for i in (114, 116, 118)])
    xval = np.array([550., 600., 650.])
    ypt = np.array([drawings[i]['rect'].y0 for i in (120, 122, 124, 126, 128, 130, 132)])
    yval = np.arange(7)*.05
    xc = np.polyfit(xpt, xval, 1)
    yc = np.polyfit(ypt, yval, 1)
    assert np.max(abs(np.polyval(xc,xpt)-xval)) < 1e-4
    assert np.max(abs(np.polyval(yc,ypt)-yval)) < 5e-5
    definitions = [('L01', 2, (0., .5019454956054688, 0.)),
                   ('NL01', 12, (1., 0., 1.)),
                   ('NL02', 28, (0., .7490196228027344, .7490196228027344)),
                   ('NL03', 36, (1., 0., 0.))]
    results, rows, curves, paths = [], [], {}, []
    for case, index, color in definitions:
        drawing = drawings[index]
        assert np.allclose(drawing['color'], color)
        xy = vertices(drawing)
        rr, gg = np.polyval(xc, xy[:, 0]), np.polyval(yc, xy[:, 1])
        # Figure-frame clipping only. Keep actual in-frame path vertices.
        keep = (gg >= 0) & (gg <= .3)
        rr, gg, xy = rr[keep], gg[keep], xy[keep]
        assert np.all(np.diff(rr) > 0)
        curves[case] = (rr, gg)
        for r, g, pt in zip(rr, gg, xy):
            rows.append({'case': case, 'T': 6. if case != 'L01' else None,
                         'R': float(r), 'g_ln_RMS': float(g),
                         'pdf_x_pt': float(pt[0]), 'pdf_y_pt': float(pt[1]),
                         'path_index_zero_based': index})
        paths.append({'case': case, 'path_index_zero_based': index,
                      'stroke_rgb': list(color), 'retained_vertices': len(rr),
                      'visible_R_range': [float(rr[0]), float(rr[-1])],
                      'legend_identity': case+' source legend and same-colored markers',
                      'retained_mask': keep.tolist()})
        for a, b in WINDOWS:
            if a < rr[0] or b > rr[-1]:
                continue
            exact_nodes = np.r_[a, rr[(rr > a) & (rr < b)], b]
            exact_g = np.interp(exact_nodes, rr, gg)
            mean_g = np.trapezoid(exact_g, exact_nodes)/(b-a)
            results.append({'dataset': 'DNS_Fig8' if case != 'L01' else 'L01_Fig8',
                            'case': case, 'status': 'new_vector_digitization',
                            'T': 6. if case != 'L01' else None,
                            'z': 1.3, 'variable': 'total_azimuthal_RMS',
                            'normalization': 'local_disk_speed', 'log_base': 'ln',
                            'method': 'window_average_of_published_local_derivative',
                            'window_lower': a, 'window_upper': b,
                            'value': float(mean_g), 'n_actual_interior': int(np.sum((rr >= a)&(rr <= b))),
                            'source': 'source PDF page14/printed345 Fig8',
                            'omega_lab': None,
                            'omega_note': 'DNS time dependent; coherent spectrum is not a monochromatic derivative fit'})
        # Mark sparse original vertices directly on the source page for identity checks.
        for pt in xy[::12]:
            page.draw_circle(fitz.Point(*pt), 1.25, color=(0., 0., 0.), width=.4)
    page.get_pixmap(matrix=fitz.Matrix(2, 2)).save(OUT/'fig8_source_overlay.png')
    csvwrite('fig8_digitized.csv', rows)
    provenance = {'pdf_page_one_based': 14, 'printed_page': 345, 'figure': 8,
                  'x_tick_pdf_points': xpt.tolist(), 'x_tick_R': xval.tolist(),
                  'x_calibration_slope_intercept': xc.tolist(),
                  'max_x_anchor_residual_R': float(np.max(abs(np.polyval(xc,xpt)-xval))),
                  'y_tick_pdf_points': ypt.tolist(), 'y_tick_growth': yval.tolist(),
                  'y_calibration_slope_intercept': yc.tolist(),
                  'max_y_anchor_residual_growth': float(np.max(abs(np.polyval(yc,ypt)-yval))),
                  'paths': paths,
                  'uncertainty': {'coordinate_reading_tolerance_pt': .5,
                                  'R_tolerance': float(abs(xc[0])*.5),
                                  'growth_vertical_tolerance': float(abs(yc[0])*.5),
                                  'type': 'graphical envelope, not DNS/statistical uncertainty'},
                  'definition': 'published d[ln(vrms)]/dr; no unpublished differentiation or smoothing metadata recovered',
                  'clipping': 'retain only actual vector vertices within y-axis 0..0.3; no radial extrapolation',
                  'overlay': 'fig8_source_overlay.png; black circles mark extracted vertices',
                  'grey_line_warning': 'theory at complex omega_disk=-15.56+0.568i or -13.78+0.981i is not extracted as DNS or project k-minus'}
    write('fig8_provenance.json', provenance)
    return curves, results


def intersections(drawing, yy):
    xs = []
    for item in drawing['items']:
        assert item[0] == 'l'
        p, q = item[1], item[2]
        if abs(q.y-p.y) < 1e-12:
            continue
        if min(p.y, q.y)-1e-9 <= yy <= max(p.y, q.y)+1e-9:
            xs.append(p.x+(q.x-p.x)*(yy-p.y)/(q.y-p.y))
    return sorted(set(round(x, 9) for x in xs))


def fig6_times(doc, nl01_prov):
    page = doc[11]
    drawings = page.get_drawings()
    # NL01: independently verified existing calibration. NL03: actual panel(e)
    # tick strokes, with T6 at y334.631 and T0 at y406.084.
    top_ticks = sorted(set(round(d['rect'].x0, 6) for d in drawings
          if len(d['items']) == 1 and d['items'][0][0] == 'l' and
          abs(d['rect'].y0-334.6310119628906) < .002 and
          d['rect'].width < .001 and 1.2 < d['rect'].height < 1.4 and
          180 < d['rect'].x0 < 306))
    assert len(top_ticks) == 7, top_ticks
    nl03_xc = np.polyfit(top_ticks, np.arange(450.,751.,50.),1).tolist()
    cases = [('NL01', range(50,56), nl01_prov['calibration']['R_of_x_slope_intercept'],
              82.1820068359375, 14.884002685546875),
             ('NL03', range(11257,11263),
              nl03_xc,
              334.6310119628906, (406.0840148925781-334.6310119628906)/6)]
    rows, audit = [], []
    colors = [(0.,.5019454956054688,1.),(0.,1.,1.),
              (.5019454956054688,1.,.5019454956054688),
              (1.,1.,0.),(1.,.5019454956054688,0.),(1.,0.,0.)]
    for case, indices, xc, y6, dt in cases:
        for index,color in zip(indices,colors):
            assert np.allclose(drawings[index]['fill'],color), (case,index)
        for T in (3.,4.,5.,6.):
            yy = y6+(6-T)*dt
            for level, idx in zip(range(-7,-1), indices):
                xs = intersections(drawings[idx], yy)
                rr = [float(np.polyval(xc,x)) for x in xs]
                valid = [r for r in rr if 450 < r < 698]
                assert valid, (case,T,level,xs)
                # Upstream-most contour boundary; remove downstream closure only.
                r = min(valid)
                rows.append({'case': case, 'T': T, 'z': 1.3,
                             'log10_RMS_level': level, 'R': r,
                             'R_graphical_envelope': 2.5,
                             'definition': 'upstream_fixed_RMS_isocontour; not Fig8 front',
                             'pdf_path_index_zero_based': idx})
                audit.append({'case':case,'T':T,'level':level,'y_pt':yy,
                              'path_index_zero_based':idx,'all_crossings_R':rr,
                              'selected_R':r})
                xx = (r-xc[1])/xc[0]
                page.draw_circle(fitz.Point(xx,yy),.8,color=(0.,0.,0.),width=.35)
    page.get_pixmap(matrix=fitz.Matrix(2,2)).save(OUT/'fig6_time_source_overlay.png')
    csvwrite('dns_fixed_contour_times.csv', rows)
    write('fig6_time_provenance.json', {'pdf_page_one_based':12,'printed_page':343,
          'panels':{'NL01':'b','NL03':'e'},
          'calibrations':[{'case':case,'R_of_x':xc,'y_T6':y6,'pt_per_rotation':dt}
                          for case,_,xc,y6,dt in cases],
          'level_identity':'same verified six fill RGB levels as NL01 existing extraction; -7 through -2',
          'all_intersections':audit,
          'R_tolerance':2.5,'type':'graphical envelope, not statistical uncertainty',
          'use':'independent finite-time motion diagnostic; not the EqFig8 front and not a front velocity',
          'overlay':'fig6_time_source_overlay.png'})
    return rows


def field_profile(path, meta, linear=False):
    with np.load(track(path)) as d:
        r, xi = d['radii'].copy(), d['xi'].copy()
        if linear:
            v = coeff_at_z13(xi,d['q'])[:,1,None]
            H = 1
            omega = float(d['omega_c'])
        else:
            H = int(d['H'])
            q = d['field'].reshape(len(r),2*H+1,4,len(xi))
            v = coeff_at_z13(xi,q)[:,H+1:,1]
            omega = float(d['omega'])
        fund = abs(v[:,0])
        rms = np.sqrt(2*np.sum(abs(v)**2,axis=1))
    assert np.all(np.isfinite(v)) and np.all(np.diff(r)>0)
    theta = 2*np.pi*np.arange(2*H+3)/(2*H+3)
    physical = 2*np.real(np.einsum('rj,jt->rt',v,np.exp(1j*np.arange(1,H+1)[:,None]*theta)))
    parseval = float(np.max(abs(np.sqrt(np.mean(physical**2,axis=1))-rms))/rms.max())
    assert parseval < 1e-12
    return dict(meta, r=r, fund=fund, rms=rms, H=H, n=len(xi)-1,
                intervals=len(r)-1, omega_lab=omega, source=str(Path(path).resolve()),
                parseval_relative_error=parseval)


def source_index(pdf_path):
    lines=[]
    # Source text remains an extract in the requested local evidence directory.
    text=subprocess.run(['pdftotext','-layout',str(pdf_path),'-'],capture_output=True,text=True,check=True).stdout
    for i,page in enumerate(text.split('\f')):
        if page.strip():
            lines.append(f'=== PDF PAGE {i+1}; PRINTED {332+i} ===')
            lines.extend(page.splitlines())
    (OUT/'source_paper_numbered.txt').write_text('\n'.join(f'{i+1}: {s}' for i,s in enumerate(lines))+'\n')
    index = [
      ('paper_vrms_definition','source_paper_numbered.txt','PDF11/printed342 Eq3.1','z1.3; azimuthal mean removed; local disk speed normalization'),
      ('paper_Fig8','source_paper_numbered.txt','lines770-800,838-864; PDF14-15/printed345-346','final-time radial derivative; fronts still moving'),
      ('paper_Fig9','source_paper_numbered.txt','lines829-864; PDF15/printed346','T3-6; 582.8 is inferred asymptotic position'),
      ('paper_Fig10','source_paper_numbered.txt','lines895-919; PDF16/printed347','NL01 front589; NL02/NL03 front570 at T6'),
      ('paper_Fig15','source_paper_numbered.txt','PDF21/printed352 Fig15','NL03 T6 versus separate estimated infinite-time curve'),
      ('historical_DNS_0.106','examples/fullbvp_phaseA_spatial.py','lines226-227','explicit user supplied approximate DNS comparison'),
      ('historical_reference_spread','examples/fullbvp_phaseB_report.py','line134','subsequently labeled DNS graphical approximate without a fit source'),
      ('legacy_BVP_growth','examples/fullbvp_phaseB_report.py','lines24-41','fundamental endpoint and OLS; RMS endpoint separate'),
      ('legacy_PhaseC_growth','examples/fullbvp_phaseC_postprocess.py','lines25-26,47-50','NL03 and NL01 source paths; preserves endpoint/OLS distinction'),
    ]
    csvwrite('source_index.csv',[dict(id=a,path=b,location=c,evidence=d) for a,b,c,d in index])


def main():
    start=time.monotonic()
    OUT.mkdir(parents=True,exist_ok=True)
    for path in [Path(__file__),ROOT/'examples/fullbvp_phaseA_spatial.py',
                 ROOT/'examples/fullbvp_phaseB_report.py',ROOT/'examples/fullbvp_phaseC_postprocess.py']:
        track(path)
    reference=read(LIT/'reference.json')
    pdf_path=track(reference['paper']['pdf_path'])
    assert MANIFEST[str(pdf_path)]['sha256']==reference['paper']['pdf_sha256']
    source_index(pdf_path)
    nl01p=read(REF/'nl01_T6_provenance.json')
    nl03p=read(LIT/'fig15_digitization.json')
    doc=fitz.open(pdf_path)
    doc[13].get_pixmap(matrix=fitz.Matrix(1.5,1.5)).save(OUT/'source_fig8_page345.png')
    fig8_curves,comparison=fig8(doc)
    contours=fig6_times(doc,nl01p)
    old_nl01=np.genfromtxt(track(REF/'nl01_T6_vrms.csv'),delimiter=',',names=True)
    nl01_delta=max(abs(next(p['R'] for p in contours if p['case']=='NL01' and p['T']==6. and
                      p['log10_RMS_level']==int(q['log10_vrms']))-q['R']) for q in old_nl01)
    assert nl01_delta<.001
    nl03_delta=max(abs(next(p['R'] for p in contours if p['case']=='NL03' and p['T']==6. and
                      p['log10_RMS_level']==int(q['log10_vrms']))-q['fig6e_R'])
                      for q in nl01p['source_figure_consistency']['checks'])
    assert nl03_delta<.03
    dns_profiles={}
    for case,path in [('NL03',LIT/'fig15_nl03_T6_vrms.csv'),('NL01',REF/'nl01_T6_vrms.csv')]:
        d=np.genfromtxt(track(path),delimiter=',',names=True)
        r,y=d['R'],np.log(d['vrms_normalized'])
        assert np.all(np.diff(r)>0)
        assert np.max(abs(y/np.log(10)-d['log10_vrms']))<1e-12
        dns_profiles[case]=(r,y)
        for a,b in WINDOWS:
            if not (r[0]<=a<b<=r[-1]):continue
            stats=window_stats(r,y,a,b)
            common={'dataset':'DNS_Fig15' if case=='NL03' else 'DNS_Fig6_sparse',
                    'case':case,'T':6.,'z':1.3,'variable':'total_azimuthal_RMS',
                    'normalization':'local_disk_speed','log_base':'ln',
                    'status':'recomputed_from_existing_digitization',
                    'source':str(path),'n_source_vertices':len(r),'omega_lab':52.36 if case=='NL03' else None,
                    'omega_note':'NL03 coherent spectral peak; full RMS remains time dependent',
                    **stats}
            if case=='NL03':
                dx=nl03p['uncertainty']['R_tolerance'];dy=nl03p['uncertainty']['log10_vrms_tolerance']*np.log(10)
                common['endpoint_graphical_lower']=(np.interp(b-dx,r,y)-np.interp(a+dx,r,y)-2*dy)/(b-a)
                common['endpoint_graphical_upper']=(np.interp(b+dx,r,y)-np.interp(a-dx,r,y)+2*dy)/(b-a)
            comparison.append(common)

    accepted=read(BASE/'phaseC/accepted_points.json')
    campaign=read(BASE/'phaseC/campaign.json')
    a3=read(BASE/'phaseA/A3/summary.json')
    profiles=[]
    for p in accepted['points']:
        profiles.append(field_profile(p['fields'],{'id':p['id'],'dataset':'phaseC',
            'Rout':p['Rout'],'acceptance':'campaign_accepted','front_fig8_existing':p['front_fig8'],
            'legacy_g_q_endpoint':p['g_q_570_600'],'legacy_g_q_OLS':p['g_q_least_squares_570_600']}))
    for p in sorted((BASE/'phaseC/attempts').glob('*/summary.json')):
        # Only accepted harmonic audits and the campaign's uncommitted 730 fields.
        if not (p.parent.name.startswith('harmonic_audit_') or p.parent.name.startswith('source_H4_lift_') or
                p.parent.name in ['Rout_step_9574cb229e65646e_try0','Rout_step_294f7d0ba7ea1617_try0']):continue
        s=read(p)
        if s.get('accepted'):
            status='strict_auxiliary_audit' if s['Rout']<729 else 'Newton_accepted_campaign_uncommitted_H6_failed'
            diagnostic=BASE/'phaseC/diagnostics'/f'{p.parent.name}.json'
            existing=read(diagnostic) if diagnostic.exists() else {}
            profiles.append(field_profile(p.parent/'fields.npz',{'id':p.parent.name,
                'dataset':'phaseC' if s['Rout']<729 else 'phaseC_uncommitted',
                'Rout':s['Rout'],'acceptance':status,
                'front_fig8_existing':existing.get('front_fig8'),
                'front_existing_diagnostic_source':str(diagnostic) if diagnostic.exists() else None,
                'source_summary':str(p),'independent_relative_residual':s['independent']['relative_field_inf']}))
    for grid in a3['grids']:
        profiles.append(field_profile(grid['source'],{'id':Path(grid['source']).parent.name,
            'dataset':'phaseA','Rout':None,'acceptance':'linear_neutral_existing',
            'front_fig8_existing':grid['rend_global_v13'][0]},linear=True))
        energy_path=BASE/'phaseA/A3'/f'global_growth_{Path(grid["source"]).parent.name}.npz'
        with np.load(track(energy_path)) as data:
            er=data['r']; le=np.log(np.maximum(data['energy'],1e-300))
        for a,b in WINDOWS:
            comparison.append({'dataset':'phaseA_height_integrated_energy',
                'case':Path(grid['source']).parent.name,'T':None,'z':'height_integrated',
                'variable':'sqrt_integral_all_velocity_components_abs_q_squared_dz',
                'normalization':'q=physical_velocity/local_disk_speed','log_base':'ln',
                'status':'recomputed_from_existing_energy_profile; no new mass integration',
                'omega_lab':grid['omega'],'source':str(energy_path),
                **window_stats(er,.5*le,a,b),
                'energy_proxy_definition':'height_integrated_E=integral_sum_velocity_abs_q_squared_dz; amplitude=sqrt(E)'})
    summaries,local_rows,validation=[],[],[]
    for p in profiles:
        for variable in ('fund','rms'):
            la=np.log(np.maximum(p[variable],1e-250))
            for a,b in WINDOWS:
                stats=window_stats(p['r'],la,a,b)
                comparison.append({'dataset':p['dataset'],'case':p['id'],'T':None,'z':1.3,
                    'variable':'fundamental_azimuthal_complex_modulus' if variable=='fund' else 'coherent_total_azimuthal_RMS',
                    'normalization':'local_disk_speed','log_base':'ln','status':'recomputed_from_actual_field',
                    'acceptance':p['acceptance'],'Rout':p['Rout'],'H':p['H'],'n':p['n'],'intervals':p['intervals'],
                    'omega_lab':p['omega_lab'],'source':p['source'],**stats})
                if a==570 and b==600:
                    prefix='gq' if variable=='fund' else 'gRMS'
                    p[prefix+'_endpoint']=stats['endpoint_ln_slope']
                    p[prefix+'_OLS']=stats['native_vertex_OLS_ln_slope']
                    p[prefix+'_uniform_R_OLS']=stats['uniform_R_OLS_ln_slope']
                    p['n_570_600']=stats['n_actual_interior']
                if a==550 and b==570:p[('gq' if variable=='fund' else 'gRMS')+'_550_570']=stats['endpoint_ln_slope']
            g=np.gradient(la,p['r'],edge_order=2)
            for R in (550.,570.,580.,585.,590.,600.):
                local_rows.append({'dataset':p['dataset'],'case':p['id'],'variable':variable,'R':R,
                    'g_ln_amplitude':float(np.interp(R,p['r'],g)),
                    'method':'numpy.gradient_2nd_order_then_linear_R_interpolation',
                    'omega_lab':p['omega_lab'],'H':p['H'],'n':p['n'],'acceptance':p['acceptance']})
        if 'legacy_g_q_endpoint' in p:
            err=abs(p['gq_endpoint']-p['legacy_g_q_endpoint'])
            fiterr=abs(p['gq_OLS']-p['legacy_g_q_OLS'])
            assert err<2e-11 and fiterr<2e-11
            validation.append({'case':p['id'],'legacy_endpoint_abs_difference':err,'legacy_OLS_abs_difference':fiterr})
        p['RMS_minus_fundamental_endpoint']=p['gRMS_endpoint']-p['gq_endpoint']
        p['physical_normalization_addition_570_600']=np.log(600/570)/30
        summary={k:v for k,v in p.items() if k not in ('r','fund','rms')}
        summaries.append(summary)
        csvwrite(p['id']+'_z13_profile.csv',[{'R':float(r),'abs_qv1':float(f),'coherent_RMS':float(s)}
                 for r,f,s in zip(p['r'],p['fund'],p['rms']) if 540<=r<=610])
    # Existing local frozen branches retain their own eigenfrequency and operator.
    for p in a3['spatial_at_global_frequency']:
        branch=next(x for x in p['branches'] if x['label']=='k_minus')
        local_rows.append({'dataset':'local_k_minus','case':p['kind'],'R':p['r'],
            'variable':'local_spatial_eigenmode_amplitude','g_ln_amplitude':-branch['alpha'][1],
            'method':'minus_Im_alpha_existing_mode; no new eigensolve',
            'omega_lab':p['omega_global'][0],'omega_lab_imag':p['omega_global'][1],
            'alpha_real':branch['alpha'][0],'alpha_imag':branch['alpha'][1],
            'source':str(BASE/'phaseA/A3/summary.json'),
            'status':'existing_eigenvalue_reexpressed; not recomputed'})
    for case,(r,g) in fig8_curves.items():
        for R in (550.,570.,580.,585.,590.,600.):
            if r[0]<=R<=r[-1]:local_rows.append({'dataset':'DNS_Fig8' if case!='L01' else 'L01_Fig8',
                'case':case,'R':R,'variable':'total_azimuthal_RMS','g_ln_amplitude':float(np.interp(R,r,g)),
                'method':'linear_interpolation_of_new_published_derivative_digitization','T':6. if case!='L01' else None})
    csvwrite('growth_comparison.csv',comparison)
    csvwrite('field_summary.csv',summaries)
    csvwrite('local_growth_comparison.csv',local_rows)
    write('field_summary.json',summaries)
    write('validation.json',{'status':'PASS','legacy_reproduction':validation,
          'max_parseval_error':max(p['parseval_relative_error'] for p in summaries),
          'ln_energy_log10_factors':'energy=2*ln(amplitude); log10 amplitude=ln(amplitude)/ln10',
          'all_windows_in_source_range':True,'all_radius_arrays_strictly_increasing':True,
          'native_vertex_OLS_warning':'vector path vertices are correlated drawing points, not independent DNS samples',
          'new_NL01_T6_max_R_difference_from_existing':float(nl01_delta),
          'new_NL03_T6_max_R_difference_from_existing_independent_fig6_checks':float(nl03_delta),
          'runtime_seconds':time.monotonic()-start})
    report(comparison,summaries,contours,fig8_curves,local_rows)
    write('inputs_manifest.json',MANIFEST)
    write('DONE.json',{'status':'complete','scope':'D4 postprocessing; no solver runs',
          'runtime_seconds':time.monotonic()-start,'cpu_threads':1,'GPU':False,
          'historical_0_106_metadata_status':'unverified historical origin; independently reproducible Fig15 proxy identified',
          'Rout730_campaign_status':campaign['targets']['730.0']['status'],
          'report':str(OUT/'REPORT.md')})
    print(json.dumps({'status':'PASS','seconds':round(time.monotonic()-start,2),
        'profiles':len(summaries),'comparison_rows':len(comparison),'output':str(OUT)},ensure_ascii=False))


def report(rows, fields, contours, f8, locals_):
    def fmt(value):
        return '不可拟合（不足2点）' if value is None else f'{value:.9f}'
    def dns(case,a,b):
        return next(p for p in rows if p['case']==case and p['dataset'] in ('DNS_Fig15','DNS_Fig6_sparse') and p['window_lower']==a and p['window_upper']==b)
    nl03=dns('NL03',570.,600.);nl03up=dns('NL03',550.,570.)
    nl01=dns('NL01',570.,600.)
    nl03fig8=next(p for p in rows if p['dataset']=='DNS_Fig8' and p['case']=='NL03' and p['window_lower']==570 and p['window_upper']==600)
    accepted=[p for p in fields if p['dataset']=='phaseC' and p['H']>=5]
    finest=next(p for p in fields if p['dataset']=='phaseA' and p['id']=='N40_dr0.25')
    h6=sorted([p for p in accepted if p['H']==6],key=lambda x:x['Rout'])
    fronts=[p['front_fig8_existing'] for p in h6]
    assert all(v is not None for v in fronts)
    lines=[
      '# D4：0.124 与 DNS 约 0.106 的来源和口径复核',
      '',
      f'**主要结论。** 既有 NL03、T=6、z=1.3、局部盘速归一化总方位 RMS 的 Fig.15 数字化数据，在 R570–600 的端点自然对数斜率为 **{nl03["endpoint_ln_slope"]:.6f}**，重现约0.106。相同定义下已接受上支的相干总RMS约0.124，差距仍在。原0.106的历史窗口没有留档，不能把这次反向重现当作历史来源确认；其最明确的可复现对应口径现已找到。',
      '',
      '## 1. 来源、定义与证据等级',
      '',
      '- **原文信息**：Appelquist et al. (2016), DOI 10.1017/jfm.2016.506；Eq.(3.1), p342（PDF11）定义 vrms 为局部盘速归一化方位速度去方位均值后的RMS；图6、8、15使用z=1.3。T以完整盘转数计，T=t/(2π)。NL01在盘旋转系，NL03在实验室系；NL03相干谱峰ωlab=52.36，不能把它视作全RMS的严格单频特征值。',
      '- **历史记录**：`examples/fullbvp_phaseA_spatial.py:226–227`先存 `.106` 并明确写“user supplied approximate DNS comparison”；PhaseA报告沿用；`fullbvp_phaseB_report.py:134`后来称“DNS图读约0.106”，PhaseC再沿用。所查本地论文文本没有0.106，也未找到最早值的NL01/NL03、半径、窗口、拟合或平滑参数。具体传播链可核验，历史元数据未恢复。',
      '- **已有数字化信息**：NL03为Fig.15红实线/倒三角，142个保留矢量顶点，R535.011–700.010；灰色T→∞曲线和实验点已排除。NL01为Fig.6(b)、T=6六个等值线样本，R520.638–602.759；只按稀疏样本处理。来源JSON给出原PDF SHA256与图读包络。',
      '- **本次重算**：用已有CSV重新计算窗口斜率；从原PDF新提图8的L01/NL01/NL02/NL03实际可见矢量曲线，以及图6固定等值线T3/4/5/6位置。计算、标定、路径ID与图例颜色保存在专门CSV/JSON和源图叠加中。',
      '- **推断**：约0.106与Fig.15的570–600斜率吻合，也与Fig.8 NL03在R≈585的局部导数吻合。因此数值吻合本身不能唯一确定历史取值方法。',
      '',
      '## 2. DNS数值：同变量、不同图和不同窗口',
      '',
      '| 数据/时刻 | 窗口 | 实际内点数 | 端点ln斜率 | 原生顶点OLS | 等R权重OLS |',
      '|---|---:|---:|---:|---:|---:|',
    ]
    for case in ('NL03','NL01'):
        for a,b in WINDOWS:
            p=dns(case,a,b)
            lines.append(f'| {case}, T6 | {a:g}–{b:g} | {p["n_actual_interior"]} | {p["endpoint_ln_slope"]:.9f} | {fmt(p["native_vertex_OLS_ln_slope"])} | {p["uniform_R_OLS_ln_slope"]:.9f} |')
    lines += ['',
      f'NL03 R570–600端点值的保守图读传播包络为 [{nl03["endpoint_graphical_lower"]:.6f}, {nl03["endpoint_graphical_upper"]:.6f}]。它只传播既有±0.5 PDF点所给的横纵坐标读取容差；不是DNS误差或统计置信区间。原生顶点OLS依赖绘图分段的密度，等R权重OLS是1201节点数值积分，不增加DNS独立样本。NL01在570–600内只有{nl01["n_actual_interior"]}个实际等值线点，端点需在相邻已见等值线之间作ln幅值插值；其宽图读包络和低采样密度使其不能与连续NL03曲线等精度比较。',
      '',
      '| 图8末时刻局部导数 | R570 | R585 | R600 | 570–600积分平均 | 实际可见R范围 |',
      '|---|---:|---:|---:|---:|---|',
    ]
    for case in ('L01','NL01','NL02','NL03'):
        r,g=f8[case]
        p=next(p for p in rows if p['case']==case and p.get('method')=='window_average_of_published_local_derivative' and p['window_lower']==570 and p['window_upper']==600)
        vals=[np.interp(R,r,g) for R in (570,585,600)]
        lines.append(f'| {case} | {vals[0]:.9f} | {vals[1]:.9f} | {vals[2]:.9f} | {p["value"]:.9f} | {r[0]:.3f}–{r[-1]:.3f} |')
    lines += ['',
      f'图8的NL03窗口平均为{nl03fig8["value"]:.9f}，比Fig.15端点斜率低{nl03["endpoint_ln_slope"]-nl03fig8["value"]:.9f}。两图各自保存，不能称完全一致。图8轴由实际刻度线标定：R550/600/650及增长率0–0.30，避免文字框中心偏移；图15原有标定为log10幅值轴，转成ln后才求斜率。两图的曲线表示、图形分辨率与原文未记录的导数处理可以造成差别；此处不假定唯一原因。0.0033量级差在Fig.15保守图读包络内，但这不是相同DNS采样或统计一致性的证明。',
      '',
      '## 3. T=6并非已证实的稳态',
      '',
      '原文p345明确说非线性全局模态仍缓慢移动：NL01向内，NL02/NL03向外；p346图9追踪T3–6并外推渐近前沿582.8。原文给T6的前沿约NL01=589、NL02/NL03=570。这里的582.8是有限时间趋势与经验平移理论曲线的外推，既不是T6实测位置，也不是严格稳态证明。图7的时间增长率趋零不等于图8的径向增长率趋零。',
      '',
      '以下在同一图6面板、同一归一化RMS等值线下比较多个时刻；这是**固定等值线运动**，不冒称图8前沿定义或前沿速度。各点图读半径包络±2.5。',
      '',
      '| 案例 | log10 RMS固定值 | T3的R | T4的R | T5的R | T6的R | T6−T3 | T6−T5 |',
      '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for case in ('NL01','NL03'):
        for level in (-6,-4,-2):
            rr=[next(p['R'] for p in contours if p['case']==case and p['log10_RMS_level']==level and p['T']==T) for T in (3.,4.,5.,6.)]
            lines.append(f'| {case} | {level} | '+ ' | '.join(f'{v:.3f}' for v in rr)+f' | {rr[-1]-rr[0]:+.3f} | {rr[-1]-rr[-2]:+.3f} |')
    lines += ['',
      'T5到T6的小位移可能落在图读包络内，不能据此估计精确末时速度；原文的仍移动陈述和T3–6的明显演化共同支持有限时结论。约0.106可作为明确案例/时刻/窗口的有限时DNS对照，不能当作无条件稳态标量或最终BVP必须达到的目标。',
      '',
      f'R570–600相对于图8前沿：NL03从前沿处延伸到下游30；NL01从前沿上游19延伸至下游11；当前已接受H6上支的既有诊断前沿范围为{min(fronts):.6f}–{max(fronts):.6f}，因此窗口从前沿上游{min(fronts)-570:.3f}–{max(fronts)-570:.3f}延伸至下游{600-max(fronts):.3f}–{600-min(fronts):.3f}。各场前沿与诊断来源保存在field_summary中。相同绝对窗口混入的非线性段长度并不相同。另列550–570共同上游窗口供敏感性比较，不称其为0.106的原窗口。',
      '',
      '## 4. 实际上支场与线性场的重算',
      '',
      'q_v已除当地盘速。基频A1(R)=|q_v,1(z1.3)|；相干总RMS=√(2Σj≥1|q_v,j|²)，不计均值和压力，不再次除R。所有端点斜率均为[lnA(600)−lnA(570)]/30，端点在ln幅值中插值；原生OLS使用真实网格位于闭区间内的点。DNS是全方位RMS，可能含非相干/湍流成分；BVP仅含保留谐波的相干周期场。因此已匹配变量定义、归一化、高度和选定窗口，但未证明DNS动力学状态或全谱内容与BVP完全匹配。',
      '',
      '| 场 | Rout | H,N,区间数 | ωlab | 接受状态 | gq端点 | gq原生OLS | gRMS端点 | gRMS−gq |',
      '|---|---:|---|---:|---|---:|---:|---:|---:|',
    ]
    for p in sorted(fields,key=lambda p:(p['dataset'],p['Rout'] or 0,p['H'],p['id'])):
        route='—' if p['Rout'] is None else f'{p["Rout"]:.6f}'
        lines.append(f'| {p["id"]} | {route} | {p["H"]},{p["n"]},{p["intervals"]} | {p["omega_lab"]:.9f} | {p["acceptance"]} | {p["gq_endpoint"]:.9f} | {p["gq_OLS"]:.9f} | {p["gRMS_endpoint"]:.9f} | {p["RMS_minus_fundamental_endpoint"]:+.3e} |')
    lines += ['',
      '730的H5场虽然通过Newton残差，但campaign目标仍pending，两次H6提升失败；仅保留明确标记的未提交诊断，不把它们混入已接受上支或出口收敛结论。H6在718.932、720、725均为已接受的严格辅助审计。',
      '另在统一CSV中保留PhaseA高度积分量E=∫Σ速度分量|q|²dz的√E增长率。旧g_E实际是½d lnE/dR，已经是振幅等效量；不能再当作能量ln增长率而随意除2。CSV中`energy_ln_endpoint`始终为同一标量A²的ln斜率；只有高度积分E对应的行才代表该E，RMS²行代表方位方差。',
      '',
      '| 已接受H6 | 既有前沿 | 窗口相对前沿 | 570–600相干RMS | 减NL03同窗 | 550–570相干RMS | 减NL03同上游窗 | H6−H5同窗RMS |',
      '|---|---:|---|---:|---:|---:|---:|---:|',
    ]
    for p in h6:
        p5=next(x for x in accepted if x['H']==5 and abs(x['Rout']-p['Rout'])<1e-7)
        front=p['front_fig8_existing']
        lines.append(f'| {p["Rout"]:.6f} | {front:.6f} | [{570-front:+.3f}, {600-front:+.3f}] | {p["gRMS_endpoint"]:.9f} | {p["gRMS_endpoint"]-nl03["endpoint_ln_slope"]:+.9f} | {p["gRMS_550_570"]:.9f} | {p["gRMS_550_570"]-nl03up["endpoint_ln_slope"]:+.9f} | {p["gRMS_endpoint"]-p5["gRMS_endpoint"]:+.3e} |')
    lines += ['',
      f'以725的H6为例，即使把NL03端点值取到既有图读包络上限，差仍为{h6[-1]["gRMS_endpoint"]-nl03["endpoint_graphical_upper"]:.6f}。这个比较只覆盖图读容差，不能替代DNS离散或统计不确定度评估。',
      '',
      f'精细PhaseA场（N40、ΔR≈0.25）的R570–600同口径端点值为{finest["gq_endpoint"]:.9f}，原生OLS为{finest["gq_OLS"]:.9f}；旧报告约0.132有离散局部导数均值口径，不能把它与端点/OLS强行当作完全同值。局部k−数据沿已有PhaseA中性频率追踪，ωlab≈53.085700；上支约52.68–52.71，DNS谱峰52.36。下面是局部值，不冒充570–600拟合：',
      '',
      '| 数据集 | ωlab | 诊断/接受状态 | R570 | R580 | R590 | R600 |',
      '|---|---:|---|---:|---:|---:|---:|',
    ]
    choices=[('local_k_minus','global',None,'因果k−冻结径向','现有−Imα'),
             ('local_k_minus','pier',None,'因果k−经典冻结','现有−Imα'),
             ('phaseA',finest['id'],'fund','PhaseA N40 ΔR≈0.25','基频；线性中性')]
    choices.extend([('phaseC',p['id'],'fund',f'上支H6 Rout{p["Rout"]:.6f}','基频；已接受严格审计') for p in h6])
    uncommitted=next(p for p in fields if p['dataset']=='phaseC_uncommitted')
    choices.append(('phaseC_uncommitted',uncommitted['id'],'fund','Rout730 H5','基频；Newton通过，campaign未提交'))
    choices.extend([('DNS_Fig8',case,None,f'{case} Fig8 T6','原图总RMS局部导数') for case in ('NL01','NL03')])
    unified=[]
    for dataset,case,var,label,status in choices:
        selected=[next(p for p in locals_ if p['dataset']==dataset and p['case']==case and p['R']==R and
                       (var is None or p['variable']==var)) for R in (570.,580.,590.,600.)]
        omega=selected[0].get('omega_lab')
        dns_peak=52.36 if dataset=='DNS_Fig8' and case=='NL03' else None
        omega_text=('52.36谱峰' if dns_peak is not None else '未核实') if dataset=='DNS_Fig8' else f'{omega:.6f}'
        values=[p['g_ln_amplitude'] for p in selected]
        lines.append(f'| {label} | {omega_text} | {status} | '+' | '.join(f'{v:.9f}' for v in values)+' |')
        unified.append({'dataset':dataset,'case':case,'label':label,'omega_lab':omega,
                        'DNS_spectral_peak_lab':dns_peak,
                        'diagnostic_and_status':status,'method':selected[0]['method'],
                        **{f'R{R:g}':v for R,v in zip((570.,580.,590.,600.),values)}})
    csvwrite('unified_local_growth.csv',unified)
    lines += ['',
      '此表k−为现有α重新表达−Imα，未重新求特征值；当前场局部导数使用二阶numpy.gradient，再在R上线性插值。图8灰线使用复频率（盘系−15.56+0.568i或−13.78+0.981i），与项目实频率k−不是同一构造。跨频率差不能全部归因于非线性。',
      '',
      '## 5. 对差距的解释与证据强度',
      '',
      f'1. **最强证据：案例、窗口相对前沿位置和有限时状态。** NL03同窗端点{nl03["endpoint_ln_slope"]:.6f}，上游550–570为{nl03up["endpoint_ln_slope"]:.6f}，相差{nl03up["endpoint_ln_slope"]-nl03["endpoint_ln_slope"]:.6f}；NL01同窗稀疏端点为{nl01["endpoint_ln_slope"]:.6f}，图8同窗平均约0.128。相同标签“DNS增长率”不能跨案例、半径或时刻当一个常数。当前上支前沿更接近NL01的T6位置，故与NL03同绝对窗口的差异首先表现为前沿/曲线位置差；相关性尚不确定其物理成因。',
      f'2. **已量化且不足以解释：诊断转换。** 已接受H5/H6中，基频改为相干总RMS只改变端点值约{min(p["RMS_minus_fundamental_endpoint"] for p in accepted):.2e}–{max(p["RMS_minus_fundamental_endpoint"] for p in accepted):.2e}。物理速度而非盘速归一化幅值额外增加 ln(600/570)/30={np.log(600/570)/30:.9f}，且同口径比较应同时转换两侧。能量ln斜率恰为幅值的2倍；log10斜率为ln斜率除ln10；这些明确的转换不是约17%差距的任意修正系数。',
      '3. **部分量化：非线性改变场，但目标未必相同。** 上支相对PhaseA增长率下降约0.008；H5→H6变化远小于与NL03同窗差距，说明这一级谐波截断差不是主解释。它不证明法向/径向分辨率或更高谐波已完全收敛。DNS含真实转捩/湍流，当前BVP求相干单频周期解；不同全局频率、边界和下游结构同时存在，不能把全部差值分配给一个因素。',
      '4. **尚未验证的物理原因**：海绵起点/强度、下游域长、法向与径向分辨率、DNS有限时间残余、相干与非相干能量和频率变化。D4未做新求解或参数敏感性，不将海绵相关性写成已证明原因。',
      '',
      '## 6. 可复现性与未解决边界',
      '',
      '运行 `bash results/fullbvp_global_20261030/phaseD/D4/commands.sh`。仅单线程读取现有场和原PDF，不编译、不求解、不用GPU。`growth_comparison.csv`给出窗口端点、实际内点数、首末内点、三种斜率定义、能量/log10/物理速度转换；`field_summary.csv/json`为场级汇总；`local_growth_comparison.csv`保留局部值与频率。`inputs_manifest.json`保存确切输入路径、SHA256和大小；`source_index.csv`保存来源行号/页码；两张源图叠加用于检查新数字化的曲线身份。',
      '',
      '尚未恢复0.106最早输入的原始拟合窗口、DNS原场/平滑方法或统计误差；也未得到多个时刻的Fig.8径向导数全曲线。本次多时刻证据是Fig.6固定等值线位置，不能用于宣称某个稳态径向增长率。新比较已经明确重现NL03约0.106的可执行口径，并确认同口径BVP差距仍存在；不把缺失元数据补造成“完全匹配”。',
    ]
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


if __name__=='__main__':
    try:
        main()
    except Exception as exc:
        OUT.mkdir(parents=True,exist_ok=True)
        write('FAILED.json',{'status':'failed','type':type(exc).__name__,'message':str(exc)})
        raise
