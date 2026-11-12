#!/usr/bin/env python3
"""Plot the saved-field Ro=-0.5 wave structure at final paper size.

Only the frozen projections in this release are read. No flow solves.
"""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/prf_bek_mpl')
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import _mathtext
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.ticker import AutoMinorLocator, MultipleLocator, FixedLocator

PAPER = Path(__file__).resolve().parent
DATA = PAPER/'data/wave_structure'
OUT = PAPER/'figures'
META = json.loads((DATA/'metadata.json').read_text())
W = META['wave']
assert hashlib.sha256((DATA/'projections.npz').read_bytes()).hexdigest() == META['array_sha256']
D = np.load(DATA/'projections.npz',allow_pickle=False)
BLUE,RED,GRAY = '#2166ac','#b2182b','#626262'
for FONT in ('Times New Roman','Nimbus Roman','STIXGeneral'):
    try:
        fm.findfont(fm.FontProperties(family=FONT), fallback_to_default=False)
        break
    except ValueError:
        continue
else:
    raise RuntimeError('Install Times New Roman, Nimbus Roman, or STIXGeneral.')
_mathtext.SHRINK_FACTOR = .8
_mathtext.GROW_FACTOR = 1/.8
plt.rcParams.update({'font.family':'serif','font.serif':[FONT],
                     'mathtext.fontset':'stix','font.size':10.,'axes.labelsize':10.,
                     'xtick.labelsize':10.,'ytick.labelsize':10.,'legend.fontsize':10.,
                     'axes.linewidth':.85,'axes.edgecolor':'black','text.color':'black',
                     'axes.labelcolor':'black','xtick.color':'black','ytick.color':'black',
                     'lines.linewidth':1.3,'pdf.fonttype':42,'savefig.dpi':360,
                     'figure.facecolor':'white','axes.facecolor':'white','axes.unicode_minus':True})
anchors = np.array([(33,102,172),(103,169,207),(209,229,240),(247,247,247),
                    (253,219,199),(239,138,98),(178,24,43)])/255.
cmap = LinearSegmentedColormap.from_list('author_tecplot_red_blue',anchors,N=257)
cmap.set_bad('#bfbfbf')
fig = plt.figure(figsize=(7.,5.5))
ax = fig.add_axes([.10,.62,.87,.31])
bx = fig.add_axes([.10,.20,.37,.28])
cx = fig.add_axes([.60,.20,.37,.28])
for label,a in zip('abc',[ax,bx,cx]):
    a.tick_params(which='major',direction='in',top=True,right=True,length=4.,width=.8,pad=3.)
    a.tick_params(which='minor',direction='in',top=True,right=True,length=2.,width=.65)
    a.xaxis.labelpad=5.5
    a.yaxis.labelpad=6.
    a.text(0.,1.025,'($'+label+'$)',transform=a.transAxes,va='bottom',ha='left',fontsize=10.)
    a.set_xlabel(r'$R$')
    a.xaxis.set_minor_locator(AutoMinorLocator(2))

series = [
    (D['linear_r'],D['linear_v1_abs_scaled'],RED,'--',r'$C|v_L|$'),
    (D['nonlinear_r'],D['v1_abs'],GRAY,'-',r'$|v_1|$'),
    (D['nonlinear_r'],D['v2_abs'],BLUE,'-.',r'$|v_2|$'),
    (D['nonlinear_r'],D['v0_abs'],'black',':',r'$|v_0|$')]
for r,y,color,ls,label in series:
    ax.semilogy(r,np.ma.masked_less_equal(y,0),color=color,ls=ls,label=label)
ax.set(xlim=(600,856),ylim=(1e-8,.30),ylabel=r'$|v_j|$')
ax.xaxis.set_major_locator(MultipleLocator(50))
ax.yaxis.set_major_locator(FixedLocator([1e-8,1e-6,1e-4,1e-2]))
for pos,lab in [(W['R_ca'],r'$R_{\rm ca}$'),(W['nonlinear_front'],r'$R_f$'),
                (W['nonlinear_a'],r'$a$')]:
    ax.axvline(pos,color='#909090',ls=(0,(2,2)),lw=.75,zorder=0)
    ax.text(pos-1.6,.97,lab,transform=ax.get_xaxis_transform(),
            va='top',ha='right',fontsize=10.)
ax.legend(loc='lower right',frameon=False,ncol=1,handlelength=2.2,columnspacing=1.6,
          handletextpad=.6,labelspacing=.35,borderaxespad=.8)

field_specs = [(bx,D['field_v1_real_twice'],.08,r'$2\mathrm{Re}\,v_1$',[-.08,0,.08],.10),
               (cx,D['field_v0_real'],.05,r'$v_0$',[-.05,0,.05],.60)]
field_audits = []
for a,values,limit,label,ticks,left in field_specs:
    actual=float(np.max(abs(values)))
    assert actual < limit, (label,'field color clipping',actual,limit)
    im = a.pcolormesh(D['field_r'],D['field_z'],values.T,shading='nearest',
                      cmap=cmap,norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),
                      rasterized=True)
    a.axvline(W['nonlinear_front'],color='black',lw=.65,ls=(0,(2,2)))
    a.axvline(W['nonlinear_a'],color='black',lw=.65,ls=(0,(5,3)))
    a.set(xlim=(680,790),ylim=(0,5),ylabel=r'$z$')
    a.xaxis.set_major_locator(MultipleLocator(40))
    a.yaxis.set_major_locator(MultipleLocator(2))
    a.yaxis.set_minor_locator(AutoMinorLocator(2))
    ca = fig.add_axes([left,.087,.37,.022])
    cb = fig.colorbar(im,cax=ca,orientation='horizontal',ticks=ticks)
    cb.ax.tick_params(direction='in',labelsize=10.,length=3.,width=.65,pad=3.)
    cb.set_label(label,labelpad=5.5,fontsize=10.)
    cb.outline.set_linewidth(.7)
    field_audits.append({'quantity':label,'display_max_abs':actual,'color_limits':[-limit,limit],
                         'color_clipping':False})

OUT.mkdir(exist_ok=True)
fig.savefig(OUT/'wave_structure.pdf')
fig.savefig(OUT/'wave_structure.png')
plt.close(fig)
record = {'figure':'wave_structure','size_inches':[7.,5.5],'font':FONT,'font_points':10.,
          'math_script_points':8.,'pdf':'figures/wave_structure.pdf','png':'figures/wave_structure.png',
          'panels':{'a':'Scaled azimuthal coefficient envelopes at z=1.3; arbitrary linear amplitude C is matched in a common pre-sponge interval.',
                    'b':'Fundamental contribution 2 Re v1 at beta theta-omega t=0 in the saved phase convention.',
                    'c':'Real azimuthal mean correction v0.'},
          'field_audits':field_audits,'field_display_R':[680.,790.],'field_display_z':[0.,5.],'array_sha256':hashlib.sha256((DATA/'projections.npz').read_bytes()).hexdigest(),
          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'line_semantic_colors':{'linear':RED,'fundamental':GRAY,'second_harmonic':BLUE,'mean':'black'},
          'sponge_marking':'a=732.404649566 is the NL onset; no claim that the mean is damped.',
          'whole_domain':'Top panel shows R=600..856 of the full solution; field panels show R=680..790,z=0..5, with native R nodes and documented z interpolation.'}
(OUT/'wave_structure_metadata.json').write_text(json.dumps(record,indent=2)+'\n')
print('Generated wave_structure PDF/PNG; source projections',record['array_sha256'])
