"""Extract AF0.1 first peak/valley from two converged full radial BVPs."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import BarycentricInterpolator

from llsa.secondary.radial import RadialConfig


ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/radial_sponge_validation'
OUT=ROOT/'results/beta42_af1_steady/af01_full_bvp_peak_valley'
CASES=[('sponge at 480','h5_dr05_s480_w40_a10_end520',480,'#777777','--'),
       ('sponge at 500','h5_dr05_s500_w40_a10_end540',500,'#1764a0','-')]
Z=1.1858025795710305


def curve(folder):
    with np.load(SOURCE/folder/'fields.npz') as d:
        r,modes=d['radii'],d['modes'];c=RadialConfig(**json.loads(str(d['config_json'])))
    op=c.operators(r[0]);finite=np.isfinite(op.grid.z)
    v=modes[:,c.harmonics+1,op.m:2*op.m]
    a=np.sqrt(2)*np.abs(BarycentricInterpolator(op.grid.z[finite],v[:,finite],axis=1)(Z))/op.disk_rotation
    return abs(c.ro)*r,a


def extremum(x,y,kind):
    ids=np.where(((y[1:-1]>y[:-2])&(y[1:-1]>=y[2:])) if kind=='max' else
                 ((y[1:-1]<y[:-2])&(y[1:-1]<=y[2:])))[0]+1
    i=int(ids[0]);c=np.polyfit(x[i-2:i+3],np.log(y[i-2:i+3]),2);xp=-c[1]/(2*c[0])
    return float(xp),float(np.exp(np.polyval(c,xp)))


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','STIXGeneral','DejaVu Serif'],
        'mathtext.fontset':'stix','font.size':11,'axes.linewidth':1,'xtick.direction':'in',
        'ytick.direction':'in','xtick.top':True,'ytick.right':True,'legend.frameon':False,'pdf.fonttype':42})
    fig,ax=plt.subplots(1,2,figsize=(9.5,3.8),constrained_layout=True);report=[]
    for label,folder,sponge,color,style in CASES:
        re,a=curve(folder);physical=re<=sponge+1e-10;p=extremum(re[physical],a[physical],'max');v=extremum(re[physical],a[physical],'min')
        for axis in ax:axis.plot(re[physical],a[physical],style,color=color,lw=1.9,label=label)
        ax[0].plot(*p,'o',color=color,ms=5);ax[0].plot(*v,'s',color=color,ms=5)
        report.append({'case':folder,'sponge_start_Re':sponge,'first_peak':{'Re':p[0],'A42':p[1]},
            'first_valley':{'Re':v[0],'A42':v[1]},'amplitude_decay_factor':p[1]/v[1],
            'energy_decay_factor':(p[1]/v[1])**2})
    ax[0].set(xlim=(320,500),ylim=(0,.125),xlabel=r'$Re$',ylabel=r'$v_{\mathrm{rms},42}/(\Omega_d r)$',title='(a) Full radial BVP')
    ax[1].set(xlim=(380,480),ylim=(0,.125),xlabel=r'$Re$',ylabel=r'$v_{\mathrm{rms},42}/(\Omega_d r)$',title='(b) First peak and valley')
    for axis in ax:axis.legend(fontsize=8.5);axis.tick_params(length=3.5)
    for ext in ('png','pdf'):fig.savefig(OUT/f'af01_full_bvp_peak_valley.{ext}',dpi=300,facecolor='white')
    plt.close(fig)
    (OUT/'af01_full_bvp_peak_valley.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
