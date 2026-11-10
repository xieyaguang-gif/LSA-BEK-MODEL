#!/usr/bin/env python3
"""Plot exact saved successful Ro/beta trajectory; no interpolation or smoothing."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/ro_scan_20261101/S0/diag_0p5'
sys.path.insert(0,'/home/lthpc/.codex/skills/nature-figure/scripts')
from audit_panel_alignment import require_matplotlib_panel_alignment

def main():
    data=json.loads((OUT/'trajectory_last20.json').read_text()); rows=data['records']
    ro=np.array([r['ro'] for r in rows]); re=np.array([r['Re_ca'] for r in rows]); a=np.array([r['alpha'] for r in rows]); w=np.array([r['omega_raw'][0] for r in rows]); b=np.array([r.get('beta_requested',r['beta']) for r in rows])
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Nimbus Roman','DejaVu Serif'],'mathtext.fontset':'stix','font.size':9,'axes.labelsize':10,'axes.linewidth':.8,'pdf.fonttype':42,'xtick.direction':'in','ytick.direction':'in','axes.formatter.useoffset':False})
    fig,axs=plt.subplots(2,2,figsize=(7.2,5.0),sharex=True)
    fig.subplots_adjust(left=.12,right=.97,bottom=.12,top=.93,wspace=.38,hspace=.35)
    lines=[[(re,'#14638d','o',None)],[(a[:,0],'#14638d','o',r'$\alpha_r$'),(a[:,1],'#b65d21','s',r'$\alpha_i$')],[(w,'#14638d','o',None)],[(b,'#14638d','o',None)]]
    labels=[r'$Re$',r'$\alpha$',r'$\mathrm{Re}(\omega_{\mathrm{raw}})$',r'$\beta$ (continuous)']
    for i,(ax,series,label) in enumerate(zip(axs.ravel(),lines,labels)):
        for y,c,m,leg in series: ax.plot(ro,y,color=c,marker=m,markersize=3,linewidth=1,label=leg)
        ax.set_ylabel(label); ax.text(0,1.025,chr(97+i),transform=ax.transAxes,va='bottom',fontweight='bold')
        ax.tick_params(top=True,right=True)
        if i==1: ax.legend(loc='center left',frameon=False,fontsize=8)
        if i>1: ax.set_xlabel(r'$Ro$')
        ax.locator_params(axis='x',nbins=4)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig,json_out=OUT/'trajectory.alignment.json',tolerance_pt=1.5)
    fig.savefig(OUT/'trajectory_last20.pdf'); fig.savefig(OUT/'trajectory_last20.png',dpi=300)
    plt.close(fig)
    brief={'claim':'Saved neutral parameters remain continuous before the ARPACK stop; beta growth follows the prescribed beta*abs(Ro) path.','type':'quantitative grid','panels':{'a':'neutral Re approach','b':'complex saddle location','c':'neutral raw frequency','d':'prescribed azimuthal path'},'input':'trajectory_last20.json','rows':len(rows),'exclusions':'only successful steps; final up-to-20 requested by task','transformations':'none; straight segments join samples','units':'raw local nondimensional alpha and omega; beta continuous','normalization':'none; independent y limits; shared Ro axis','figure_inches':[7.2,5.0],'beta_absRo_minmax':[float(min(b*abs(ro))),float(max(b*abs(ro)))],'review_risk':'A continuous noninteger beta path is a seed path, not the optimized integer-beta critical curve.'}
    (OUT/'figure_contract.json').write_text(json.dumps(brief,indent=2)+'\n')
    print(json.dumps(brief))
if __name__=='__main__': main()
