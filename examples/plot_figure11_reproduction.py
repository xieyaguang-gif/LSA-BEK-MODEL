"""Publication-style Python remake of the supplied paper's Figure 11."""
import json
from pathlib import Path
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; inp=ROOT/'results/figure11_ro_minus0p751/figure11_data.json'; out=inp.parent; d=json.loads(inp.read_text())
mpl.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral','DejaVu Serif'],'mathtext.fontset':'stix','font.size':8,'axes.linewidth':.8,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,ax=plt.subplots(1,2,figsize=(7.0,2.8),constrained_layout=True)
valid=[r for r in d['beta_scan'] if r['success']]
ax[0].plot([r['re'] for r in valid],[r['beta'] for r in valid],'-o',ms=2.8,lw=1.25,color='black')
ax[0].axhline(85,color='#777777',lw=.7,ls='--'); ax[0].axvline(409.6,color='#777777',lw=.7,ls='--')
ax[0].set(xlim=(300,700),ylim=(0,300),xlabel=r'$Re$',ylabel=r'$\beta$',title='(a) Absolute-neutral saddle')
valid=[r for r in d['frequency_scan_beta85'] if r['success']]
re=np.array([r['re'] for r in valid]); wr=-np.array([r['omega'][0] for r in valid]); wi=np.array([r['omega'][1] for r in valid])
ax[1].plot(re,wi,'k-',lw=1.3,label=r'$\omega_i$'); ax[1].plot(re,wr,color='#b0b0b0',lw=1.3,label=r'$-\omega_r$')
ax[1].axhline(0,color='#777777',lw=.7); ax[1].axvline(409.6,color='#777777',lw=.7,ls='--')
ax[1].set(xlim=(400,1200),xlabel=r'$Re$',ylabel=r'$\omega_i$',title=r'(b) $\beta=85$')
ax[1].text(.80,.18,r'$\omega_i$',transform=ax[1].transAxes,color='black')
ax[1].text(.80,.08,r'$-\omega_r$',transform=ax[1].transAxes,color='#777777')
fig.savefig(out/'figure11_reproduction.png',dpi=400); fig.savefig(out/'figure11_reproduction.pdf'); fig.savefig(out/'figure11_reproduction.svg'); print(out/'figure11_reproduction.png')
