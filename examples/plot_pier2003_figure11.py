"""Plot the Pier (2003) Figure 11 secondary absolute-growth scan."""
import json
from pathlib import Path
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]; out=ROOT/'results/pier2003'
rows=np.genfromtxt(out/'figure11_current_even_0_50.csv',delimiter=',',names=True,dtype=None,encoding='utf-8')
paper=np.genfromtxt(out/'pier2003_figure11_digitized.csv',delimiter=',',names=True,dtype=None,encoding='utf-8')
mpl.rcParams.update({'font.family':'serif','font.serif':['STIXGeneral','DejaVu Serif'],'mathtext.fontset':'stix','font.size':9,'axes.linewidth':.8,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,ax=plt.subplots(figsize=(4.2,3.0),constrained_layout=True)
ax.plot(rows['beta_hat'],rows['omega_imag'],'o-',color='black',ms=3,lw=1.1,label='current solver, adaptive continuation')
valid=np.isfinite(paper['omega0_imag'])
ax.plot(paper['beta_hat'][valid],paper['omega0_imag'][valid],'s--',color='#b23a48',mfc='white',ms=3.5,lw=1.2,label='Pier (2003), Fig. 11 digitized')
ax.plot([20],[1.1597],marker='D',ms=4.5,mfc='#2b6cb0',mec='#2b6cb0',ls='none',label='Pier Table 3')
ax.axhline(0,color='#777777',lw=.7)
ax.set(xlim=(0,52),ylim=(-1.0,1.35),xlabel=r'$\hat\beta$',ylabel=r'$\hat\omega_{0,i}$')
ax.legend(frameon=False,fontsize=7,loc='upper right')
fig.savefig(out/'figure11_reproduction_current_vs_pier.pdf'); fig.savefig(out/'figure11_reproduction_current_vs_pier.svg'); fig.savefig(out/'figure11_reproduction_current_vs_pier.png',dpi=400)
print(out/'figure11_reproduction_current_vs_pier.png')
