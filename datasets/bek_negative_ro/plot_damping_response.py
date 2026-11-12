#!/usr/bin/env python3
"""Reproduce damping-response comparison from frozen scalars and projections."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS'): os.environ[key]='1'
os.environ.setdefault('MPLCONFIGDIR','/tmp/prf_bek_mpl')
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib import _mathtext
from matplotlib.ticker import AutoMinorLocator,MultipleLocator
PAPER=Path(__file__).resolve().parent
DATA=PAPER/'data/damping_response'
D=json.loads((DATA/'metadata.json').read_text())
assert hashlib.sha256((DATA/'profiles.npz').read_bytes()).hexdigest()==D['array_sha256']
P=np.load(DATA/'profiles.npz',allow_pickle=False)
for FONT in ('Times New Roman','Nimbus Roman','STIXGeneral'):
 try:
  fm.findfont(fm.FontProperties(family=FONT),fallback_to_default=False)
  break
 except ValueError:
  continue
else:
 raise RuntimeError('Install Times New Roman, Nimbus Roman, or STIXGeneral.')
_mathtext.SHRINK_FACTOR=.8;_mathtext.GROW_FACTOR=1/.8
plt.rcParams.update({'font.family':'serif','font.serif':[FONT],'mathtext.fontset':'stix',
 'font.size':10,'axes.labelsize':10,'xtick.labelsize':10,'ytick.labelsize':10,'legend.fontsize':10,
 'axes.linewidth':.85,'pdf.fonttype':42,'savefig.dpi':360,'lines.linewidth':1.35,
 'figure.facecolor':'white','axes.facecolor':'white','axes.unicode_minus':True})
BLUE,RED,GRAY='#2166ac','#b2182b','#626262'
fig,ax=plt.subplots(1,2,figsize=(7,3.15))
fig.subplots_adjust(left=.095,right=.985,bottom=.225,top=.91,wspace=.40)
for i,a in enumerate(ax):
 a.tick_params(which='major',direction='in',top=True,right=True,length=4,width=.8,pad=3)
 a.tick_params(which='minor',direction='in',top=True,right=True,length=2,width=.65)
 a.xaxis.labelpad=5.5;a.yaxis.labelpad=6
 a.text(0,1.025,'($'+chr(97+i)+'$)',transform=a.transAxes,ha='left',va='bottom')
 a.xaxis.set_minor_locator(AutoMinorLocator(2))
rows=D['rows'];base=next(r for r in rows if r['strength']==28)
x=[r['strength'] for r in rows]
lin=[r['linear_Rout_c']-base['linear_Rout_c'] for r in rows]
nonlin=[r['diagnostic']['front']-base['diagnostic']['front'] for r in rows]
ax[0].plot(x,lin,'s-',color=RED,mfc='white',label=r'$\Delta R_{\mathrm{out},c}$',markersize=5)
ax[0].plot(x,nonlin,'o--',color=BLUE,mfc='white',label=r'$\Delta R_f$',markersize=5)
ax[0].axhline(0,color='black',lw=.65,ls=(0,(2,2)),zorder=0)
ax[0].set(xlim=(10,60),ylim=(-4.1,4.1),xlabel=r'$S$',ylabel='Position change')
ax[0].set_xticks([14,28,56]);ax[0].xaxis.set_minor_locator(AutoMinorLocator(2))
ax[0].yaxis.set_major_locator(MultipleLocator(2));ax[0].yaxis.set_minor_locator(AutoMinorLocator(2))
ax[0].legend(loc='upper left',frameon=False,handlelength=2.2,handletextpad=.6)
for r,color,ls in zip(rows,[BLUE,GRAY,RED],['--','-',':']):
 key=f"S{r['strength']}";rad=P[key+'_r'];rms=P[key+'_rms']
 ax[1].semilogy(rad,np.ma.masked_less_equal(rms,0),color=color,ls=ls,label=rf"$S={r['strength']}$")
ax[1].axvline(base['a'],color='black',lw=.75,ls=(0,(2,2)),zorder=0)
ax[1].text(base['a']+2,2e-5,r'$a$',ha='left',va='center')
ax[1].set(xlim=(650,810),ylim=(1e-7,.15),xlabel=r'$R$',ylabel=r'$V_q$')
ax[1].xaxis.set_major_locator(MultipleLocator(50))
ax[1].legend(loc='upper left',frameon=False,handlelength=2.2,handletextpad=.6,labelspacing=.4)
for ext in ('pdf','png'):fig.savefig(PAPER/'figures'/f'damping_response.{ext}')
plt.close(fig)
print('Generated damping_response: neutral boundary and fixed-domain front/profile comparison.')
print('Spans:',max(lin)-min(lin),max(nonlin)-min(nonlin))
