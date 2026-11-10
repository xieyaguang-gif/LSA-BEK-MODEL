"""AF1 step-sensitivity audit focused on first peak, valley, and decay."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT/"results/beta42_af1_steady/af1_delta_r_independence"
OUT = ROOT/"results/beta42_af1_steady/af1_first_peak_valley_independence"
CASES = [(5,OLD/"dr5/march.csv"),(5.5,OUT/"dr5p5/march.csv"),
         (6,OLD/"dr6/march.csv"),(6.5,OUT/"dr6p5/march.csv"),
         (7,OLD/"dr7/march.csv"),(8,OLD/"dr8/march.csv")]


def fit(x,y,i):
    c=np.polyfit(x[i-1:i+2],np.log(y[i-1:i+2]),2)
    xp=-c[1]/(2*c[0])
    return float(xp),float(np.exp(np.polyval(c,xp)))


def main():
    plt.rcParams.update({"font.family":"serif","font.serif":["Times New Roman","STIXGeneral","DejaVu Serif"],
        "mathtext.fontset":"stix","font.size":10,"axes.linewidth":1,"xtick.direction":"in",
        "ytick.direction":"in","xtick.top":True,"ytick.right":True,"legend.frameon":False,"pdf.fonttype":42})
    rows=[]; curves=[]
    for dr,path in CASES:
        d=np.genfromtxt(path,delimiter=',',names=True);x=d['Re'];a=d['A1_vtheta_over_disk_speed']
        ip=np.where((a[1:-1]>a[:-2])&(a[1:-1]>=a[2:]))[0][0]+1
        iv=np.where((a[1:-1]<a[:-2])&(a[1:-1]<=a[2:]))[0][0]+1
        peak,valley=fit(x,a,ip),fit(x,a,iv)
        rows.append(dict(dr=dr,peak_Re=peak[0],peak_A42=peak[1],valley_Re=valley[0],
                         valley_A42=valley[1],decay_factor=peak[1]/valley[1],
                         remaining_fraction=valley[1]/peak[1]))
        curves.append((dr,x,a))
    fine=rows[:3]
    def band(key):
        v=np.array([r[key] for r in fine]);return dict(mean=float(v.mean()),half_range=float(np.ptp(v)/2),minimum=float(v.min()),maximum=float(v.max()))
    report={"runs":rows,"finest_stable_window_dr_5_to_6":{k:band(k) for k in
        ('peak_Re','peak_A42','valley_Re','valley_A42','decay_factor','remaining_fraction')},
        "strict_delta_r_to_zero_convergence":False,
        "reason":"Delta r=2,3,4,4.5 fail before the first peak in the current adaptive-carrier PSE/BDF2 scheme."}

    fig,ax=plt.subplots(2,2,figsize=(9.5,7),constrained_layout=True)
    cmap=plt.get_cmap('viridis')
    for k,(dr,x,a) in enumerate(curves):
        ax[0,0].plot(x,a,color=cmap(k/(len(curves)-1)),lw=1.6,label=rf'$\Delta r={dr:g}$')
    ax[0,0].set(xlim=(320,465),ylim=(0,.11),xlabel=r'$Re$',ylabel=r'$A_{42}$',title='(a) First peak and valley')
    ax[0,0].legend(fontsize=8,ncol=2)
    dr=np.array([r['dr'] for r in rows])
    ax[0,1].plot(dr,[r['peak_Re'] for r in rows],'o-',label=r'$Re_{peak}$')
    ax[0,1].plot(dr,[r['valley_Re'] for r in rows],'s-',label=r'$Re_{valley}$')
    ax[0,1].set(xlabel=r'$\Delta r$',ylabel=r'$Re$',title='(b) Locations');ax[0,1].legend()
    ax[1,0].plot(dr,[r['peak_A42'] for r in rows],'o-',label=r'$A_{peak}$')
    ax[1,0].plot(dr,[r['valley_A42'] for r in rows],'s-',label=r'$A_{valley}$')
    ax[1,0].set(xlabel=r'$\Delta r$',ylabel=r'$A_{42}$',title='(c) Amplitudes');ax[1,0].legend()
    ax[1,1].plot(dr,[r['decay_factor'] for r in rows],'o-',color='#D55E00')
    ax[1,1].set(xlabel=r'$\Delta r$',ylabel=r'$A_{peak}/A_{valley}$',title='(d) Decay factor')
    for a in ax.ravel():a.tick_params(length=3.5)
    for ext in ('png','pdf'):fig.savefig(OUT/f'af1_first_peak_valley_independence.{ext}',dpi=300,facecolor='white')
    plt.close(fig)
    (OUT/'af1_first_peak_valley_independence.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report['finest_stable_window_dr_5_to_6'],indent=2))


if __name__=='__main__':main()
