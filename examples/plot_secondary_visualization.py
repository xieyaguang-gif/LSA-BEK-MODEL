"""Reconstruct saved Fourier coefficients; export plots, animation and offline HTML.

Uses system Python with NumPy, Matplotlib and Pillow; no SciPy or solver needed.
All pictures show w-W_laminar, not streamlines or vortex-core isosurfaces.
"""
import base64
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR','/tmp/llsa_matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/secondary_visualization'
DATA=json.loads((OUT/'mode_data.json').read_text())
P=np.array(DATA['primary_real'])+1j*np.array(DATA['primary_imag'])
S=np.array(DATA['secondary_real'])+1j*np.array(DATA['secondary_imag'])
NS=np.array(DATA['harmonics'])
AH=complex(*DATA['alpha_hat'])
WH=complex(*DATA['omega_hat'])
R=DATA['radius']
ALPHA=DATA['alpha']
OMEGA=DATA['omega']
BETA=DATA['beta']
BH=DATA['beta_hat']
REF=3
RMS=DATA['primary_w_rms_reference']
CMAP=LinearSegmentedColormap.from_list('signed_w',['#164b86','#4d93c7','#f8f8f8','#e78b75','#a52d32'])
plt.rcParams.update({'font.family':['Noto Serif CJK JP','STIXGeneral'],
                     'mathtext.fontset':'stix','font.size':10,'axes.linewidth':.8,
                     'xtick.direction':'in','ytick.direction':'in','pdf.fonttype':42,
                     'savefig.facecolor':'white'})


def fields(x,theta,t=0.,a=.25,ref=REF,comoving=True):
    """Return primary and epsilon*Re(secondary), in physical Pier velocity units."""
    x,theta=np.broadcast_arrays(x,theta)
    physical_theta=theta+OMEGA/BETA*t if comoving else theta
    phase=ALPHA*x+BETA*physical_theta-OMEGA*t
    exp=np.exp(1j*phase[...,None]*NS)
    primary=(exp@P[:,2,ref]).real
    secondary=(np.exp(1j*(AH*x+BH*physical_theta-WH*t))*(exp@S[:,2,ref])).real*a
    return primary,secondary


def save(fig,name):
    # Include full Chinese/math title glyph extents in the exported canvas.
    fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight',pad_inches=.13)
    fig.savefig(OUT/(name+'.png'),dpi=210,bbox_inches='tight',pad_inches=.13)
    plt.close(fig)


def field_comparison():
    x=np.linspace(-20,20,181)[:,None]
    th=np.deg2rad(np.linspace(0,36,1001))[None,:]
    p,s=fields(x,th)
    panels=[p,s,p+s]
    limit=max(np.max(abs(v)) for v in panels)/RMS
    fig,axes=plt.subplots(3,1,figsize=(24/2.54,17/2.54),layout='constrained')
    titles=['(a) 非线性饱和主波：完整 H=5 Fourier 场',
            '(b) 真实二次模态：幅值取参考主波法向速度 RMS 的 25%',
            '(c) 叠加后：相邻主波条带出现不同的增强、减弱和变形']
    for ax,q,title in zip(axes,panels,titles):
        im=ax.imshow(q/RMS,origin='lower',extent=[0,R*np.deg2rad(36),-20,20],
                     aspect='equal',cmap=CMAP,vmin=-limit,vmax=limit,interpolation='nearest')
        ax.set_title(title,loc='left',fontsize=11,pad=6)
        ax.set_yticks([-20,0,20],['490','510','530'])
        ax.set_ylabel(r'$r$')
        ticks=np.arange(0,37,6)
        ax.set_xticks(R*np.deg2rad(ticks),[str(v)+'°' for v in ticks])
        ax.set_xlabel(r'周向角 $\theta$；横向绘图距离为 $R\theta$')
    bar=fig.colorbar(im,ax=axes,orientation='horizontal',fraction=.05,pad=.03,aspect=45)
    bar.set_label(r'法向速度偏差 $(w-W_{\rm laminar})/w_{1,\mathrm{rms}}$；三幅使用同一色标')
    fig.suptitle(r'$R=510,\ \beta=68,\ \hat\beta=20,\ z=1.6,\ t=0$'+'\n'+
                 '局部环带展开：r、z 已用 δ 无量纲化；保留长度比例',fontsize=13)
    fig.get_layout_engine().set(h_pad=.08,hspace=.08)
    save(fig,'primary_plus_secondary')


def phase_explanation():
    theta=np.deg2rad(np.linspace(0,90,4001))
    deg=np.rad2deg(theta)
    p,s=fields(0,theta,a=.25)
    fig,axs=plt.subplots(4,1,figsize=(24/2.54,22/2.54),layout='constrained',
                         gridspec_kw={'height_ratios':[1,1,1.2,1.2]})
    axs[0].plot(deg,p/RMS,color='#0072b2',lw=1.3)
    axs[0].set_title('(a) 主波：90° 内有 17 个相同周期；整圆对应 β=68',loc='left',fontsize=11)
    axs[0].set_ylabel(r'$w_1/w_{1,\mathrm{rms}}$')
    axs[1].plot(deg,np.cos(BH*theta),color='#6551a4',lw=1.3,
                label=r'仅 Floquet 因子 $\cos(20\theta)$，不是完整二次场')
    j=np.arange(18)
    tj=360/BETA*j
    axs[1].scatter(tj,np.cos(2*np.pi*BH/BETA*j),color='#d55e00',s=18,zorder=3,
                   label='相邻主波的同一相位位置')
    axs[1].set_title('(b) 这 17 个主波之间，二次附加相位走了 5 圈',loc='left',fontsize=11)
    axs[1].set_ylabel('相位因子的实部')
    axs[1].legend(loc='upper center',bbox_to_anchor=(.5,1.0),ncol=2,frameon=False,fontsize=8.5)
    axs[1].set_ylim(-1.4,1.85)
    axs[2].plot(deg,p/RMS,color='#0072b2',lw=1.0,label='主波')
    axs[2].plot(deg,(p+s)/RMS,color='#d55e00',lw=1.0,label='主波 + 完整二次扰动')
    axs[2].set_title('(c) 真正叠加后的周向剖面：不能用一条 cos(20θ) 替代二次特征函数',loc='left',fontsize=11)
    axs[2].set_ylabel(r'$(w-W_{\rm laminar})/w_{1,\mathrm{rms}}$')
    axs[2].legend(frameon=False,ncol=2,loc='upper right',fontsize=8.5)
    for ax in axs[:3]:
        ax.set_xlim(0,90)
        ax.set_xticks(np.arange(0,91,15))
        ax.set_xlabel(r'$\theta$ (deg)')
        ax.grid(axis='x',alpha=.12)
    ell=BH+NS*BETA
    energies=np.array(DATA['sideband_energy_fraction'])*100
    colors=['#d55e00' if n==-1 else '#6551a4' if n==0 else '#9abbd0' for n in NS]
    axs[3].bar(np.arange(11),energies,color=colors,width=.72)
    axs[3].set_xticks(np.arange(11),[str(v) for v in ell])
    axs[3].set_ylim(0,79)
    axs[3].set_ylabel('二次模态能量份额 (%)')
    axs[3].set_xlabel(r'实际周向 Fourier 谐波 $\ell_n=\hat\beta+n\beta$（带符号）')
    axs[3].set_title(r'(d) 完整模态含多个周向谐波：最强的是 $\ell=-48$，$\hat\beta=20$ 是 Floquet 参数',loc='left',fontsize=11)
    for i in (4,5):
        axs[3].text(i,energies[i]+1.5,f'{energies[i]:.1f}%',ha='center')
    fig.suptitle(r'$\hat\beta$ 的直观含义：已有周期涡列上的附加相位变化'+'\n'+
                 r'$\Delta\theta_1=360^\circ/68=5.294^\circ,\quad'
                 r'\Delta\psi=360^\circ(20/68)=105.882^\circ$',fontsize=13)
    save(fig,'azimuthal_phase_explained')


def animate():
    x=np.linspace(-20,20,101)[:,None]
    th=np.deg2rad(np.linspace(0,36,621))[None,:]
    times=np.linspace(0,2,49)
    p,s=fields(x,th,t=0,a=.06)
    shape=np.exp(1j*(AH*x+BH*th))*(np.exp(1j*(ALPHA*x+BETA*th)[...,None]*NS)@S[:,2,REF])
    # Time-independent upper bound avoids clipping at any intermediate phase.
    limit=(np.max(abs(p))+.06*np.exp(WH.imag*times[-1])*np.max(abs(shape)))/RMS*1.02
    fig,axs=plt.subplots(3,1,figsize=(9,6),layout='constrained')
    ims=[]
    for ax,title in zip(axs,['非线性饱和主波（随主波旋转观察）','线性二次扰动','两者叠加']):
        im=ax.imshow(p/RMS,origin='lower',extent=[0,R*np.deg2rad(36),-20,20],
                     aspect='equal',cmap=CMAP,vmin=-limit,vmax=limit,interpolation='nearest')
        ims.append(im)
        ax.set_title(title,loc='left',fontsize=10)
        ticks=np.arange(0,37,6)
        ax.set_xticks(R*np.deg2rad(ticks),[str(v)+'°' for v in ticks])
        ax.set_yticks([-20,0,20],['490','510','530'])
        ax.set_ylabel(r'$r$')
    axs[-1].set_xlabel(r"随主波旋转的周向角 $\theta'=\theta-(50.5/68)t$")
    cbar=fig.colorbar(ims[0],ax=axs,orientation='horizontal',fraction=.05,pad=.03,aspect=40)
    cbar.set_label(r'$(w-W_{\rm laminar})/w_{1,\mathrm{rms}}$；整个动画共用固定色标',fontsize=10)
    title=fig.suptitle('',fontsize=12)
    def update(i):
        t=times[i]
        p,s=fields(x,th,t=t,a=.06)
        for im,q in zip(ims,[p,s,p+s]):
            im.set_data(q/RMS)
        title.set_text('β=68 主波 + β̂=20 二次模态 | t='+f'{t:.2f}'+
                       ' | 二次幅值增长倍数='+f'{np.exp(WH.imag*t):.2f}'+'\n'+
                       '初始参考 RMS 比 6%；仅线性模态叠加，不模拟二次饱和')
        return ims+[title]
    update(0)
    fig.canvas.draw()
    fig.set_layout_engine('none')
    class CleanPaletteWriter(PillowWriter):
        def finish(self):
            palette=Image.new('P',(1,1))
            colors=[int(round(v*255)) for c in CMAP(np.linspace(0,1,240)) for v in c[:3]]
            colors += [int(v) for shade in np.linspace(0,255,16) for v in [shade]*3]
            palette.putpalette(colors)
            frames=[f.convert('RGB').quantize(palette=palette,dither=0) for f in self._frames]
            frames[0].save(self.outfile,save_all=True,append_images=frames[1:],duration=round(1000/self.fps),
                           loop=0,optimize=False,disposal=2)
    FuncAnimation(fig,update,frames=len(times),interval=100,blit=False).save(
        OUT/'secondary_growth.gif',writer=CleanPaletteWriter(fps=10),dpi=100)
    plt.close(fig)


def html():
    template=(ROOT/'examples/secondary_visualization.html').read_text()
    template=template.replace('__MODE_DATA__',json.dumps(DATA,ensure_ascii=False))
    for key,name in [('__PHASE_FIGURE__','azimuthal_phase_explained.png')]:
        template=template.replace(key,'data:image/png;base64,'+
                    base64.b64encode((OUT/name).read_bytes()).decode())
    (OUT/'index.html').write_text(template)


def export_samples():
    indices=[0,639,50*640+320,99*640,99*640+639]
    settings=[(3,0,.25,True,True),(3,.6,.12,True,True),(3,.6,.12,False,True),
              (3,.6,.12,True,False),(0,1.1,.08,False,True)]
    cases=[]
    for z,t,a,moving,growth in settings:
        x=np.array([20-40*(i//640)/99 for i in indices])
        th=np.array([np.pi/5*(i%640)/639 for i in indices])
        p,s=fields(x,th,t=t,a=a*(1 if growth else np.exp(-WH.imag*t)),ref=z,comoving=moving)
        cases.append(dict(z=z,t=t,a=a,moving=moving,growth=growth,indices=indices,
                          primary=(p/RMS).tolist(),secondary=(s/RMS).tolist()))
    (OUT/'field_samples.json').write_text(json.dumps(cases,indent=2)+'\n')


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=True)
    field_comparison()
    phase_explanation()
    animate()
    html()
    export_samples()
    print('Saved figures, GIF and offline HTML to',OUT)
