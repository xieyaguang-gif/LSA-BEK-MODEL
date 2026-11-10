#!/usr/bin/env python3
"""General-Ro single-beta full-BVP linear campaign. No nonlinear harmonics."""
from __future__ import annotations
import argparse,gc,hashlib,json,os,resource,sys,time,traceback
from dataclasses import dataclass
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/ro_scan_20261101/S1';S0=OUT.parent/'S0';REF=50.5
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'examples')]
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','BLIS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMEXPR_MAX_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
_,hard=resource.getrlimit(resource.RLIMIT_AS);resource.setrlimit(resource.RLIMIT_AS,(min(200_000_000_000,hard) if hard>=0 else 200_000_000_000,hard))
import numpy as np
from scipy import linalg,optimize
from scipy.interpolate import CubicSpline,PchipInterpolator
from scipy.sparse.linalg import LinearOperator,eigs,splu
from llsa import solve_bek
from llsa.secondary.discretization import rotation_rates
from llsa.secondary.operators import PierOperators
from llsa.secondary.radial import RadialConfig,radial_matrices
from llsa.secondary.global_radial import GlobalRadial,OutletSponge
from appelquist2015_hopf_common import complex_block,mass_norm,smooth_step_and_derivative
from fullbvp_phaseA_global import interpolate_mode,common_correlation
from fullbvp_phaseA_spatial import Frozen as LegacyFrozen
from roscan_absolute import CODE as S0CODE,ROS,tag,dump,pair
CODE=hashlib.sha256(b''.join(p.read_bytes() for p in [Path(__file__),ROOT/'examples/fullbvp_phaseA_global.py',ROOT/'examples/fullbvp_phaseA_spatial.py',ROOT/'examples/appelquist2015_hopf_common.py',*sorted((ROOT/'src/llsa').rglob('*.py'))])).hexdigest()
DEADLINE=None
class BudgetExceeded(RuntimeError):pass
class TrackingError(RuntimeError):pass
def check_budget():
 if DEADLINE and time.monotonic()>DEADLINE:raise BudgetExceeded('S1 stage wall budget reached; checkpoints preserved')
def s0row(ro):
 path=S0/'absolute_table.json'
 if not path.exists():return None
 table=json.loads(path.read_text())
 if table.get('code')!=S0CODE:return None
 return next((r for r in table['rows'] if abs(r['ro']-ro)<1e-10),None)
def plan_row(ro):
 row=s0row(ro);valid=bool(row and row['status']=='numerically_validated');rca=row['R_ca'] if row and row.get('R_ca') else 507.389019046/abs(ro);s=rca/507.389019046;rin=400*s;ranges=[690*s,810*s];grids=[]
 for n,dr in [(32,.5),(40,.25)]:
  intervals=round((750*s-rin)/dr);ratio=intervals/1463*((n+1)/41)**2;rss=9.34*ratio;sec=51.65*intervals/1463*((n+1)/41)**3
  grids.append(dict(n=n,dr_requested=dr,intervals_fixed_during_root=intervals,dr_actual_range=[(v-rin)/intervals for v in ranges],complex_dof=(intervals+1)*4*(n+1),measured_peak_RSS_GiB_reference=9.34,estimated_peak_RSS_GiB=rss,conservative_address_space_GB=3*rss*1.073741824,estimated_eigensolve_seconds=sec,execution_allowed=3*rss*1.073741824<200))
 expected_solves_per_grid=8 if ro==-1 else 24
 nominal=sum(g['estimated_eigensolve_seconds'] for g in grids)*expected_solves_per_grid+742.5*s
 # New-Ro homotopy, independent spectral search, root refinement and two grids need margin.
 seconds=nominal+100+1011.5*(0 if ro==-1 else max(.4,abs(ro+1)/.249));upper=seconds*1.65
 calibration_path=OUT/'cost_calibration.json';calibration_hash=None
 if calibration_path.exists() and valid:
  calibration=json.loads(calibration_path.read_text());measured=calibration.get('targets',{}).get(tag(ro));proof=hashlib.sha256((ROOT/row['source']).read_bytes()).hexdigest()
  if measured and measured['beta_c']==row['beta_c'] and abs(measured['R_ca']/rca-1)<1e-10 and measured['S0_validator_code']==S0CODE and measured['S0_evidence_sha256']==proof:
   seconds=measured['nominal_seconds'];upper=measured['conservative_seconds'];calibration_hash=hashlib.sha256(calibration_path.read_bytes()).hexdigest()
 return dict(ro=ro,status='ready' if valid else 'blocked_by_S0',beta_c=row.get('beta_c') if valid else None,S0_source=row.get('source') if row else None,S0_evidence_sha256=hashlib.sha256((S0.parent.parent.parent/row['source']).read_bytes()).hexdigest() if row and row.get('source') and (S0.parent.parent.parent/row['source']).exists() else None,frequency_convention='negative-Ro raw-to-inertial-v1: Omega=beta*disk-Re*conj(raw); alpha=conj(raw)',expected_solves_per_grid=expected_solves_per_grid,R_ca=rca,R_ca_is_estimate=not valid,R_ca_basis='validated S0 minimum' if valid else 'unvalidated candidate or 507.389/absRo planning estimate',scale=s,Rin=rin,Rout_scan_range=ranges,Rout_initial=720.4698608*s,sponge=dict(distance_to_outlet=100*s,width=40*s,strength=28),inlet_damping=dict(strength=.8,zero_damping_radius=420*s,width=10*s,formula='.8*S((420*scale-R)/(10*scale))'),grids=grids,estimated_total_seconds=seconds,conservative_total_seconds=upper,cost_calibration_sha256=calibration_hash,launch_by_executor=valid and upper<=3600 and all(g['execution_allowed'] for g in grids),memory_policy='one global process at a time; <200 GB address space and <400 GB task total',long_domain_risk='radial intervals grow with R_ca; dr held near .25/.5 rather than fixing intervals across Ro; root search fixes intervals within each Ro/grid',if_nominal_grid_exceeds_cap='Do not start N40; stage N32/dr=.5 first, or use distributed/supercomputer factorization. A coarse solve cannot establish final convergence.')
def prepare():
 OUT.mkdir(parents=True,exist_ok=True);rows=[plan_row(ro) for ro in ROS]
 for row in rows:
  dest=OUT/tag(row['ro']);dest.mkdir(exist_ok=True);flow=solve_bek(row['ro'],zmax=40,tol=1e-10,samples=16001);np.savez_compressed(dest/'baseflow.npz',z=flow.z,state=flow.state,ro=row['ro'],source_hash=CODE)
 dump(OUT/'plan.json',dict(schema='roscan-linear-v1',code=CODE,S0_code=S0CODE,rows=rows,resource_anchors=dict(source='results/ro0751_global_20261101/phaseA/beta85/logs/*.resources.txt',seed_seconds=1011.50,scan_seconds=306.74,neutral_seconds=423.69,spatial_seconds=742.50,spectrum_seconds=100,nominal_eigensolve_seconds=51.65,neutral_peak_RSS_GiB=9.34),estimate_uncertainty='Factorization model proportional to radial intervals times wall-normal bandwidth squared for memory/cubed for time; total upper estimate factor1.65 includes unmeasured homotopy failures. Not a memory guarantee.'))
 report()
@dataclass(frozen=True)
class ScaledSponge(OutletSponge):
 scale:float=1.
 def profile(self,radii):
  r=np.asarray(radii,float);s=self.scale
  return self.strength*smooth_step_and_derivative((r-self.start)/self.width)[0]+.8*smooth_step_and_derivative((420*s-r)/(10*s))[0]
 def start_derivative(self,radii):return -self.strength/self.width*smooth_step_and_derivative((np.asarray(radii)-self.start)/self.width)[1]
def build(ro,beta,n,intervals,rout,scale,omega=REF):
 return GlobalRadial(np.linspace(400*scale,float(rout),int(intervals)+1),RadialConfig(ro=ro,beta=beta,n=n,harmonics=1,omega=omega,nonlinear=False),outlet_alpha=0j,outlet_sponge=ScaledSponge(rout-100*scale,40*scale,28.,False,scale))
def load_state(path,legacy=False):
 with np.load(path,allow_pickle=False) as a:
  ro=float(a['ro']) if 'ro' in a else (-1. if legacy else None);beta=float(a['beta']) if 'beta' in a else (68. if legacy else None)
  if ro is None or beta is None:raise ValueError('Missing Ro/beta metadata')
  return dict(q=(a['q'] if 'q' in a else a['q1']).reshape(len(a['radii']),4,len(a['xi'])),radii=a['radii'].copy(),xi=a['xi'].copy(),ro=ro,beta=beta,omega=float(a['omega_c']),sigma=float(a['sigma_residual']),rout=float(a['radii'][-1]),scale=float(a['scale']) if 'scale' in a else 1.,source=str(path))
def save_state(path,state,key):
 path.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(path,q=state['q'],radii=state['radii'],xi=state['xi'],omega_c=state['omega'],sigma_residual=state['sigma'],ro=state['ro'],beta=state['beta'],scale=state['scale'],cache_key=key,code=CODE)
def solve(ro,beta,n,intervals,rout,scale,seed,prefix,k=8,allow_competitors=False):
 check_budget();prefix=Path(prefix);jpath=Path(str(prefix)+'.json');npath=Path(str(prefix)+'.npz');config=dict(ro=ro,beta=beta,n=n,intervals=int(intervals),Rin=400*scale,Rout=rout,scale=scale,boundary='homogeneous inlet; outlet_alpha=0; Pier wall/farfield',sponge=[rout-100*scale,40*scale,28],inlet=[420*scale,10*scale,.8],code=CODE,S0_code=S0CODE,S0_evidence_sha256=(plan_row(ro).get('S0_evidence_sha256') if abs(ro-round(ro,1))<1e-8 else None),frequency_convention='negative-Ro raw-to-inertial-v1',seed_source=seed['source'],seed_omega=seed['omega'],seed_sigma=seed['sigma']);key=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
 if jpath.exists() and npath.exists():
  rec=json.loads(jpath.read_text())
  if rec.get('cache_key')==key:
   state=load_state(npath)
   if state['ro']!=ro or state['beta']!=beta:raise RuntimeError('Cache Ro/beta mismatch')
   return rec,state
 begin=time.perf_counter();p=build(ro,beta,n,intervals,rout,scale);A,M=complex_block(p,1);target=complex(-seed['sigma'],seed['omega']-REF);shift=target+.025+.035j;lu=splu(A-shift*M);K=LinearOperator(A.shape,matvec=lambda x:lu.solve(M@x),dtype=complex);v0=interpolate_mode(seed,p).ravel();v0/=linalg.norm(v0);mu,vec=eigs(K,k=k,ncv=max(24,2*k+4),which='LM',tol=3e-11,maxiter=1600,v0=v0);candidates=[]
 for i,lam in enumerate(shift+1/mu):
  q=vec[:,i];res=linalg.norm(A@q-lam*(M@q))/(linalg.norm(A@q)+abs(lam)*linalg.norm(M@q));candidates.append(dict(index=i,omega=float(REF+lam.imag),sigma=float(-lam.real),residual=float(res),correlation=common_correlation(seed,p,q),distance=float(abs(lam-target))))
 good=[c for c in candidates if c['residual']<1e-7 and c['correlation']>=.85 and c['distance']<=.3]
 if not good:
  dump(Path(str(prefix)+'_rejected.json'),dict(configuration=config,candidates=candidates));raise TrackingError('Mode identity/spectral gate failed')
 best=max(good,key=lambda c:c['correlation']);competitors=[c for c in candidates if c['residual']<1e-7 and c['sigma']>best['sigma']+1e-4 and abs(complex(c['omega']-best['omega'],c['sigma']-best['sigma']))>.01]
 if competitors and not allow_competitors:
  dump(Path(str(prefix)+'_competitors.json'),dict(configuration=config,candidates=candidates));raise TrackingError('A more unstable distinct mode requires branch review')
 q=vec[:,best['index']].copy();q/=mass_norm(p,q);vel=q.reshape(len(p.radii),4,p.ops[0].m)[:,:3];q*=np.exp(-1j*np.angle(vel.ravel()[np.argmax(abs(vel))]));state=dict(q=q.reshape(len(p.radii),4,p.ops[0].m),radii=p.radii.copy(),xi=p.ops[0].grid.xi.copy(),ro=ro,beta=beta,omega=best['omega'],sigma=best['sigma'],rout=rout,scale=scale,source=str(npath));rec=dict(**config,rout=rout,cache_key=key,omega=best['omega'],omega_Omega_d=best['omega']/rotation_rates(ro)[0],sigma=best['sigma'],correlation=best['correlation'],residual=best['residual'],candidates=candidates,seconds=time.perf_counter()-begin,peak_RSS_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,dr_actual=(rout-400*scale)/intervals)
 save_state(npath,state,key);dump(jpath,rec);del p,A,M,lu,K,vec;gc.collect();return rec,state
class Frozen(LegacyFrozen):
 def __init__(self,r,n,ro,beta,kind='global'):
  self.r,self.n,self.kind=float(r),n,kind;self.op=op=PierOperators(radius=r,beta=beta,n=n,ro=ro)
  if kind=='global':self.A,self.B,self.C=radial_matrices(op,1,0.)
  else:
   self.A=op.linear(0.,beta);plus,minus=op.linear(1.,beta),op.linear(-1.,beta);self.B=(plus-minus)/(2j);self.C=-(plus+minus-2*self.A)/2
  self.M,self.mass=op.M,op.grid.mass
def crossings(r,y):return [float(r[i]-y[i]*(r[i+1]-r[i])/(y[i+1]-y[i])) for i in range(len(r)-1) if y[i]*y[i+1]<0]
def radial_end(row,n,state):
 ro=row['ro'];beta=row['beta_c'];scale=row['scale'];s0=s0row(ro);a=complex(*s0['alpha']).conjugate();w=complex(*s0['omega_BEK']);previous=None;curve=[];rlo=550*scale;rhi=630*scale
 for r in np.r_[row['R_ca'],np.arange(rlo,rhi+scale,2*scale)]:
  check_budget();f=Frozen(r,n,ro,beta);rec,q=f.saddle(a,w)
  if previous is not None and f.correlation(previous,q)<.85:raise TrackingError('Local saddle mode identity lost')
  previous=q;a=complex(*rec['alpha']);w=complex(*rec['omega']);curve.append(rec)
 # The first point seeds the new-Ro local branch and is excluded from crossing interpolation.
 curve=curve[1:];op=PierOperators(radius=row['R_ca'],beta=beta,n=n,ro=ro);r=state['radii'];v=state['q'][:,:3];energy=np.einsum('rvi,ij,rvj->r',v.conj(),op.grid.mass,v).real;use=(r>=rlo)&(r<=rhi)
 if np.any(energy[use]<=1e-290):raise RuntimeError('Mode amplitude too small in diagnostic window')
 log=np.log(np.maximum(energy,1e-300));gE=.5*np.gradient(log,r,edge_order=2);gEc=.5*CubicSpline(r,log).derivative()(r);cr=np.array([x['r'] for x in curve]);cg=np.array([x['growth'] for x in curve]);rr=r[use];variants=dict(linear=np.interp(rr,cr,cg),pchip=PchipInterpolator(cr,cg)(rr),cubic=CubicSpline(cr,cg)(rr),step4=np.interp(rr,cr[::2],cg[::2]));cross={k:crossings(rr,gE[use]-value) for k,value in variants.items()};cross['global_derivative_cubic']=crossings(rr,gEc[use]-variants['linear'])
 middle=curve[len(curve)//2];pin=Frozen(middle['r'],n,ro,beta).pinch(middle,steps=90)
 if any(len(x)!=1 for x in cross.values()) or not pin['opposite_half_planes_at_end']:status='partial'
 else:status='numerically_validated'
 return dict(n=n,ro=ro,beta=beta,code=CODE,status=status,definition='gE=0.5*d log(integral |q_velocity|^2 dz)/dR intersects -Im(alpha0) of physical full-cylindrical frozen saddle; same phaseA linear r_end diagnostic',crossings=cross,r_end=cross['linear'][0] if len(cross['linear'])==1 else None,Re_end=abs(ro)*cross['linear'][0] if len(cross['linear'])==1 else None,local_curve=curve,pinch=pin,rout_c=state['rout'],global_omega_BEK=state['omega'],global_omega_Omega_d=state['omega']/rotation_rates(ro)[0])
def seed_stage(row):
 ro=row['ro'];beta=row['beta_c'];scale=row['scale'];dest=OUT/tag(ro)/'seed';old=ROOT/'results/fullbvp_global_20261030/phaseA/neutral/N40_dr0.25/neutral_state.npz';seed=load_state(old,True);n=32;intervals=round((row['Rout_initial']-row['Rin'])/.5);records=[]
 def bridge(state,nro,nbeta,nscale,depth=0):
  check_budget();s=dict(state);factor=nscale/state['scale'];s['radii']=state['radii']*factor;s['rout']=state['rout']*factor;od=rotation_rates(state['ro'])[0];nd=rotation_rates(nro)[0];s['omega']=nbeta*nd+(state['omega']-state['beta']*od)*(abs(nro)*nscale)/(abs(state['ro'])*state['scale']);s['source']=state['source']+f'|initial_guess_transform_to_{nro}_{nbeta}_{nscale}'
  prefix=dest/f'ro{nro:.8f}_beta{nbeta:.8f}_scale{nscale:.8f}'
  try:r,new=solve(nro,nbeta,n,intervals,720.4698608*nscale,nscale,s,prefix)
  except TrackingError:
   if depth>=8:raise
   mid=bridge(state,(state['ro']+nro)/2,(state['beta']+nbeta)/2,(state['scale']+nscale)/2,depth+1);return bridge(mid,nro,nbeta,nscale,depth+1)
  records.append(r);return new
 steps=max(1,int(np.ceil(abs(ro+1)/.025)),int(np.ceil(abs(beta-68)/.5)))
 for t in np.linspace(0,1,steps+1)[1:]:seed=bridge(seed,-1+t*(ro+1),68+t*(beta-68),1+t*(scale-1))
 dump(dest/'summary.json',dict(code=CODE,method='accepted Ro=-1 mode only as initial guess; explicit radius and disk-frequency rescaling then adaptive Ro/beta continuation; each solve retains velocity overlap>=.85 and spectral distance<=.3',records=records,final_source=seed['source']));return seed
def neutral_stage(row,seed):
 dest=OUT/tag(row['ro']);grids=[];states=[]
 for grid in row['grids']:
  if not grid['execution_allowed']:raise RuntimeError('Conservative memory estimate reaches 200 GB; grid blocked')
  n=grid['n'];intervals=grid['intervals_fixed_during_root'];cache={};gdest=dest/f'N{n}'
  def follow_radius(initial,target,depth=0):
   prefix=gdest/f'Rout_{target:.9f}'
   try:return solve(row['ro'],row['beta_c'],n,intervals,target,row['scale'],initial,prefix)
   except TrackingError:
    if depth>=10 or abs(target-initial['rout'])<.02*row['scale']:raise
    mid=.5*(initial['rout']+target);_,middle=follow_radius(initial,mid,depth+1);return follow_radius(middle,target,depth+1)
  def evaluate(rout):
   nonlocal seed
   rout=float(rout)
   if rout in cache:return cache[rout][0]['sigma']
   near=min(cache,key=lambda x:abs(x-rout)) if cache else None;initial=cache[near][1] if near is not None else seed
   rec,state=follow_radius(initial,rout);cache[rout]=(rec,state);seed=state;return rec['sigma']
  center=row['Rout_initial'];s=row['scale'];radii=[center]
  for delta in [1,2,4,8,16,32,64,96]:
   radii.extend([max(row['Rout_scan_range'][0],center-delta*s),min(row['Rout_scan_range'][1],center+delta*s)])
  radii=list(dict.fromkeys(radii))
  bracket=None
  for rout in radii:
   evaluate(rout);keys=sorted(cache)
   changes=[(a,b) for a,b in zip(keys[:-1],keys[1:]) if cache[a][0]['sigma']*cache[b][0]['sigma']<0]
   if len(changes)>1:raise RuntimeError('Multiple neutral brackets; branch review required')
   if changes:bracket=changes[0];break
  if bracket is None:raise RuntimeError('No neutral root in finite Rout scan; checkpoint evidence retained')
  root=optimize.brentq(evaluate,*bracket,xtol=2e-6);evaluate(root);rec,state=cache[root];save_state(gdest/'neutral_state.npz',state,rec['cache_key']);dump(gdest/'neutral.json',rec);grids.append(rec);states.append(state)
 dump(dest/'neutral_summary.json',dict(code=CODE,grids=grids,topology='fixed radial interval count per grid throughout Rout root; rebuild all radial stencils and boundaries'))
 return grids,states
def spectrum(row,state):
 check_budget();grid=row['grids'][-1];p=build(row['ro'],row['beta_c'],grid['n'],grid['intervals_fixed_during_root'],state['rout'],row['scale']);A,M=complex_block(p,1);candidates=[]
 for offset in [-8.,-4.,0.,4.,8.]:
  check_budget();shift=complex(.025,state['omega']+offset-REF+.035);lu=splu(A-shift*M);K=LinearOperator(A.shape,matvec=lambda x:lu.solve(M@x),dtype=complex);mu,vec=eigs(K,k=8,ncv=24,which='LM',tol=3e-11,maxiter=1600)
  for j,lam in enumerate(shift+1/mu):
   q=vec[:,j];res=linalg.norm(A@q-lam*(M@q))/(linalg.norm(A@q)+abs(lam)*linalg.norm(M@q));candidates.append(dict(omega=float(REF+lam.imag),sigma=float(-lam.real),residual=float(res),correlation=common_correlation(state,p,q),center_offset=offset))
  del lu,K,vec;gc.collect()
 suspicious=[r for r in candidates if r['residual']<1e-7 and r['sigma']>.002 and abs(complex(r['omega']-state['omega'],r['sigma']))>.01];result=dict(code=CODE,scope='finite five centers +/-8 around continued global frequency; not exhaustive spectrum',candidates=candidates,suspicious=suspicious,passed=not suspicious);dump(OUT/tag(row['ro'])/'spectrum.json',result);return result

def run(ro,regression=False,supervisor_long=False):
 row=plan_row(ro)
 if regression and ro!=-1:raise ValueError('regression entry only accepts Ro=-1')
 if row['conservative_total_seconds']>3600 and not supervisor_long:raise RuntimeError('Estimated duration exceeds one hour; only supervisor may launch with --supervisor-long')
 if row['status']!='ready':raise RuntimeError('blocked_by_S0: requires current source-hash, actual beta minimum, branch pinch and both convergence checks')
 dest=OUT/tag(ro);dest.mkdir(parents=True,exist_ok=True);start=time.perf_counter();seed=seed_stage(row);grids,states=neutral_stage(row,seed);spec=spectrum(row,states[-1]);ends=[]
 for grid,state in zip(row['grids'],states):
  end=radial_end(row,grid['n'],state);ends.append(end);dump(dest/f'rend_N{grid["n"]}.json',end)
 result=make_summary(row,grids,ends,spec,time.perf_counter()-start)
 dump(dest/'summary.json',result);report()
def make_summary(row,grids,ends,spec,seconds,producer_code=None):
 ro=row['ro'];rout=lambda g:g.get('rout',g.get('Rout'))
 result=dict(ro=ro,beta=row['beta_c'],code=CODE,producer_code=producer_code or CODE,status='numerically_validated' if spec['passed'] and all(e['status']=='numerically_validated' for e in ends) else 'partial',rout_c=rout(grids[-1]),global_omega_BEK=grids[-1]['omega'],global_omega_Omega_d=grids[-1]['omega_Omega_d'],r_end=ends[-1]['r_end'],Re_end=ends[-1]['Re_end'],grid_difference_r_end=abs(ends[-1]['r_end']-ends[0]['r_end']) if all(e['r_end'] is not None for e in ends) else None,grid_difference_Rout_c=abs(rout(grids[-1])-rout(grids[0])),grid_difference_omega=abs(grids[-1]['omega']-grids[0]['omega']),seconds=seconds,peak_RSS_kb=max(g.get('peak_RSS_kb',0) for g in grids),reference_r_end=594.19 if ro==-1 else None,S0_code=S0CODE,S0_evidence_sha256=row['S0_evidence_sha256'])
 if ro==-1 and result['r_end'] is not None:result['difference_from_594p19']=result['r_end']-594.19
 return result
def finalize_existing(ro):
 if ro is None:raise ValueError('--ro required for finalize')
 row=plan_row(ro)
 if row['status']!='ready':raise RuntimeError('Finalization requires current verified S0 evidence')
 dest=OUT/tag(ro);ledger=json.loads((OUT/'source_compatibility.json').read_text());snapshot=ROOT/ledger['producer_source']
 if hashlib.sha256(snapshot.read_bytes()).hexdigest()!=ledger['producer_source_sha256']:raise RuntimeError('Producer snapshot hash mismatch')
 neutral=json.loads((dest/'neutral_summary.json').read_text());spec=json.loads((dest/'spectrum.json').read_text());producer=ledger['producer_code']
 if neutral['code']!=producer or spec['code']!=producer:raise RuntimeError('Producer evidence mismatch')
 grids=neutral['grids'];ends=[json.loads((dest/f'rend_N{g["n"]}.json').read_text()) for g in grids]
 for g,e in zip(grids,ends):
  rout=g.get('rout',g.get('Rout'))
  if not (g['ro']==ro and g['beta']==row['beta_c'] and g['code']==producer and g['S0_code']==S0CODE and g['S0_evidence_sha256']==row['S0_evidence_sha256'] and g['residual']<1e-7 and abs(g['sigma'])<1e-6 and e['n']==g['n'] and e['rout_c']==rout and e['global_omega_BEK']==g['omega']):raise RuntimeError('Neutral/diagnostic identity or source evidence check failed')
 result=make_summary(row,grids,ends,spec,sum(g['seconds'] for g in grids),producer);result.update(postprocess_only_finalization=True,numerical_solves_repeated=False,producer_source_snapshot=ledger['producer_source'],source_compatibility='results/ro_scan_20261101/S1/source_compatibility.json',resolved_failure_artifact='results/ro_scan_20261101/S1/ro_m1p000/assembly_failure_v1.json')
 failure=dest/'stage_failure.json'
 if failure.exists():failure.rename(dest/'assembly_failure_v1.json')
 dump(dest/'summary.json',result);report();print(json.dumps(result))

def preflight():
 records=[]
 for ro in [-1.,-.9]:
  row=plan_row(ro);s=row['scale'];p=build(ro,68,8,30,720*s,s);A,M=complex_block(p,1);p2=build(ro,68,8,30,720*s,s,REF+.125);A2,_=complex_block(p2,1);defect=float(abs(A2-A+.125j*M).max());sp=ScaledSponge(620*s,40*s,28.,False,s);v=sp.profile(np.array([400,420,620,660])*s);records.append(dict(ro=ro,frequency_derivative_defect=defect,inlet_sponge_profile=v.tolist(),passed=bool(defect<1e-10 and v[1]==0 and v[2]==0 and abs(v[3]-28)<1e-9)))
 dump(OUT/'preflight.json',dict(code=CODE,records=records,passed=all(r['passed'] for r in records)));print(json.dumps(dict(preflight_passed=all(r['passed'] for r in records))))
def report():
 rows=[plan_row(ro) for ro in ROS];lines=['# S1 general-Ro full-BVP preparation','','S1 global eigenvalues and r_end remain separate from local S0 absolute frequencies.','','| Ro | S0 gate | beta | Rin | Rout range | upper time / h |','|---:|---|---:|---:|---|---:|']
 for r in rows:lines.append(f"| {r['ro']} | {r['status']} | {r['beta_c']} | {r['Rin']:.2f} | {r['Rout_scan_range']} | {r['conservative_total_seconds']/3600:.2f} |")
 lines+=['','All 10 Ro values have new cached base flows; commands require current validated S0 input.','The inlet and outlet damping both move with R_ca/507.389; dr remains resolved and intervals increase with domain length.','Cost anchors: old beta85 campaign seed1011.50s, scan306.74s, neutral423.69s, spatial742.50s, spectrum100s; neutral peak9.34GiB.','New-Ro continuation and failure margin determine the one-hour launch gate. Ro=-0.9 above one hour is prepared for supervisor launch.','For |Ro|<=0.3, mesh length and factorization memory may become limiting. Conservative address-space estimates >=200GB block the affected grid.','The accepted linear r_end definition intersects the global energy growth with the local full-cylindrical saddle growth, with phaseA interpolation variants and finite Briggs checks.','Ro=-1 old r_end594.19 is a source reference until a new summary.json explicitly records recomputation.']
 for ro in [-1.,-.9]:
  p=OUT/tag(ro)/'summary.json'
  if p.exists():lines+=['',json.dumps(json.loads(p.read_text()),ensure_ascii=False)]
 (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
def main():
 global DEADLINE
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=['prepare','preflight','all','regression','finalize','report']);p.add_argument('--ro',type=float);p.add_argument('--budget',type=float,default=18000);p.add_argument('--supervisor-long',action='store_true');a=p.parse_args();OUT.mkdir(parents=True,exist_ok=True);DEADLINE=time.monotonic()+a.budget
 if a.stage=='prepare':prepare()
 elif a.stage=='preflight':preflight()
 elif a.stage=='report':report()
 elif a.stage=='finalize':finalize_existing(a.ro)
 else:
  if a.ro is None:raise ValueError('--ro required')
  try:run(a.ro,regression=a.stage=='regression',supervisor_long=a.supervisor_long)
  except Exception as ex:dump(OUT/tag(a.ro)/'stage_failure.json',dict(error=str(ex),traceback=traceback.format_exc(),code=CODE));raise
if __name__=='__main__':main()
