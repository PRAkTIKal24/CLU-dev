import os, numpy as np, jax, jax.numpy as jnp
from chlu.core.emission_head import EmissionHead, pretrain_emission_head
g=lambda k,d: float(os.environ.get(k,d))
crowd=g('CROWD',0); rho=g('RHO',2.0); steps=int(g('STEPS',400))
dmin=g('DMIN',0.05); dmax=g('DMAX',3.0); wmin=g('WMIN',0.15); wmax=g('WMAX',0.8)
lr=g('LR',3e-3); rw=g('RW',1.0); magg=os.environ.get('MAGG','mean'); aw=g('AW',0.0); am=g('AM',0.15)
d_safe=0.1238; addr,m,dim=8,1,9
rng=np.random.default_rng(0)
phi=rng.normal(size=(256,addr)); phi/=np.maximum(np.linalg.norm(phi,axis=1,keepdims=True),1e-9)
phi*=rng.uniform(0.2,1.0,size=(256,1)); pay=rng.uniform(-0.5,0.5,size=(256,1))
head=EmissionHead(addr_dim=addr,payload_dim=m,key=jax.random.PRNGKey(9001),hidden=64,layers=2,
                  width_min=wmin,width_max=wmax,depth_min=dmin,depth_max=dmax,center_skip_gain=g('SKIP',1.0))
lk=dict(n_perturb=32,sigma_addr=0.25,sigma_pay=0.6,margin=0.15,barrier=0.2,min_agg=magg)
if crowd>0: lk.update(crowd_weight=crowd,crowd_d_safe=d_safe)
h,hist=pretrain_emission_head(head,phi,pay,jax.random.PRNGKey(5150),dim=dim,confine=0.05,addr_dim=addr,
    steps=steps,batch=16,lr=lr,reach_weight=rw,reach_rho=rho,attr_weight=aw,attr_margin=am,loss_kwargs=lk)
p=h.emit(jnp.asarray(phi[:16],dtype=jnp.float32),jnp.asarray(pay[:16],dtype=jnp.float32))
C=np.asarray(p.centers)[:,:addr]; D=np.linalg.norm(C[:,None]-C[None],axis=-1); np.fill_diagonal(D,np.inf)
gap=np.linalg.norm(C-phi[:16],axis=1)
PP=np.linalg.norm(phi[:16,None]-phi[None,:16],axis=-1); np.fill_diagonal(PP,np.inf)
print(f"  phi NN spacing med {np.median(np.min(PP,axis=1)):.4f}")
print(f"crowd={crowd} rho={rho} st={steps} D[{dmin},{dmax}] W[{wmin},{wmax}] lr={lr} rw={rw} magg={magg} aw={aw} am={am}")
print(f"  loss {hist['loss_first']:.3f}->{hist['loss_last']:.4f} | sep min {D.min():.4f} med {np.median(np.min(D,axis=1)):.4f} | below d_safe {int(np.sum(np.min(D,axis=1)<d_safe))}/16")
print(f"  depth {np.asarray(p.depths).min():.3f}/{np.median(np.asarray(p.depths)):.3f}/{np.asarray(p.depths).max():.3f} | width {np.asarray(p.widths).min():.3f}/{np.median(np.asarray(p.widths)):.3f}/{np.asarray(p.widths).max():.3f} | |c-phi| med {np.median(gap):.3f} | |c| med {np.median(np.linalg.norm(C,axis=1)):.3f}")
