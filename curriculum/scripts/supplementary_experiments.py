"""Small CPU experiments for supplementary reading. No external package/model download.

Outputs measure analytic controls and synthetic clips, not pretrained quality.
"""
import json,math,os,platform
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'curriculum/artifacts/supplementary';OUT.mkdir(parents=True,exist_ok=True)
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'mpl-cache'))
import matplotlib.pyplot as plt

def parameterizations():
    generator=torch.Generator().manual_seed(420)
    data=torch.randn(256,2,generator=generator,dtype=torch.float64)
    eps=torch.randn(256,2,generator=generator,dtype=torch.float64)
    tau=torch.linspace(.05,.95,256,dtype=torch.float64)[:,None]
    phi=math.pi*tau/2;a,sigma=torch.cos(phi),torch.sin(phi)
    xt=a*data+sigma*eps
    vp_v=a*eps-sigma*data
    restored_data=a*xt-sigma*vp_v;restored_eps=sigma*xt+a*vp_v
    error=max(float((restored_data-data).abs().max()),float((restored_eps-eps).abs().max()))
    assert error<1e-12
    derivative=math.pi/2*(-torch.sin(phi)*data+torch.cos(phi)*eps)
    assert torch.allclose(derivative,math.pi/2*vp_v)
    rf_target=data-eps
    return {'vp_inverse_max_error':error,
            'vp_v_vs_rf_target_mse':float((vp_v-rf_target).square().mean()),
            'scope':'exact parameterization identities; VP v is not a generic RF velocity'}

def rectification():
    generator=torch.Generator().manual_seed(421)
    x0=torch.randn(512,2,generator=generator,dtype=torch.float64)*torch.tensor([1.5,.35])
    omega=math.pi/2;J=torch.tensor([[0.,-1.],[1.,0.]],dtype=torch.float64)
    identity=torch.eye(2,dtype=torch.float64);shift=torch.tensor([2.,1.],dtype=torch.float64)
    def rotation(t):
        return identity*math.cos(omega*t)+J*math.sin(omega*t)
    R=rotation(1.);target=x0@R.T+shift
    def teacher(x,t):return omega*((x-t*shift)@J.T)+shift
    def oracle(x,t):
        interpolation=(1-t)*identity+t*R
        source=(x-t*shift)@torch.linalg.inv(interpolation).T
        return source@(R-identity).T+shift
    errors=[]
    for steps in [1,4,16,64]:
        estimates=[]
        for field in [teacher,oracle]:
            x=x0.clone()
            for k in range(steps):x=x+field(x,k/steps)/steps
            estimates.append(float((x-target).square().mean().sqrt()))
        errors.append({'nfe':steps,'teacher_euler_rmse':estimates[0],'analytic_rectified_rmse':estimates[1]})
    assert errors[0]['analytic_rectified_rmse']<1e-12
    assert errors[-1]['teacher_euler_rmse']<errors[0]['teacher_euler_rmse']
    times=torch.linspace(0,1,101,dtype=torch.float64)
    curve=torch.stack([x0@rotation(float(t)).T+t*shift for t in times])
    straight=torch.stack([(1-t)*x0+t*target for t in times])
    velocities=torch.stack([teacher(x,float(t)) for x,t in zip(curve,times)])
    action=torch.trapezoid(velocities.square().sum(-1),times,dim=0).mean()
    displacement=(target-x0).square().sum(-1).mean()
    gap=float(action-displacement)
    assert gap>0
    fig,axes=plt.subplots(1,2,figsize=(8,3))
    for axis,path,title in zip(axes,[curve,straight],['Analytic curved teacher','Analytic rectified control']):
        for i in range(10):axis.plot(path[:,i,0],path[:,i,1],alpha=.7)
        axis.set_title(title);axis.set_xlabel('x1');axis.set_ylabel('x2')
    plt.tight_layout();plt.savefig(OUT/'rectification.png');plt.close(fig)
    return {'errors':errors,'teacher_action_minus_displacement_squared':gap,
            'scope':'known affine teacher and oracle rectification; no learned student, no one-step quality claim'}

def video_reasoning():
    rng=np.random.default_rng(422);T,H,W=12,48,48
    yy,xx=np.mgrid[:H,:W]
    base=np.exp(-((xx-14)**2+(yy-24)**2)/(2*3.5**2))
    truth=np.stack([np.roll(base,t,axis=1) for t in range(T)])
    variants={'coherent':truth,'frozen':np.repeat(truth[:1],T,axis=0),
              'independent_noise':np.clip(truth+rng.normal(0,.07,truth.shape),0,1)}
    rows=[]
    for name,clip in variants.items():
        previous_warped=np.roll(clip[:-1],1,axis=2)
        rows.append({'variant':name,'reference_mse':float(np.mean((clip-truth)**2)),
                     'raw_temporal_mse':float(np.mean(np.diff(clip,axis=0)**2)),
                     'known_motion_residual_mse':float(np.mean((clip[1:]-previous_warped)**2))})
    assert rows[1]['raw_temporal_mse']==0 and rows[1]['reference_mse']>0
    assert rows[0]['known_motion_residual_mse']==0
    # Perturb future frames and compare past outputs of a causal vs symmetric temporal convolution.
    generator=torch.Generator().manual_seed(423)
    x=torch.randn(1,1,12,8,8,generator=generator)
    changed=x.clone();changed[:,:,6:]+=7
    weight=torch.ones(1,1,3,1,1)/3
    def conv(v,causal):
        v=torch.nn.functional.pad(v,(0,0,0,0,2,0) if causal else (0,0,0,0,1,1))
        return torch.nn.functional.conv3d(v,weight)
    causal_delta=float((conv(x,True)[:,:,:6]-conv(changed,True)[:,:,:6]).abs().max())
    symmetric_delta=float((conv(x,False)[:,:,:6]-conv(changed,False)[:,:,:6]).abs().max())
    assert causal_delta==0 and symmetric_delta>0
    fig,axes=plt.subplots(3,6,figsize=(9,4.5))
    for row,(name,clip) in enumerate(variants.items()):
        for col,t in enumerate(range(0,T,2)):
            axes[row,col].imshow(clip[t],cmap='gray',vmin=0,vmax=1);axes[row,col].set_xticks([]);axes[row,col].set_yticks([])
            if row==0:axes[row,col].set_title(f't={t}')
            if col==0:axes[row,col].set_ylabel(name)
    plt.tight_layout();plt.savefig(OUT/'temporal-metrics.png');plt.close(fig)
    return {'metrics':rows,'causal_past_output_delta':causal_delta,'symmetric_past_output_delta':symmetric_delta,
            'scope':'synthetic translation, exact known motion and isolated Conv3d; not a production video benchmark'}

def main():
    torch.set_num_threads(4)
    report={'scope':'small deterministic CPU controls; no upstream code execution or model download',
            'environment':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__},
            'parameterizations':parameterizations(),'rectification':rectification(),'video':video_reasoning()}
    path=ROOT/'curriculum/reports/supplementary-experiments.json'
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
