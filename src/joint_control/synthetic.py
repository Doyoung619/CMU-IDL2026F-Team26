"""Exact quadratic oracle, empirical certificate checks, and real noisy runs."""
from pathlib import Path
import time
import numpy as np
from scipy.stats import spearmanr
from .controller import score,select,armijo
from .io import save_json


def oracle(config,output):
    rng=np.random.default_rng(771)
    d=config['synthetic_dimension']; h=np.geomspace(1.,config['synthetic_condition'],d)
    batches=config['batch_sizes']; betas=config['momentums']; rows=[]; states=[]
    for state in range(config['oracle_states']):
        theta=rng.normal(size=d)*10**rng.uniform(-1.,0.)
        g=h*theta
        history=state%3
        perturbation=rng.normal(size=d)*np.linalg.norm(g)/np.sqrt(d)
        v=g.copy() if history==0 else (-g if history==1 else .7*g+.7*perturbation)
        # Relative noise keeps difficult states present across objective scales.
        variance=config['synthetic_noise_variances'][state%2]*float(g@g)
        local=[]
        # Common standard-normal draws improve precision in pair comparisons.
        normal=rng.normal(size=(config['oracle_draws'],d))
        for b in batches:
            for beta in betas:
                m=beta*v+(1-beta)*g
                a=float(g@m)
                noise=(1-beta)**2*variance/b
                evhv=float(m@(h*m))+noise*float(h.mean())
                best_expected=max(a,0.)**2/(2*evhv) if evhv else 0.
                s=score(g,v,variance,b,beta)
                directions=m+normal*np.sqrt(noise/d)
                aligns=directions@g
                etas=np.ones(len(normal)); active=aligns>0
                # Evaluate the exact objective; no Hessian enters this search.
                old=.5*float(theta@(h*theta))
                def passes(steps):
                    trial=theta-steps[:,None]*directions
                    f=.5*np.sum(trial*trial*h,axis=1)
                    return f <= old-.5*steps*aligns
                for _ in range(50):
                    ok=passes(etas)
                    bad=active & ~ok
                    if not bad.any(): break
                    etas[bad]*=.5
                else: raise RuntimeError('MC shrink failed')
                for _ in range(50):
                    expand=active & passes(2*etas)
                    if not expand.any(): break
                    etas[expand]*=2
                else: raise RuntimeError('MC expand failed')
                etas[~active]=0
                trial=theta-etas[:,None]*directions
                progress=old-.5*np.sum(trial*trial*h,axis=1)
                record=dict(state=state,history=['aligned','opposed','perturbed'][history],variance=variance,
                            batch=b,beta=beta,score=s,certificate=s/(4*h.max()),
                            expected_quadratic_oracle=best_expected,mc_progress=float(progress.mean()),
                            mc_standard_error=float(progress.std(ddof=1)/np.sqrt(len(progress))))
                rows.append(record);local.append(record)
        selected=max(local,key=lambda r:r['score'])
        optimal=max(local,key=lambda r:r['expected_quadratic_oracle'])
        regret=optimal['expected_quadratic_oracle']-selected['expected_quadratic_oracle']
        states.append(dict(state=state,history=selected['history'],variance=variance,
                           agrees=regret<1e-9,relative_regret=regret/max(optimal['expected_quadratic_oracle'],1e-30),
                           rank_correlation=float(spearmanr([r['score'] for r in local],
                                                   [r['expected_quadratic_oracle'] for r in local]).statistic)))
    summary=dict(states=len(states),candidate_evaluations=len(rows),draws_per_candidate=config['oracle_draws'],
                 noise_protocol='variance = relative noise factor (1 or 100) * squared gradient norm',
                 oracle_agreement=float(np.mean([r['agrees'] for r in states])),
                 mean_relative_regret=float(np.mean([r['relative_regret'] for r in states])),
                 mean_rank_correlation=float(np.mean([r['rank_correlation'] for r in states])),
                 mc_certificate_violations_3se=sum(r['mc_progress']+3*r['mc_standard_error']<r['certificate']-1e-10 for r in rows))
    save_json(output/'raw/oracle.json',dict(config=config,states=states,candidates=rows,summary=summary))
    print('ORACLE',summary,flush=True)


def trajectory(config,seed,variance,method,lr=.1,beta=.9):
    rng=np.random.default_rng(seed+1000)
    d=config['synthetic_dimension'];h=np.geomspace(1.,config['synthetic_condition'],d)
    theta=np.random.default_rng(seed).normal(size=d);v=np.zeros(d)
    horizon=config['synthetic_horizon'];remaining=config['synthetic_budget'];consumed=0
    q=config['initial_price'];rows=[];objective=lambda z:.5*float(z@(h*z))
    start=time.perf_counter()
    for t in range(horizon):
        old=objective(theta);g=h*theta
        if method in ('fixed','cosine'):
            b=64;momentum=beta;s=score(g,v,variance,b,momentum)
        else:
            candidates=[64] if method=='fixed_batch' else config['batch_sizes']
            candidates=[b for b in candidates if b<=remaining-(horizon-t-1)*min(candidates)]
            momentum_grid=[beta] if method=='pairwise' else config['momentums']
            price=q*float(g@g)/min(config['batch_sizes'])
            b,momentum,s=select(g,v,variance,candidates,momentum_grid,price)
            if method=='joint':
                assert s+1e-9>=score(g,v,variance,32,0.)
        noise=rng.normal(size=d)*np.sqrt(variance/(d*b))
        v=momentum*v+(1-momentum)*(g+noise)
        if method in ('fixed','cosine'):
            eta=lr*(.5*(1+np.cos(np.pi*t/horizon)) if method=='cosine' else 1.)
            status='fixed_schedule';calls=0
        else:
            search=armijo(theta,v,g,objective,1.,max_eta=1e6)
            eta,status,calls=search.eta,search.status,search.evaluations
            q=max(0.,q+config['price_step']*(b/64.-1))
        theta=theta-eta*v;remaining-=b;consumed+=b
        loss=objective(theta)
        if not np.isfinite(loss): raise FloatingPointError('Synthetic divergence')
        rows.append(dict(step=t,examples=consumed,loss=loss,batch=b,beta=momentum,eta=eta,score=s,
                         actual_decrease=old-loss,status=status,objective_evaluations=calls))
    return dict(status='completed',seed=seed,variance=variance,method=method,lr=lr,beta=beta,
                initial_loss=rows[0]['loss']+rows[0]['actual_decrease'],final_loss=rows[-1]['loss'],
                maximum_loss=max(row['loss'] for row in rows),
                update_examples=consumed,seconds=time.perf_counter()-start,steps=rows)


def run(config,output):
    output=Path(output);oracle(config,output);summaries=[];trials=[]
    for variance in config['synthetic_noise_variances']:
        choices={}
        for method in ['fixed','cosine']:
            for lr in [.01,.03,.1]:
                for beta in [0.,.9]:
                    result=trajectory(config,991,variance,method,lr,beta)
                    trial={k:v for k,v in result.items() if k!='steps'}
                    trial['development_stable']=trial['maximum_loss']<=10*trial['initial_loss']
                    trials.append(trial)
            choices[method]=min([r for r in trials if r['variance']==variance and r['method']==method],
                                 key=lambda r:(not r['development_stable'],r['final_loss']))
        for seed in config['seeds']:
            for method in ['fixed','cosine','pairwise','joint','fixed_batch']:
                setting=choices['cosine' if method=='cosine' else 'fixed']
                result=trajectory(config,seed,variance,method,setting['lr'],setting['beta'])
                save_json(output/f'raw/synthetic_v{variance}_{method}_seed{seed}.json',result)
                summaries.append({k:v for k,v in result.items() if k!='steps'})
    save_json(output/'aggregated/synthetic_tuning.json',trials)
    save_json(output/'aggregated/synthetic_runs.json',summaries)
    print('SYNTHETIC completed',len(summaries),'runs',flush=True)
