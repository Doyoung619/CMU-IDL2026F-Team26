"""Matched total-gradient-budget Fashion-MNIST experiments."""
from pathlib import Path
import time
import hashlib
import numpy as np
from .controller import armijo, pilot_statistics, select, residual_learning_rate
from .diagnostics import audit_update
from .data import load
from .model import MLP
from .io import save_json


def train(config, data, method, seed, lr, beta, budget=None, test=True):
    x,y,vx,vy,tx,ty,_=data
    budget=budget or config['gradient_budget']
    model=MLP(hidden=config['hidden'])
    theta=model.initialize(seed)
    velocity=np.zeros_like(theta)
    # Same initialization and fresh-gradient random stream across methods;
    # varying batch sizes mean the trajectories are not identical sample paths.
    updates=np.random.default_rng(seed+10000)
    pilots=np.random.default_rng(seed+20000)
    q=config['initial_price']; eta_previous=1.; previous_applied_eta=0.; last_accepted_eta=0.
    tail_policy=config.get('tail_policy','cap_previous')
    consumed=0; next_log=config['log_every']; steps=[]; curve=[]
    pilot_total=0; update_total=0; forward_total=0
    start=time.perf_counter(); evaluation_seconds=0.; audit_seconds=0.
    audits=[];tail_events=[]
    monitor_indices=np.linspace(0,len(y)-1,min(1000,len(y)),dtype=int)
    def evaluate():
        nonlocal evaluation_seconds
        mark=time.perf_counter()
        metric=model.evaluate(theta,vx,vy)
        # Indices span every class of the stratified, class-ordered training set.
        training=model.evaluate(theta,x[monitor_indices],y[monitor_indices])
        curve.append(dict(gradient_examples=consumed,validation_loss=metric['loss'],
                          validation_accuracy=metric['accuracy'],train_monitor_loss=training['loss']))
        evaluation_seconds+=time.perf_counter()-mark
    evaluate()
    while consumed < budget:
        remaining=budget-consumed
        adaptive=method not in ('fixed','cosine')
        pilot_size=32 if method=='joint_p32' else config['pilot_size']
        tail=adaptive and remaining < pilot_size+min(config['batch_sizes'])
        pilot_grad=None
        selection_details={}
        if adaptive and not tail:
            indices=pilots.integers(0,len(y),size=pilot_size)
            parts=np.split(indices,config['pilot_groups'])
            gradients=[model.loss_gradient(theta,x[idx],y[idx])[1] for idx in parts]
            pilot_grad,variance=pilot_statistics(gradients,len(parts[0]))
            norm=float(pilot_grad @ pilot_grad)
            price=q*norm/min(config['batch_sizes'])
            batches=([64] if method=='fixed_batch' else config['batch_sizes'])
            batches=[b for b in batches if b+pilot_size <= remaining]
            if not batches:
                batches=[remaining-pilot_size]
            betas=[beta] if method=='pairwise' else config['momentums']
            batch,momentum,selected_score,selection_details=select(
                pilot_grad,velocity,variance,batches,betas,price,
                rtol=config.get('selection_rtol',1e-12),
                atol=config.get('selection_atol',1e-15),return_details=True)
            pilot_total+=pilot_size
        else:
            batch=min(128,remaining); momentum=beta; pilot_size=0
            selected_score=0.; variance=0.; price=0.
        indices=updates.integers(0,len(y),size=batch)
        loss,gradient=model.loss_gradient(theta,x[indices],y[indices])
        velocity=momentum*velocity+(1-momentum)*gradient
        update_total+=batch
        if pilot_grad is not None:
            objective=lambda candidate: model.loss_gradient(candidate,x[parts_flat],y[parts_flat],False)[0]
            parts_flat=np.concatenate(parts)
            result=armijo(theta,velocity,pilot_grad,objective,eta_previous,max_eta=16.)
            eta=result.eta; eta_previous=eta if eta>0 else eta_previous
            if eta>0:
                last_accepted_eta=eta
            forward_examples=result.evaluations*pilot_size
            forward_total+=forward_examples
            status=result.status
            q=max(0.,q+config['price_step']*(batch/config['target_update_batch']-1))
        else:
            eta=lr*(.5*(1+np.cos(np.pi*consumed/budget)) if method=='cosine' else 1.)
            if tail:
                eta=residual_learning_rate(lr,last_accepted_eta,tail_policy)
            status=('tail_capped_sgdm' if tail_policy=='cap_previous' else 'tail_fixed_sgdm') if tail else 'fixed_schedule'
            forward_examples=0
        audit_every=config.get('audit_every',0) if method=='joint' else 0
        if pilot_grad is not None and audit_every and (len(steps)%audit_every==0 or remaining==batch+pilot_size):
            mark=time.perf_counter()
            audit=audit_update(model,theta,velocity,pilot_grad,x,y,eta,status)
            audit.update(step=len(steps),optimization_gradient_examples=consumed+batch+pilot_size,batch=batch,beta=momentum)
            audits.append(audit);audit_seconds+=time.perf_counter()-mark
        tail_record=None
        if tail:
            mark=time.perf_counter()
            tail_record=dict(step=len(steps),policy=tail_policy,previous_eta=last_accepted_eta,
                             previous_applied_eta=previous_applied_eta,
                             eta=eta,batch=batch,validation_before=model.evaluate(theta,vx,vy),
                             pre_update_state_sha256=hashlib.sha256(theta.tobytes()+velocity.tobytes()).hexdigest())
            audit_seconds+=time.perf_counter()-mark
        theta=theta-eta*velocity
        previous_applied_eta=eta
        if tail_record is not None:
            mark=time.perf_counter()
            tail_record['validation_after']=model.evaluate(theta,vx,vy)
            tail_events.append(tail_record);audit_seconds+=time.perf_counter()-mark
        consumed+=batch+pilot_size
        if not np.all(np.isfinite(theta)):
            raise FloatingPointError(f'{method} seed {seed} diverged at {consumed}')
        steps.append(dict(step=len(steps),gradient_examples=consumed,batch=batch,pilot=pilot_size,
                          beta=momentum,eta=eta,score=selected_score,variance=variance,price=price,
                          update_batch_loss=loss,line_search_status=status,
                          line_search_forward_examples=forward_examples,
                          selection=selection_details))
        if consumed>=next_log or consumed==budget:
            evaluate(); next_log+=config['log_every']
    training_seconds=time.perf_counter()-start-evaluation_seconds-audit_seconds
    validation=model.evaluate(theta,vx,vy)
    test_result=model.evaluate(theta,tx,ty) if test else None
    result=dict(status='completed',method=method,seed=seed,config=config,learning_rate=lr,
                fixed_momentum=beta,gradient_examples=consumed,pilot_gradient_examples=pilot_total,
                update_gradient_examples=update_total,line_search_forward_examples=forward_total,
                training_seconds=training_seconds,monitor_evaluation_seconds=evaluation_seconds,
                diagnostic_seconds=audit_seconds,audits=audits,tail_events=tail_events,
                diagnostic_gradient_examples=sum(a['gradient_examples'] for a in audits),
                diagnostic_forward_examples=sum(a['forward_examples'] for a in audits),
                parameter_sha256=hashlib.sha256(theta.tobytes()).hexdigest(),
                validation=validation,test=test_result,curve=curve,steps=steps)
    assert consumed==budget and pilot_total+update_total==budget
    return result


def run(config, output):
    output=Path(output)
    data=load(train_per_class=config['train_per_class'],val_per_class=config['validation_per_class'],
              split_seed=config['split_seed'])
    save_json(output/'raw/data_manifest.json',data[-1])
    chosen={}; trials=[]
    for method in ['fixed','cosine']:
        for lr in config['tuning_learning_rates']:
            for beta in config['tuning_momentums']:
                result=train(config,data,method,config['tuning_seed'],lr,beta,
                             config['tuning_budget'],test=False)
                save_json(output/f'raw/tuning_{method}_lr{lr}_b{beta}.json',result)
                trial=dict(method=method,lr=lr,beta=beta,validation=result['validation'])
                trials.append(trial)
                print('TUNE',trial,flush=True)
        candidates=[r for r in trials if r['method']==method]
        chosen[method]=min(candidates,key=lambda r:r['validation']['loss'])
    save_json(output/'aggregated/tuning.json',dict(selection='minimum validation cross entropy; no test evaluation',
                                                  trials=trials,chosen=chosen))
    summaries=[]
    for seed in config['seeds']:
        for method in config['methods']:
            settings=chosen['cosine' if method=='cosine' else 'fixed']
            result=train(config,data,method,seed,settings['lr'],settings['beta'])
            save_json(output/f'raw/neural_{method}_seed{seed}.json',result)
            summary={k:v for k,v in result.items() if k not in ('steps','curve','config','audits','tail_events')}
            summaries.append(summary)
            print('NEURAL',method,seed,'test',result['test'],'seconds',round(result['training_seconds'],2),flush=True)
    save_json(output/'aggregated/neural_runs.json',summaries)
