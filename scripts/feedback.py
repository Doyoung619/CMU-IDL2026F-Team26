"""Controlled residual/tie-rule comparisons and training-only pilot diagnostics."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from threadpoolctl import threadpool_limits
from joint_control.controller import score,select,pilot_statistics
from joint_control.data import load
from joint_control.io import save_json
from joint_control.model import MLP
from joint_control.neural import train

PUBLISHED='6836c47f346c182fa236f7069e608ffec79a2dde'


def frequencies(runs,budget):
    all_steps=[s for r in runs for s in r['steps']]
    decisions=[s for s in all_steps if s['pilot']>0]
    def proportions(records,key):
        counts=Counter(str(s[key]) for s in records)
        return {k:dict(count=v,percent=100*v/len(records)) for k,v in sorted(counts.items(),key=lambda x:float(x[0]))}
    phases=[]
    for i in range(4):
        records=[s for s in decisions if min(3,int((s['gradient_examples']-1)*4/budget))==i]
        phases.append(dict(budget_fraction=[i/4,(i+1)/4],steps=len(records),
                           batch_64_percent=100*sum(s['batch']==64 for s in records)/len(records),
                           beta_zero_percent=100*sum(s['beta']==0 for s in records)/len(records),
                           mean_eta=float(np.mean([s['eta'] for s in records]))))
    return dict(total_steps=len(all_steps),controller_decisions=len(decisions),
                tail_steps=len(all_steps)-len(decisions),batch=proportions(decisions,'batch'),
                beta=proportions(decisions,'beta'),phases=phases)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results',default='results');args=parser.parse_args()
    root=Path(args.results);config=json.loads((root/'executed_config.json').read_text())
    folder=root/'feedback';folder.mkdir(exist_ok=True)
    source_files=sorted(Path('src').rglob('*.py'))+[Path('scripts/feedback.py'),Path('configs/midterm.json')]
    save_json(folder/'source_manifest.json',{str(p).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})
    data=load(train_per_class=config['train_per_class'],val_per_class=config['validation_per_class'],split_seed=config['split_seed'])
    fixed=json.loads((root/'aggregated/tuning.json').read_text())['chosen']['fixed']
    records=[];comparisons=[];first_scores=[]
    with threadpool_limits(limits=2):
        for seed in config['seeds']:
            variants={}
            for tolerant in [False,True]:
                for capped in [False,True]:
                    name=('tolerant' if tolerant else 'strict')+'_'+('capped' if capped else 'legacy')
                    if tolerant and capped:
                        result=json.loads((root/f'raw/neural_joint_p32_seed{seed}.json').read_text())
                    else:
                        settings=dict(config,selection_rtol=config['selection_rtol'] if tolerant else 0.,
                                      selection_atol=config['selection_atol'] if tolerant else 0.,
                                      tail_policy='cap_previous' if capped else 'legacy_fixed',audit_every=0)
                        result=train(settings,data,'joint_p32',seed,fixed['lr'],fixed['beta'],test=False)
                    variants[name]=result
                    save_json(folder/f'raw/p32_{name}_seed{seed}.json',result)
                    tail=result['tail_events'][-1] if result['tail_events'] else None
                    record=dict(seed=seed,variant=name,has_tail=tail is not None,
                                validation_accuracy=result['validation']['accuracy'],
                                previous_eta=tail['previous_eta'] if tail else None,
                                residual_eta=tail['eta'] if tail else None,
                                validation_before=tail['validation_before']['accuracy'] if tail else None,
                                validation_after=tail['validation_after']['accuracy'] if tail else None)
                    records.append(record)
                    print('RESIDUAL',record,flush=True)
            for rule in ['strict','tolerant']:
                a,b=variants[rule+'_legacy'],variants[rule+'_capped']
                ta,tb=a['tail_events'],b['tail_events']
                assert bool(ta)==bool(tb),'A tail policy must not alter the prefix'
                if ta:
                    assert ta[-1]['pre_update_state_sha256']==tb[-1]['pre_update_state_sha256']
                    assert ta[-1]['validation_before']==tb[-1]['validation_before']
                else:
                    assert a['parameter_sha256']==b['parameter_sha256']
                comparisons.append(dict(seed=seed,tie_rule=rule,identical_prefix=True,
                                        capped_minus_legacy_validation_pp=100*(b['validation']['accuracy']-a['validation']['accuracy'])))
            # Show cancellation of beta when v=0 using the actual first pilot.
            x,y,*_=data;model=MLP(hidden=config['hidden']);theta=model.initialize(seed)
            indices=np.random.default_rng(seed+20000).integers(0,len(y),size=config['pilot_size'])
            parts=np.split(indices,config['pilot_groups'])
            g,var=pilot_statistics([model.loss_gradient(theta,x[p],y[p])[1] for p in parts],len(parts[0]))
            price=config['initial_price']*float(g@g)/32
            values={str(b):[dict(beta=beta,score=score(g,np.zeros_like(g),var,b,beta)) for beta in config['momentums']] for b in config['batch_sizes']}
            strict=select(g,np.zeros_like(g),var,config['batch_sizes'],config['momentums'],price,rtol=0.,atol=0.)
            tolerant=select(g,np.zeros_like(g),var,config['batch_sizes'],config['momentums'],price)
            first_scores.append(dict(seed=seed,scores_by_batch=values,strict=list(strict),tolerant=list(tolerant)))
    joint=[json.loads((root/f'raw/neural_joint_seed{s}.json').read_text()) for s in config['seeds']]
    audits=[dict(seed=r['seed'],**a) for r in joint for a in r['audits']]
    audit_summary=dict(protocol='Every 20th controller step from step 0, plus a final controller step that exhausts the budget; selected before observing outcomes.',
                       objective='Entire retained 12000-example training split; no validation/test gradient.',
                       sampled_steps=len(audits),accepted=sum(a['eta']>0 for a in audits),
                       accepted_loss_increases=sum(a['accepted_loss_increase'] for a in audits),
                       skipped=sum(a['status']=='non_descent' for a in audits),
                       skipped_full_descent=sum(a['skipped_full_descent'] for a in audits),
                       diagnostic_gradient_examples=sum(a['gradient_examples'] for a in audits),
                       diagnostic_forward_examples=sum(a['forward_examples'] for a in audits),
                       decision_independent=True,excluded_from_optimization_budget_and_training_timing=True)
    published=[]
    for seed in config['seeds']:
        raw=subprocess.check_output(['git','show',f'{PUBLISHED}:results/raw/neural_joint_seed{seed}.json'])
        published.append(json.loads(raw))
    published_summary=dict(commit=PUBLISHED,frequencies=frequencies(published,config['gradient_budget']),
                           first_beta={str(r['seed']):r['steps'][0]['beta'] for r in published})
    save_json(folder/'published_reference.json',published_summary)
    summary=dict(residual_variants=records,residual_comparisons=comparisons,
                 revised_frequencies=frequencies(joint,config['gradient_budget']),
                 pilot_audit=audit_summary,first_step_scores=first_scores,
                 interpretation='Residual cap and tie rule are separate interventions. Full-training audit shows sampled phenomena, not the cause of the entire accuracy gap.')
    save_json(folder/'pilot_audit.json',dict(summary=audit_summary,steps=audits))
    save_json(folder/'summary.json',summary)
    print('PILOT AUDIT',audit_summary,flush=True)
    print('FREQUENCIES',summary['revised_frequencies'],flush=True)


if __name__=='__main__':main()
