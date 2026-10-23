"""Fail on inconsistent saved evidence; does not rerun or alter experiments."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def main():
    root=Path('results');config=read(root/'executed_config.json')
    hashes=read(root/'source_manifest.json')
    for path,expected in hashes.items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
    data=read(root/'raw/data_manifest.json')
    assert len(data['train_indices'])==12000 and len(data['validation_indices'])==2000
    assert not set(data['train_indices']) & set(data['validation_indices'])
    summaries=read(root/'aggregated/neural_runs.json')
    assert len(summaries)==len(config['methods'])*len(config['seeds'])
    stats=read(root/'aggregated/neural_summary.json')
    for method in config['methods']:
        accuracies=[]
        for seed in config['seeds']:
            run=read(root/f'raw/neural_{method}_seed{seed}.json')
            assert run['status']=='completed'
            assert sum(step['pilot']+step['batch'] for step in run['steps'])==38400
            assert run['pilot_gradient_examples']+run['update_gradient_examples']==38400
            assert sum(step['line_search_forward_examples'] for step in run['steps'])==run['line_search_forward_examples']
            assert run['curve'][-1]['gradient_examples']==38400
            for tail in run['tail_events']:
                assert tail['policy']=='cap_previous' and tail['eta']<=tail['previous_eta']
            for s in run['steps']:
                detail=s['selection']
                if detail:
                    assert detail['priced_gap']<=detail['tolerance']+1e-20
            assert run['diagnostic_gradient_examples']==sum(a['gradient_examples'] for a in run['audits'])
            assert all(a['gradient_examples']==12000 for a in run['audits'])
            if method=='joint':assert run['steps'][0]['beta']==0.
            assert 0<=run['test']['accuracy']<=1 and run['test']['loss']>=0
            accuracies.append(100*run['test']['accuracy'])
        np.testing.assert_allclose(stats[method]['test_accuracy'],
                                   [np.mean(accuracies),np.std(accuracies,ddof=1)],atol=1e-12)
    tuning=read(root/'aggregated/tuning.json')
    assert len(tuning['trials'])==12
    for path in (root/'raw').glob('tuning_*.json'):
        assert read(path)['test'] is None,'test leakage in tuning'
    synthetic=read(root/'aggregated/synthetic_runs.json')
    assert len(synthetic)==30
    for run in synthetic:
        full=read(root/f"raw/synthetic_v{run['variance']}_{run['method']}_seed{run['seed']}.json")
        assert len(full['steps'])==200 and full['update_examples']<=12800
        assert np.isclose(sum(s['batch'] for s in full['steps']),full['update_examples'])
        if run['method'] not in ('fixed','cosine'):
            assert all(s['actual_decrease']>=-1e-10 for s in full['steps'])
    oracle=read(root/'raw/oracle.json')
    assert len(oracle['states'])==60 and len(oracle['candidates'])==720
    assert sum(s['agrees'] for s in oracle['states'])==57
    assert all(c['mc_progress']+3*c['mc_standard_error']>=c['certificate']-1e-10 for c in oracle['candidates'])
    feedback=read(root/'feedback/summary.json')
    assert len(feedback['residual_variants'])==12 and all(c['identical_prefix'] for c in feedback['residual_comparisons'])
    audit=read(root/'feedback/pilot_audit.json')
    assert audit['summary']['sampled_steps']==len(audit['steps'])
    assert audit['summary']['diagnostic_gradient_examples']==12000*len(audit['steps'])
    assert audit['summary']['accepted_loss_increases']==sum(a['accepted_loss_increase'] for a in audit['steps'])
    assert audit['summary']['skipped_full_descent']==sum(a['skipped_full_descent'] for a in audit['steps'])
    decisions=[s for seed in config['seeds'] for s in read(root/f'raw/neural_joint_seed{seed}.json')['steps'] if s['pilot']>0]
    frequencies=feedback['revised_frequencies']
    assert frequencies['controller_decisions']==len(decisions)
    for key in ['batch','beta']:
        actual=Counter(str(s[key]) for s in decisions)
        assert actual==Counter({k:v['count'] for k,v in frequencies[key].items()})
        for value in frequencies[key].values():
            assert np.isclose(value['percent'],100*value['count']/len(decisions))
    for path,expected in read(root/'feedback/source_manifest.json').items():
        assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==expected,path
    print('PASS: source hashes, budgets, split/test separation, residual caps, tolerant ties, paired prefixes, pilot audit, aggregation, synthetic oracle')


if __name__=='__main__':main()
