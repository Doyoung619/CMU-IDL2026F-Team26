"""Fail on inconsistent saved evidence; does not rerun or alter experiments."""
import hashlib
import json
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
    print('PASS: source hashes, splits, 18 neural budgets, no test tuning, 30 synthetic runs, oracle, aggregation')


if __name__=='__main__':main()
