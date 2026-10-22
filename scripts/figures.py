"""Aggregate measured logs, create paper figures/tables, and export to Overleaf."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from joint_control.io import save_json

LABELS={'fixed':'Fixed SGDM','cosine':'Cosine SGDM','pairwise':'Adaptive B + LR',
        'joint':'Joint (pilot 64)','fixed_batch':'Joint, fixed B','joint_p32':'Joint (pilot 32)'}
COLORS={'fixed':'#697386','cosine':'#172554','pairwise':'#059669','joint':'#dc5a29',
        'fixed_batch':'#9563b5','joint_p32':'#d5a21c'}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--results',default='results')
    parser.add_argument('--report-dir');args=parser.parse_args()
    root=Path(args.results);figs=root/'figures';tables=root/'tables'
    figs.mkdir(exist_ok=True);tables.mkdir(exist_ok=True)
    runs=json.loads((root/'aggregated/neural_runs.json').read_text())
    synthetic=json.loads((root/'aggregated/synthetic_runs.json').read_text())
    oracle=json.loads((root/'raw/oracle.json').read_text())
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.titlesize':8.5,
                         'axes.spines.top':False,'axes.spines.right':False,
                         'pdf.fonttype':42,'ps.fonttype':42})
    methods=list(LABELS);stats={}
    def mean_sd(values):
        return [float(np.mean(values)),float(np.std(values,ddof=1))]
    for method in methods:
        selected=[r for r in runs if r['method']==method]
        raw=[json.loads((root/f'raw/neural_{method}_seed{r["seed"]}.json').read_text()) for r in selected]
        stats[method]=dict(test_accuracy=mean_sd([100*r['test']['accuracy'] for r in selected]),
            test_loss=mean_sd([r['test']['loss'] for r in selected]),
            seconds=mean_sd([r['training_seconds'] for r in selected]),
            pilot_percent=mean_sd([100*r['pilot_gradient_examples']/r['gradient_examples'] for r in selected]),
            forward_examples=mean_sd([r['line_search_forward_examples'] for r in selected]),
            updates=mean_sd([len(r['steps']) for r in raw]),
            skip_percent=mean_sd([100*sum(s['line_search_status']=='non_descent' for s in r['steps'])/len(r['steps']) for r in raw]),
            capped=sum(s['line_search_status']=='capped' for r in raw for s in r['steps']),
            tail_steps=sum(s['line_search_status']=='tail_fixed_sgdm' for r in raw for s in r['steps']))
    save_json(root/'aggregated/neural_summary.json',stats)
    fig,axes=plt.subplots(1,2,figsize=(5.5,2.35),gridspec_kw={'width_ratios':[1.25,1]})
    grid=np.arange(0,38401,3200)
    for method in methods:
        curves=[]
        for r in [x for x in runs if x['method']==method]:
            data=json.loads((root/f'raw/neural_{method}_seed{r["seed"]}.json').read_text())['curve']
            curves.append(np.interp(grid,[x['gradient_examples'] for x in data],[x['validation_loss'] for x in data]))
        values=np.array(curves);avg=values.mean(axis=0);sd=values.std(axis=0,ddof=1)
        axes[0].plot(grid/1000,avg,label=LABELS[method],color=COLORS[method],lw=1.6)
        axes[0].fill_between(grid/1000,avg-sd,avg+sd,color=COLORS[method],alpha=.10)
    axes[0].set(xlabel='Gradient examples (thousands)',ylabel='Validation cross entropy',ylim=(.35,2.8))
    axes[0].legend(fontsize=7,ncol=1,loc='upper right',frameon=False,handlelength=1.5,labelspacing=.2)
    names=['Fixed','Cosine','B + LR','Joint','Fixed B','Pilot 32']
    accuracy=[stats[m]['test_accuracy'][0] for m in methods]
    errors=[stats[m]['test_accuracy'][1] for m in methods]
    axes[1].bar(np.arange(len(methods)),accuracy,yerr=errors,capsize=3,color=[COLORS[m] for m in methods])
    axes[1].set(xticks=np.arange(len(methods)),xticklabels=names,ylabel='Final test accuracy (%)',ylim=(40,90))
    axes[1].tick_params(axis='x',rotation=45,labelsize=7)
    fig.tight_layout(pad=.5)
    for ext in ['pdf','png']:fig.savefig(figs/f'neural_results.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(5.5,2.05))
    for axis,variance in zip(axes[:2],[1.,100.]):
        for method in methods[:5]:
            records=[json.loads((root/f'raw/synthetic_v{variance}_{method}_seed{s}.json').read_text())
                     for s in [0,1,2]]
            losses=np.array([[step['loss'] for step in r['steps']] for r in records])
            label='Joint reference' if method=='joint' else LABELS[method]
            axis.semilogy(np.arange(1,201),losses.mean(axis=0),color=COLORS[method],lw=1.4,label=label)
        axis.set(xlabel='Updates',title=f'Noise variance = {variance:g}')
    axes[0].set_ylabel('Objective gap')
    axes[0].legend(fontsize=7,frameon=False,handlelength=1,labelspacing=.1)
    candidates=oracle['candidates']
    for history,color in [('aligned','#059669'),('opposed','#697386'),('perturbed','#dc5a29')]:
        records=[r for r in candidates if r['history']==history and r['score']>0]
        axes[2].loglog([r['certificate'] for r in records],[r['mc_progress'] for r in records],'.',
                       ms=2.5,alpha=.45,label=history,color=color)
    bounds=axes[2].get_xlim();axes[2].plot(bounds,bounds,'k--',lw=.7)
    axes[2].set(xlabel='Certificate S / (4L)',ylabel='Mean one-step decrease',title='Exact Armijo')
    axes[2].legend(fontsize=7,frameon=False,handlelength=.7,labelspacing=.1)
    fig.tight_layout(pad=.5)
    for ext in ['pdf','png']:fig.savefig(figs/f'synthetic_results.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)
    # A compact cost diagnostic uses all actual per-step records.
    fig,axes=plt.subplots(1,3,figsize=(7.6,1.95))
    joint=[json.loads((root/f'raw/neural_joint_seed{s}.json').read_text()) for s in [0,1,2]]
    for axis,key,title in zip(axes,['batch','beta','eta'],['Chosen update batch','Chosen momentum','Accepted learning rate']):
        for seed,run in enumerate(joint):
            axis.plot([r['gradient_examples']/1000 for r in run['steps']],[r[key] for r in run['steps']],
                      lw=.5,alpha=.6,label=f'Seed {seed}')
        axis.set(xlabel='Gradient examples (thousands)',title=title)
    axes[2].set_yscale('symlog',linthresh=.01)
    axes[0].legend(fontsize=6,frameon=False)
    fig.tight_layout(pad=.5)
    for ext in ['pdf','png']:fig.savefig(figs/f'controller_diagnostics.{ext}',dpi=200,bbox_inches='tight')
    plt.close(fig)
    lines=['\\begin{tabular}{lrrrr}',r'\toprule',
           r'Method & Test acc. (\%) & Test CE & Time (s) & Pilot (\%) \\',r'\midrule']
    for method in methods:
        s=stats[method]
        a,ad=s['test_accuracy'];loss,ls=s['test_loss'];sec,_=s['seconds'];pilot,_=s['pilot_percent']
        lines.append(f"{LABELS[method]} & ${a:.2f}\\pm{ad:.2f}$ & ${loss:.3f}\\pm{ls:.3f}$ & {sec:.2f} & {pilot:.1f} \\")
        lines[-1]+='\\'
    lines += [r'\bottomrule',r'\end{tabular}']
    (tables/'neural.tex').write_text('\n'.join(lines)+'\n')
    synthstats={}
    for variance in [1.,100.]:
        synthstats[str(variance)]={}
        for method in methods[:5]:
            records=[r for r in synthetic if r['variance']==variance and r['method']==method]
            synthstats[str(variance)][method]=dict(final_loss=mean_sd([r['final_loss'] for r in records]),
                                                  examples=mean_sd([r['update_examples'] for r in records]))
    save_json(root/'aggregated/synthetic_summary.json',synthstats)
    if args.report_dir:
        report=Path(args.report_dir)
        for name,folder in [('figures',figs),('tables',tables)]:
            destination=report/name;destination.mkdir(parents=True,exist_ok=True)
            for path in folder.iterdir():shutil.copy2(path,destination/path.name)
        evidence=report/'evidence';evidence.mkdir(exist_ok=True)
        for name in ['neural_summary.json','synthetic_summary.json','tuning.json']:
            shutil.copy2(root/'aggregated'/name,evidence/name)
        save_json(evidence/'oracle_summary.json',oracle['summary'])
        hashes={str(p.relative_to(root)).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest()
                for folder in [figs,tables,root/'aggregated'] for p in folder.iterdir() if p.is_file()}
        save_json(evidence/'artifact_hashes.json',hashes)
    save_json(root/'plotting_manifest.json',{'script':'scripts/figures.py',
              'sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(json.dumps(stats,indent=2))
    print('SYNTHETIC',json.dumps(synthstats,indent=2))
    print('ORACLE',oracle['summary'])


if __name__=='__main__':main()
