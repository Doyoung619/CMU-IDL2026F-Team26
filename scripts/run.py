"""Run from repository root: python scripts/run.py --suite all."""
import argparse
import json
from pathlib import Path
import sys
import hashlib
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from threadpoolctl import threadpool_limits
from joint_control.io import save_json,environment


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--suite',choices=['all','synthetic','neural'],default='all')
    parser.add_argument('--config',default='configs/midterm.json')
    parser.add_argument('--output',default='results')
    args=parser.parse_args()
    config=json.loads(Path(args.config).read_text())
    save_json(Path(args.output)/'environment.json',environment())
    save_json(Path(args.output)/'executed_config.json',config)
    source_files=sorted(Path('src').rglob('*.py'))+[Path('scripts/run.py'),Path(args.config)]
    save_json(Path(args.output)/'source_manifest.json',
              {str(p).replace('\\','/'):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files})
    with threadpool_limits(limits=2):
        if args.suite in ('all','synthetic'):
            from joint_control.synthetic import run
            run(config,args.output)
        if args.suite in ('all','neural'):
            from joint_control.neural import run
            run(config,args.output)
