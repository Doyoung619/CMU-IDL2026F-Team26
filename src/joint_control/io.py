import json
from pathlib import Path
import platform
import subprocess
import sys
import numpy as np


def save_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    def scalar(item):
        if isinstance(item,np.generic):
            return item.item()
        raise TypeError(f'Unsupported JSON value {type(item)}')
    path.write_text(json.dumps(value,indent=2,allow_nan=False,default=scalar)+'\n',encoding='utf-8')


def environment():
    return dict(python=sys.version,platform=platform.platform(),processor=platform.processor(),
                numpy=np.__version__,hardware='Intel Core Ultra 5 226V; 8 cores; 16 GB RAM',
                compute='CPU float64; BLAS threads limited to 2; no CUDA',
                dependencies=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True).splitlines())
