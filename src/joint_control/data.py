"""Download official Fashion-MNIST IDX files and retain verified data provenance."""
import gzip
import hashlib
import json
from pathlib import Path
import struct
import requests
import numpy as np

FILES = {
 'train-images-idx3-ubyte.gz':'8d4fb7e6c68d591d4c3dfef9ec88bf0d',
 'train-labels-idx1-ubyte.gz':'25c81989df183df01b3e8a0aad5dffbe',
 't10k-images-idx3-ubyte.gz':'bef4ecab320f06d8554ea6380940ec79',
 't10k-labels-idx1-ubyte.gz':'bb300cfdad3c16e7a12a480ee83cd310'}
BASE = 'https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/data/fashion/'


def load(root='data', train_per_class=1200, val_per_class=200, split_seed=2026):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    manifest={}
    arrays={}
    for name, expected in FILES.items():
        path=root/name
        if not path.exists():
            response=requests.get(BASE+name,timeout=120)
            response.raise_for_status()
            path.write_bytes(response.content)
        raw=path.read_bytes()
        if hashlib.md5(raw).hexdigest()!=expected:
            raise ValueError(f'Dataset checksum mismatch: {name}')
        manifest[name]={'url':BASE+name,'md5':expected,'sha256':hashlib.sha256(raw).hexdigest()}
        unpacked=gzip.decompress(raw)
        magic,count=struct.unpack('>II',unpacked[:8])
        if magic==2051:
            rows,cols=struct.unpack('>II',unpacked[8:16])
            array=np.frombuffer(unpacked,dtype=np.uint8,offset=16).reshape(count,rows*cols)
        elif magic==2049:
            array=np.frombuffer(unpacked,dtype=np.uint8,offset=8)
        else:
            raise ValueError(f'Invalid IDX magic: {magic}')
        arrays[name]=array
    labels=arrays['train-labels-idx1-ubyte.gz']
    rng=np.random.default_rng(split_seed)
    train=[]; valid=[]
    for label in range(10):
        indices=rng.permutation(np.flatnonzero(labels==label))
        train.extend(indices[:train_per_class].tolist())
        valid.extend(indices[train_per_class:train_per_class+val_per_class].tolist())
    train=np.asarray(train); valid=np.asarray(valid)
    assert not np.intersect1d(train,valid).size
    # Pixel scaling uses no fitted statistics, and never inspects the test labels.
    images=arrays['train-images-idx3-ubyte.gz']
    train_x=images[train].astype(np.float64)/255.
    valid_x=images[valid].astype(np.float64)/255.
    test_x=arrays['t10k-images-idx3-ubyte.gz'].astype(np.float64)/255.
    manifest.update(split_seed=split_seed,train_indices=train.tolist(),validation_indices=valid.tolist(),
                    normalization='pixel / 255; no augmentation',test_split='official 10000')
    return (train_x,labels[train],valid_x,labels[valid],test_x,
            arrays['t10k-labels-idx1-ubyte.gz'],manifest)
