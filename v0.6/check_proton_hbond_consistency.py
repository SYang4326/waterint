#!/usr/bin/env python3
"""Compare every binned surface from WaterInt Python and C++ backends exactly."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import yaml
from waterint._04_workflows.workflows.proton_sharing_hbond import run_proton_sharing_hbond

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',required=True,type=Path); ap.add_argument('--output',required=True,type=Path); args=ap.parse_args()
    base=yaml.safe_load(args.config.read_text()); args.output.mkdir(parents=True,exist_ok=True); results={}
    for backend in ('python','cpp'):
        cfg={**base,'proton_sharing_hbond':dict(base['proton_sharing_hbond']),'output':dict(base['output'])}
        cfg['proton_sharing_hbond']['backend']=backend; cfg['output']['directory']=str(args.output/backend)
        results[backend]=run_proton_sharing_hbond(cfg)
    failures=[]; comparisons=[]
    for py in sorted((args.output/'python').glob('*.npz')):
        cp=args.output/'cpp'/py.name
        a=np.load(py)['counts']; b=np.load(cp)['counts']; diff=np.abs(a-b); comparisons.append({'surface':py.name,'python_pairs':float(a.sum()),'cpp_pairs':float(b.sum()),'max_abs_histogram_difference':float(diff.max()),'different_bins':int((diff>0).sum())})
        if diff.max()!=0: failures.append(py.name)
    report={'config':str(args.config),'frames':results['python'].frames,'comparisons':comparisons,'status':'PASS' if not failures else 'FAIL','failed_surfaces':failures}
    (args.output/'CONSISTENCY_CHECK.json').write_text(json.dumps(report,indent=2)+'\n')
    if failures: raise SystemExit('Backend mismatch: '+', '.join(failures))
if __name__=='__main__': main()
