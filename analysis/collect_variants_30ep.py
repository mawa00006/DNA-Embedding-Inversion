import torch, glob, re, os
import numpy as np
from collections import defaultdict

rows = defaultdict(lambda: {'var': [], 'ref': [], 'seen': [], 'unseen': [], 'nvar': 0, 'n': 0})

for d in sorted(glob.glob('outputs_1000g/eval_105k_30ep_*')):
    model = os.path.basename(d).replace('eval_105k_', '')
    for p in glob.glob(os.path.join(d, '*_L*/plot_data.pt')):
        z = torch.load(p, map_location='cpu', weights_only=False)
        lev = np.asarray(z['levenshtein_similarities'], dtype=float)
        vm = z.get('variant_bearing_mask')
        sm = z.get('train_seen_mask')
        r = rows[model]
        r['n'] += len(lev)
        if vm is not None:
            vm = np.asarray(vm, dtype=bool)
            r['nvar'] += int(vm.sum())
            if vm.any():
                r['var'].append(lev[vm].mean())
            if (~vm).any():
                r['ref'].append(lev[~vm].mean())
        if sm is not None:
            sm = np.asarray(sm, dtype=bool)
            if sm.any():
                r['seen'].append(lev[sm].mean())
            if (~sm).any():
                r['unseen'].append(lev[~sm].mean())

def m(v):
    return sum(v) / len(v) if v else float('nan')

print('')
print('%-16s %9s %8s %8s %9s %8s %8s %9s' % (
    'model', 'n_variant', 'var', 'ref', 'delta', 'seen', 'unseen', 'delta'))
for k in sorted(rows, key=lambda k: -m(rows[k]['var'])):
    r = rows[k]
    print('%-16s %9d %8.4f %8.4f %+9.4f %8.4f %8.4f %+9.4f' % (
        k, r['nvar'], m(r['var']), m(r['ref']), m(r['var']) - m(r['ref']),
        m(r['seen']), m(r['unseen']), m(r['seen']) - m(r['unseen'])))
