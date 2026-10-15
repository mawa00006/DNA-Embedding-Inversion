import torch, glob, re, os
from collections import defaultdict

LENGTHS = [10,15,20,25,30,35,40,45,50,60,70,80,90,100]
acc, exm, base = defaultdict(dict), defaultdict(dict), defaultdict(dict)

for d in sorted(glob.glob('outputs/eval_105k_*')):
    model = os.path.basename(d).replace('eval_105k_', '')
    for p in glob.glob(os.path.join(d, '*_L*/plot_data.pt')):
        m = re.search(r'_L(\d+)/plot_data\.pt$', p)
        if not m:
            continue
        L = int(m.group(1))
        try:
            z = torch.load(p, map_location='cpu', weights_only=False)
        except Exception as e:
            print('  ! %s L=%d: %s' % (model, L, e))
            continue
        a = z['accuracy_metrics']
        acc[model][L] = a['nucleotide_accuracy']
        exm[model][L] = a['exact_match']
        bp = z.get('baseline_position_accuracy')
        if bp is not None and len(bp):
            base[model][L] = float(sum(bp) / len(bp))

models = sorted(acc, key=lambda m: -(sum(acc[m].values()) / max(len(acc[m]), 1)))

print('')
print('=== mean over available lengths ===')
print('%-18s%3s  %8s %8s %9s %7s' % ('model', 'n', 'nuc_acc', 'exact', 'baseline', 'ratio'))
for m in models:
    v, e, b = list(acc[m].values()), list(exm[m].values()), list(base[m].values())
    ma, me = sum(v) / len(v), sum(e) / len(e)
    if b:
        mb = sum(b) / len(b)
        print('%-18s%3d  %8.4f %8.4f %9.4f %6.2fx' % (m, len(v), ma, me, mb, ma / mb))
    else:
        print('%-18s%3d  %8.4f %8.4f %9s %7s' % (m, len(v), ma, me, 'n/a', 'n/a'))

print('')
print('=== nucleotide_accuracy per length ===')
print('%-18s' % 'model' + ''.join('%7d' % L for L in LENGTHS))
for m in models:
    row = ''.join(('%7.3f' % acc[m][L]) if L in acc[m] else '%7s' % '-' for L in LENGTHS)
    print('%-18s' % m + row)
