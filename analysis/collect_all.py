import torch, glob, re, os
from collections import defaultdict

LENGTHS = [10,15,20,25,30,35,40,45,50,60,70,80,90,100]
TREES = {'hg38': 'outputs/eval_105k_*', 'ood': 'outputs_1000g/eval_105k_*'}

acc = {k: defaultdict(dict) for k in TREES}
base = {k: defaultdict(dict) for k in TREES}

for tag, pattern in TREES.items():
    for d in sorted(glob.glob(pattern)):
        model = os.path.basename(d).replace('eval_105k_', '')
        for p in glob.glob(os.path.join(d, '*_L*/plot_data.pt')):
            m = re.search(r'_L(\d+)/plot_data\.pt$', p)
            if not m:
                continue
            L = int(m.group(1))
            try:
                z = torch.load(p, map_location='cpu', weights_only=False)
            except Exception as e:
                print('  ! %s %s L=%d: %s' % (tag, model, L, e))
                continue
            acc[tag][model][L] = z['accuracy_metrics']['nucleotide_accuracy']
            bp = z.get('baseline_position_accuracy')
            if bp is not None and len(bp):
                base[tag][model][L] = float(sum(bp) / len(bp))

def mean(d):
    v = list(d.values())
    return sum(v) / len(v) if v else float('nan')

models = sorted(acc['hg38'], key=lambda m: -mean(acc['hg38'][m]))

print('')
print('=== hg38 vs 1000G OOD (mean over lengths) ===')
print('%-16s %5s %8s %5s %8s %9s %8s' % ('model', 'n', 'hg38', 'n', 'ood', 'gap', 'ood/base'))
for m in models:
    h, o = mean(acc['hg38'][m]), mean(acc['ood'].get(m, {}))
    ob = mean(base['ood'].get(m, {}))
    gap = h - o
    ratio = o / ob if ob == ob and ob else float('nan')
    print('%-16s %5d %8.4f %5d %8.4f %+9.4f %7.2fx' % (
        m, len(acc['hg38'][m]), h, len(acc['ood'].get(m, {})), o, gap, ratio))

print('')
print('=== 1000G OOD nucleotide_accuracy per length ===')
print('%-16s' % 'model' + ''.join('%7d' % L for L in LENGTHS))
for m in models:
    row = ''.join(('%7.3f' % acc['ood'][m][L]) if L in acc['ood'].get(m, {}) else '%7s' % '-'
                  for L in LENGTHS)
    print('%-16s' % m + row)
