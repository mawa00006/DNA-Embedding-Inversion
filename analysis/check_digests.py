import h5py, glob, re, sys
ref = {}
for f in glob.glob('data/hg38_multilen_seqdisjoint/train_seqlens_*.h5'):
    L = int(re.search(r'_(\d+)_hg38', f).group(1))
    with h5py.File(f) as h: ref[L] = str(h.attrs.get('input_sequence_sha256', ''))
for m in sys.argv[1:]:
    fs = sorted(glob.glob('data/hg38_multilen_seqdisjoint/train_%s_*.h5' % m),
                key=lambda p: int(re.search(r'_(\d+)_hg38', p).group(1)))
    bad, have = [], []
    for f in fs:
        L = int(re.search(r'_(\d+)_hg38', f).group(1))
        have.append(L)
        with h5py.File(f) as h:
            d = str(h.attrs.get('input_sequence_sha256', ''))
            c = bool(h.attrs.get('generation_complete', False))
            n = h['embeddings'].shape[0] if 'embeddings' in h else 0
        if d != ref.get(L) or not c or n == 0:
            bad.append((L, d[:12], c, n))
    missing = [L for L in sorted(ref) if L not in have]
    print('%-12s %2d/14  missing=%s  bad=%s' % (m, len(fs), missing, bad))
