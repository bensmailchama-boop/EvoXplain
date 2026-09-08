import numpy as np
import glob

paths = sorted(glob.glob('split*/aggregate_split*.npz'))
print('splits found:', len(paths))

z = np.load(paths[0], allow_pickle=True)
print('keys:', sorted(z.files))
print('')

for p in paths[:3]:
    z = np.load(p, allow_pickle=True)
    X = z['expvec_normed_shap'] if 'expvec_normed_shap' in z.files else z['expvec_normed']
    acc = z['test_acc']
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    C = Xn @ Xn.T
    iu = np.triu_indices(len(X), 1)
    cos = C[iu]
    print(p)
    print('   n_runs=%d  unique_acc=%d  acc range %.6f-%.6f'
          % (len(acc), len(np.unique(acc)), acc.min(), acc.max()))
    print('   pairwise cosine: min %.6f  mean %.6f  frac>0.9999 %.3f'
          % (cos.min(), cos.mean(), (cos > 0.9999).mean()))
    if 'run_C_values' in z.files:
        print('   unique C values: %d' % len(np.unique(z['run_C_values'])))
    for k in ('run_seeds', 'run_ids', 'run_params'):
        if k in z.files:
            a = z[k]
            print('   %s: %d unique of %d' % (k, len(np.unique(a)), len(a)))
