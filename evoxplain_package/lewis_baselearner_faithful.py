# lewis_baselearner_array.py  (v2 - quota-safe)
# Array-job adaptation of lewis_baselearner_patched.py
# -------------------------------------------------------------------
# WHY v2: v1 hit [Errno 122] Disk quota exceeded. Each split's hyperopt
# handoff cache (_files/split_N/data.pickle) held the full 5-fold slice of
# an 885 x 28613 matrix (~1 GB), and 12 concurrent tasks wrote ~12 GB at
# once -> quota blown. That disk handoff is a vestigial Lewis-ism: they
# pickled data so DISTRIBUTED hyperopt workers could read it. We run serial
# Trials() in ONE process per split, so the data just lives in memory.
#
# v2 removes ALL per-iteration disk writes:
#   - data passed to hyperopt via an in-memory global (no data.pickle)
#   - dropped validation.pickle / validation_xgb.pickle entirely
#     (validation_xgb.pickle was never read downstream -- dead weight)
#   - no _files/ directory at all
# This is BIT-IDENTICAL to v1: fmin is driven by rstate + the loss sequence,
# both unchanged; only writes of an unused file were deleted.
#
# Persistent writes are now ONLY: _individual/split_N.pickle (~40 MB, the
# SHAP values you keep) and _perf/split_N.csv (one row).
# -------------------------------------------------------------------
# (unchanged from v1) one split per $SLURM_ARRAY_TASK_ID; split LABEL N uses
# all_splits[N-1]; N_HYPEROPT=32 to match completed splits 1-8.
# -------------------------------------------------------------------

import gc, os, sys, pickle, random, time, json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import log_loss, balanced_accuracy_score, roc_auc_score
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt.pyll import scope
import xgboost as xgb
import shap

# ======================================================================
# CONFIG
# ======================================================================
DATA_PICKLE = '/home2/chamabens/evoxplain/data/lewis_audit/gene_lewis_exact.pickle'
OUT_DIR     = '/home2/chamabens/evoxplain/results/lewis_audit_faithful'  # SAME dir as pilot
os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(f'{OUT_DIR}/_individual', exist_ok=True)
os.makedirs(f'{OUT_DIR}/_perf', exist_ok=True)

N_SPLITS_TOTAL   = 20           # Lewis: 20  (full design)
TEST_SIZE        = 0.20         # Lewis: same
K_TRAIN_VAL      = 5            # Lewis: same
EARLY_STOP_SIZE  = 0.125        # Lewis: same
N_HYPEROPT       = 256
SEED             = 1            # Lewis: same -> bit-identical splits

random.seed(SEED)
np.random.seed(SEED)

# ----- which split am I? (1-indexed label, matches split_{N}.pickle) -----
_env = os.environ.get('SLURM_ARRAY_TASK_ID')
if _env is not None:
    LABEL = int(_env)
elif len(sys.argv) > 1:
    LABEL = int(sys.argv[1])
else:
    raise SystemExit("No split given. Set $SLURM_ARRAY_TASK_ID or pass an int arg (1..20).")
if not (1 <= LABEL <= N_SPLITS_TOTAL):
    raise SystemExit(f"LABEL {LABEL} out of range 1..{N_SPLITS_TOTAL}")
IDX = LABEL - 1   # index into all_splits

# ======================================================================
# LOAD DATA
# ======================================================================
print(f"[split {LABEL}] Loading {DATA_PICKLE}")
with open(DATA_PICKLE, 'rb') as f:
    X_matrix, y_vector, _ = pickle.load(f)
X_matrix.columns = [f'GENE_EXPRESSION # {c}' for c in X_matrix.columns]
print(f"[split {LABEL}]   X={X_matrix.shape}  y_sum={int(y_vector.sum())}")

# ======================================================================
# HYPEROPT  (in-memory handoff -- NO disk cache)
# ======================================================================
# data for the current split's hyperopt, set just before fmin (module global)
_HP_DATA = None   # tuple: (X_tt, y_tt, X_es, y_es, X_val, y_val)

def hyperopt_performance(X_tt, y_tt, X_es, y_es, X_val, y_val, params):
    losses = []
    preds  = []
    for i in range(K_TRAIN_VAL):
        pos_w = (y_tt[i] == 0).sum() / max((y_tt[i] == 1).sum(), 1)
        dtr = xgb.DMatrix(X_tt[i],  label=y_tt[i])
        des = xgb.DMatrix(X_es[i],  label=y_es[i])
        dva = xgb.DMatrix(X_val[i], label=y_val[i])
        p = {**params,
             'objective': 'binary:logistic',
             'eval_metric': 'logloss',
             'scale_pos_weight': pos_w,
             'seed': SEED,
             'verbosity': 0}
        bst = xgb.train(p, dtr, num_boost_round=10000,
                        evals=[(dtr,'train'), (des,'eval')],
                        early_stopping_rounds=10, verbose_eval=False)
        y_pred = bst.predict(dva, iteration_range=(0, bst.best_iteration + 1))
        pw = (y_val[i] == 0).sum() / max((y_val[i] == 1).sum(), 1)
        sw = [pw if x == 1 else 1 for x in y_val[i]]
        losses.append(log_loss(y_val[i], y_pred, sample_weight=sw))
        preds.append(y_pred)
    mean_loss = np.mean(losses) + np.std(losses) / np.sqrt(len(losses))
    return mean_loss, np.concatenate(preds)

def hyperopt_function(params):
    X_tt, y_tt, X_es, y_es, X_val, y_val = _HP_DATA
    loss, _val_pred = hyperopt_performance(X_tt, y_tt, X_es, y_es, X_val, y_val, params)
    gc.collect()
    return {'loss': loss, 'status': STATUS_OK}

# ======================================================================
# HYPERPARAMETER SEARCH SPACE  (verbatim from Lewis)
# ======================================================================
HP_SPACE = {
    'gamma':            hp.loguniform('gamma', np.log(0.0001), np.log(5)) - 0.0001,
    'max_depth':        scope.int(hp.uniform('max_depth', 1, 11)),
    'subsample':        hp.uniform('subsample', 0.5, 1),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'colsample_bylevel':hp.uniform('colsample_bylevel', 0.5, 1),
    'reg_lambda':       hp.loguniform('reg_lambda', np.log(1), np.log(4)),
    'reg_alpha':        hp.loguniform('reg_alpha', np.log(0.0001), np.log(1)) - 0.0001,
    'eta':              hp.loguniform('eta', np.log(0.01), np.log(0.5)),
}

# ======================================================================
# BUILD THE 20 SPLITS  (identical generator -> all_splits[IDX] is fixed)
# ======================================================================
sss = StratifiedShuffleSplit(n_splits=N_SPLITS_TOTAL, test_size=TEST_SIZE, random_state=SEED)
all_splits = list(sss.split(X_matrix, y_vector))
print(f"[split {LABEL}] Generated {len(all_splits)} splits, running index {IDX} (label split_{LABEL})")

# ======================================================================
# RUN ONE SPLIT
# ======================================================================
t0 = time.time()
print(f"\n========== SPLIT {LABEL}/{N_SPLITS_TOTAL} ==========")
tv_idx, te_idx = all_splits[IDX]
X_tv = X_matrix.iloc[tv_idx]; y_tv = y_vector[tv_idx]
X_te = X_matrix.iloc[te_idx]; y_te = y_vector[te_idx]

# train/val/earlystopping splits
sss_es = StratifiedShuffleSplit(n_splits=1, test_size=EARLY_STOP_SIZE, random_state=SEED)
tr_idx, es_idx = next(sss_es.split(X_tv, y_tv))
X_tt_full = X_tv.iloc[tr_idx]; y_tt_full = y_tv[tr_idx]
X_es_full = X_tv.iloc[es_idx]; y_es_full = y_tv[es_idx]

skf = StratifiedKFold(n_splits=K_TRAIN_VAL, shuffle=True, random_state=SEED)
X_tr_k, X_va_k, y_tr_k, y_va_k = [], [], [], []
for tr_, va_ in skf.split(X_tv, y_tv):
    X_tr_k.append(X_tv.iloc[tr_]); X_va_k.append(X_tv.iloc[va_])
    y_tr_k.append(y_tv[tr_]);      y_va_k.append(y_tv[va_])

X_tt_k, X_es_k, y_tt_k, y_es_k = [], [], [], []
for k in range(K_TRAIN_VAL):
    sss_inner = StratifiedShuffleSplit(n_splits=1, test_size=EARLY_STOP_SIZE, random_state=SEED)
    i_tr, i_es = next(sss_inner.split(X_tr_k[k], y_tr_k[k]))
    X_tt_k.append(X_tr_k[k].iloc[i_tr]); y_tt_k.append(y_tr_k[k][i_tr])
    X_es_k.append(X_tr_k[k].iloc[i_es]); y_es_k.append(y_tr_k[k][i_es])

# hand data to hyperopt IN MEMORY (no disk write)
_HP_DATA = (X_tt_k, y_tt_k, X_es_k, y_es_k, X_va_k, y_va_k)

# hyperopt
print(f"[split {LABEL}]   hyperopt {N_HYPEROPT} iters...")
trials = Trials()
best = fmin(hyperopt_function, HP_SPACE, algo=tpe.suggest,
            max_evals=N_HYPEROPT, trials=trials,
            rstate=np.random.default_rng(SEED), verbose=False, show_progressbar=False)

# free the k-fold data before training final + SHAP
_HP_DATA = None
del X_tr_k, X_va_k, y_tr_k, y_va_k, X_tt_k, X_es_k, y_tt_k, y_es_k
gc.collect()

# train final on full train+val
pos_w = (y_tt_full == 0).sum() / max((y_tt_full == 1).sum(), 1)
final_params = {
    'objective': 'binary:logistic', 'eval_metric': 'logloss',
    'scale_pos_weight': pos_w, 'seed': SEED, 'verbosity': 0,
    'eta': best['eta'], 'gamma': best['gamma'],
    'max_depth': int(best['max_depth']),
    'subsample': best['subsample'],
    'colsample_bytree': best['colsample_bytree'],
    'colsample_bylevel': best['colsample_bylevel'],
    'reg_lambda': best['reg_lambda'], 'reg_alpha': best['reg_alpha'],
}
dtr = xgb.DMatrix(X_tt_full, label=y_tt_full)
des = xgb.DMatrix(X_es_full, label=y_es_full)
dte = xgb.DMatrix(X_te,      label=y_te)
bst = xgb.train(final_params, dtr, num_boost_round=10000,
                evals=[(dtr,'train'), (des,'eval')],
                early_stopping_rounds=10, verbose_eval=False)

y_pred = bst.predict(dte, iteration_range=(0, bst.best_iteration + 1))

# performance
pw_te = (y_te == 0).sum() / max((y_te == 1).sum(), 1)
sw_te = [pw_te if x == 1 else 1 for x in y_te]
wll = log_loss(y_te, y_pred, sample_weight=sw_te)
bacc = balanced_accuracy_score(y_te, (y_pred >= 0.5).astype(int))
auroc = roc_auc_score(y_te, y_pred)
print(f"[split {LABEL}]   AUROC={auroc:.4f}  wll={wll:.4f}  bacc={bacc:.4f}  "
      f"best_iter={bst.best_iteration}  max_depth={int(best['max_depth'])}  time={time.time()-t0:.0f}s")

# per-split one-row perf CSV (no shared file -> no race, no blank rows)
params_logged = {k: (float(v) if not isinstance(v, int) else v)
                 for k, v in final_params.items()
                 if k not in ('objective', 'eval_metric', 'verbosity', 'seed')}
perf_row = pd.DataFrame(
    [[wll, bacc, auroc, bst.best_iteration, json.dumps(params_logged)]],
    index=[f'split_{LABEL}'],
    columns=['weighted_logloss', 'balanced_acc', 'auroc', 'best_iter', 'best_params'])
perf_row.to_csv(f'{OUT_DIR}/_perf/split_{LABEL}.csv')

# SHAP explainer - shap 0.49 API (interventional + probability), verbatim
explainer = shap.TreeExplainer(bst, data=X_tv,
                               feature_perturbation='interventional',
                               model_output='probability')
shap_vals = explainer.shap_values(X_te)

out = f'{OUT_DIR}/_individual/split_{LABEL}.pickle'
with open(out, 'wb') as f:
    pickle.dump({
        'split_idx_tv': tv_idx, 'split_idx_te': te_idx,
        'X_test_index': list(X_te.index),
        'y_test': y_te, 'y_pred': y_pred,
        'shap_values': shap_vals,
        'feature_names': list(X_matrix.columns),
        'best_params': final_params,
        'best_iter': bst.best_iteration,
        'auroc': auroc, 'wll': wll, 'bacc': bacc,
    }, f, protocol=4)
print(f"[split {LABEL}]   saved {out}")
print(f"[split {LABEL}] ========== DONE in {time.time()-t0:.0f}s ==========")
