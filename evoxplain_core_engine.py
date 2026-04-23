#!/usr/bin/env python3
"""
evoxplain_core_engine.py - Explanation Multiplicity Discovery Engine
Includes support for: Adult Income, Breast Cancer (BC), COMPAS, Colored MNIST, MIMIC-CXR, German Credit, ACS Income,
                       Synthetic Single Mechanism, Synthetic Two Mechanism,
                       TCGA Tumour vs Normal (via tcga_xena_adapter)
Attributions: SHAP, Integrated Gradients (IG), Gini, LIME
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import zipfile
from pathlib import Path
from scipy.spatial.distance import cdist, pdist, cosine

# -----------------------------------------------------------------------------
# LOGGING & UTILITIES
# -----------------------------------------------------------------------------

def log_environment():
    import sys, scipy, sklearn
    print("="*40)
    print("EVOXPLAIN REPRODUCIBILITY LOG")
    print("="*40)
    print(f"python:  {sys.version.split()[0]}")
    print(f"numpy:   {np.__version__}")
    print(f"scipy:   {scipy.__version__}")
    print(f"sklearn: {sklearn.__version__}")
    try:
        import shap
        print(f"shap:    {shap.__version__}")
    except ImportError:
        print("shap:    (not installed)")
    print("="*40, "\n")

def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)

def save_json(obj, path):
    def default(o):
        if isinstance(o, (np.integer, np.int64)): return int(o)
        if isinstance(o, (np.floating, np.float64)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        raise TypeError
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=default)

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)

def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass

# -----------------------------------------------------------------------------
# MODEL WRAPPERS
# -----------------------------------------------------------------------------

class TorchDNN:
    def __init__(self, input_dim, hidden_layers=[100, 50], lr=0.001, epochs=50, 
                 batch_size=64, dropout=0.2, seed=42, device="cpu"):
        import torch
        import torch.nn as nn
        
        self.input_dim = input_dim
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device
        self.seed = seed
        
        torch.manual_seed(seed)
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_layers:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = h_dim
            
        layers.append(nn.Linear(in_dim, 2))
        
        self.model = nn.Sequential(*layers).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr) 
        
    def fit(self, X, y):
        import torch
        self.model.train()
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(self.epochs):
            for batch_X, batch_y in loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()
        return self

    def predict_proba(self, X):
        import torch
        import torch.nn.functional as F
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model(X_tensor)
            probs = F.softmax(logits, dim=1)
            return probs.cpu().numpy()

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    
    def score(self, X, y):
        preds = self.predict(X)
        return np.mean(preds == y)

# -----------------------------------------------------------------------------
# DATA LOADING (UPDATED)
# -----------------------------------------------------------------------------

def load_dataset(dataset_name: str, drop_features=None, data_cache_dir=None, data_path=None,
                 acs_states=None, acs_year=None, tcga_gz_path=None, tcga_top_n=1000,
                 tcga_subtype_path=None):
    from sklearn.preprocessing import StandardScaler, OneHotEncoder

    # --- 1. ADULT INCOME ---
    if dataset_name.lower() in ("adult", "adult_income", "adult-income"):
        train_path = "data/adult.data"
        if not os.path.exists(train_path) and os.path.exists("data/adult.csv"):
            df = pd.read_csv("data/adult.csv")
        else:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
            cols = ["age", "workclass", "fnlwgt", "education", "education-num", 
                    "marital-status", "occupation", "relationship", "race", "sex", 
                    "capital-gain", "capital-loss", "hours-per-week", "native-country", "income"]
            df = pd.read_csv(train_path if os.path.exists(train_path) else url, names=cols, na_values=" ?")

        df = df.dropna()
        y = (df["income"].str.contains(">50K")).astype(int).values
        X_df = df.drop("income", axis=1)

    # --- 2. BREAST CANCER (BC) ---
    elif dataset_name.lower() in ("bc", "breast_cancer", "breast-cancer"):
        if os.path.exists("data/breast_cancer.csv"):
            df = pd.read_csv("data/breast_cancer.csv")
            target_col = "target" if "target" in df.columns else df.columns[-1]
            y = df[target_col].values
            X_df = df.drop(columns=[target_col])
        else:
            from sklearn.datasets import load_breast_cancer
            data = load_breast_cancer()
            X_df = pd.DataFrame(data.data, columns=data.feature_names)
            y = 1 - data.target 
    
    # --- 3. COMPAS (Recidivism) ---
    elif dataset_name.lower() == "compas":
        if os.path.exists("data/compas-scores-two-years.csv"):
            path = "data/compas-scores-two-years.csv"
        elif os.path.exists("data/compas.csv"):
            path = "data/compas.csv"
        else:
            raise FileNotFoundError("COMPAS dataset not found in data/ folder. Please upload 'compas-scores-two-years.csv'.")
        
        df = pd.read_csv(path)
        df = df[
            (df['days_b_screening_arrest'] <= 30) & 
            (df['days_b_screening_arrest'] >= -30) & 
            (df['is_recid'] != -1) & 
            (df['c_charge_degree'] != "O") & 
            (df['score_text'] != 'N/A')
        ]
        
        keep_cols = ['sex', 'age', 'race', 'priors_count', 'c_charge_degree', 'two_year_recid']
        df = df[keep_cols]
        y = df['two_year_recid'].values
        X_df = df.drop('two_year_recid', axis=1)

    # --- 4. COLORED MNIST ---
    elif dataset_name.lower() in ("cmnist", "colored_mnist", "colored-mnist", "coloredmnist"):
        if os.path.exists("data/colored_mnist.npz"):
            data = np.load("data/colored_mnist.npz")
            X_images = data["images"]
            y = data["labels"]
            if X_images.ndim == 4:
                X_images = X_images.reshape(X_images.shape[0], -1)
            elif X_images.ndim == 3:
                X_images = X_images.reshape(X_images.shape[0], -1)
            
            n_features = X_images.shape[1]
            if n_features == 28 * 28 * 3:
                feature_names = [f"px_{i//3}_{['R','G','B'][i%3]}" for i in range(n_features)]
            else:
                feature_names = [f"px_{i}" for i in range(n_features)]
            
            if X_images.max() > 1.0:
                X_images = X_images.astype(np.float32) / 255.0
            
            X_df = pd.DataFrame(X_images, columns=feature_names)
        elif os.path.exists("data/colored_mnist.csv"):
            df = pd.read_csv("data/colored_mnist.csv")
            target_col = "label" if "label" in df.columns else df.columns[-1]
            y = df[target_col].values
            X_df = df.drop(columns=[target_col])
        else:
            print("[Data] Generating Colored MNIST from torchvision MNIST...")
            try:
                import torch
                from torchvision import datasets, transforms
            except ImportError:
                raise ImportError("Please install torchvision: pip install torchvision")
            
            ensure_dir("data")
            mnist_train = datasets.MNIST("data", train=True, download=True)
            mnist_test = datasets.MNIST("data", train=False, download=True)
            
            all_images = np.concatenate([mnist_train.data.numpy(), mnist_test.data.numpy()], axis=0)
            all_labels = np.concatenate([mnist_train.targets.numpy(), mnist_test.targets.numpy()], axis=0)
            
            y = (all_labels >= 5).astype(int)
            rng = np.random.RandomState(42)
            n_samples = len(y)
            X_colored = np.zeros((n_samples, 28, 28, 3), dtype=np.float32)
            
            for i in range(n_samples):
                img = all_images[i].astype(np.float32) / 255.0
                if rng.random() < 0.8:
                    if y[i] == 0: X_colored[i, :, :, 0] = img
                    else: X_colored[i, :, :, 1] = img
                else:
                    if y[i] == 0: X_colored[i, :, :, 1] = img
                    else: X_colored[i, :, :, 0] = img
            
            X_flat = X_colored.reshape(n_samples, -1)
            feature_names = [f"px_{i//3}_{['R','G','B'][i%3]}" for i in range(X_flat.shape[1])]
            X_df = pd.DataFrame(X_flat, columns=feature_names)
            np.savez("data/colored_mnist.npz", images=X_colored, labels=y)
            print(f"[Data] Cached Colored MNIST to data/colored_mnist.npz ({n_samples} samples)")

    # --- 5. MIMIC-CXR ---
    elif dataset_name.lower() in ("mimic", "mimic-cxr", "cxr"):
        data_path = "data/mimic_pneumo.npz"
        if os.path.exists(data_path):
            print(f"[Data] Loading MIMIC-CXR from {data_path}...")
            data = np.load(data_path)
            X_images = data["images"] 
            y = data["labels"]
            if X_images.ndim > 2:
                n_samples = X_images.shape[0]
                X_flat = X_images.reshape(n_samples, -1)
            else:
                X_flat = X_images
            feature_names = [f"px_{i}" for i in range(X_flat.shape[1])]
            X_df = pd.DataFrame(X_flat, columns=feature_names)
            if X_flat.max() > 1.0:
                 X_df = X_df / 255.0
        else:
            raise FileNotFoundError(f"MIMIC file not found at {data_path}.")

    # --- 6. GERMAN CREDIT ---
    elif dataset_name.lower() in ("german_credit", "german-credit", "credit-g", "creditg"):
        df = None
        if data_path and os.path.exists(data_path):
            df = pd.read_csv(data_path)
        if df is None:
            try:
                import openml
                if data_cache_dir: openml.config.set_root_cache_directory(data_cache_dir)
                oml_dataset = openml.datasets.get_dataset(31, download_data=True, download_qualities=False, download_features_meta_data=False)
                df, _, _, _ = oml_dataset.get_data(dataset_format="dataframe")
            except: pass
        if df is None:
            for candidate in ["data/german_credit.csv", "data/credit-g.csv", "data/dataset_31_credit-g.csv"]:
                if os.path.exists(candidate):
                    df = pd.read_csv(candidate)
                    break
        if df is None: raise FileNotFoundError("German Credit dataset not found.")

        target_col = next((c for c in ["class", "target", "credit_risk"] if c in df.columns), df.columns[-1])
        raw_target = df[target_col]
        if raw_target.dtype == object or str(raw_target.dtype) == "category":
            target_map = {"good": 1, "bad": 0}
            if set(raw_target.unique()) <= set(target_map.keys()):
                y = raw_target.map(target_map).values.astype(int)
            else:
                unique_vals = sorted(raw_target.unique())
                y = (raw_target == unique_vals[-1]).astype(int).values
        else:
            y = raw_target.values.astype(int)
        X_df = df.drop(columns=[target_col])

    # --- 7. ACS INCOME ---
    elif dataset_name.lower() in ("acs_income", "acs-income", "acsincome", "folktables", "acs"):
        states = [s.strip().upper() for s in (acs_states or "CA").split(",")]
        year = acs_year or "2018"
        states_tag = "_".join(states)
        df = None
        if data_path and os.path.exists(data_path): df = pd.read_csv(data_path)
        if df is None:
            try:
                from folktables import ACSDataSource, ACSIncome
                root_dir = data_cache_dir if data_cache_dir else "data/folktables_cache"
                data_source = ACSDataSource(survey_year=year, horizon='1-Year', survey='person', root_dir=root_dir)
                acs_data = data_source.get_data(states=states, download=True)
                features, label, group = ACSIncome.df_to_numpy(acs_data)
                acs_feature_names = ['AGEP', 'COW', 'SCHL', 'MAR', 'OCCP', 'POBP', 'RELP', 'WKHP', 'SEX', 'RAC1P']
                df = pd.DataFrame(features, columns=acs_feature_names)
                df['target'] = label.astype(int)
                cache_csv = os.path.join(data_cache_dir or "data", f"acs_income_{year}_{states_tag}.csv")
                ensure_dir(os.path.dirname(cache_csv))
                df.to_csv(cache_csv, index=False)
            except: pass
        if df is None:
            for candidate in [f"data/acs_income_{year}_{states_tag}.csv", "data/acs_income.csv", f"data/acs_income_{year}.csv"]:
                if os.path.exists(candidate):
                    df = pd.read_csv(candidate)
                    break
        if df is None: raise FileNotFoundError("ACS Income dataset not found.")

        target_col = next((c for c in ["target", "PINCP", "income"] if c in df.columns), df.columns[-1])
        y = df[target_col].values.astype(int)
        X_df = df.drop(columns=[target_col])

        low_card_categoricals = ['COW', 'MAR', 'RELP', 'SEX', 'RAC1P', 'SCHL']
        for col in low_card_categoricals:
            if col in X_df.columns: X_df[col] = X_df[col].astype(int).astype(str)

    # --- 8. SYNTHETIC SINGLE ---
    elif dataset_name.lower() in ("synthetic_single", "synthetic_single_mechanism"):
        rng = np.random.RandomState(42)
        n_samples, n_features, n_causal = 5000, 20, 5
        X = rng.normal(0, 1, size=(n_samples, n_features))
        w_true = np.zeros(n_features)
        causal_idx = rng.choice(n_features, n_causal, replace=False)
        w_true[causal_idx] = rng.uniform(1.0, 2.0, size=n_causal) * rng.choice([-1, 1], size=n_causal)
        logits = X @ w_true + rng.normal(0, 0.2, size=n_samples)
        y = (logits > 0).astype(int)
        feature_names = [f"feat_{i}" for i in range(n_features)]
        return X, y, feature_names

    # --- 9. SYNTHETIC TWO ---
    elif dataset_name.lower() in ("synthetic_two", "synthetic_two_mechanism"):
        rng = np.random.RandomState(42)
        n_samples, n_features, block = 6000, 20, 5
        z = rng.normal(0, 1, size=n_samples)
        X = rng.normal(0, 1, size=(n_samples, n_features)) * 0.5
        for j in range(block): X[:, j] = z + rng.normal(0, 0.5, size=n_samples)
        for j in range(block, 2*block): X[:, j] = z + rng.normal(0, 0.5, size=n_samples)
        logits = 2.0 * z + rng.normal(0, 0.5, size=n_samples)
        y = (logits > 0).astype(int)
        feature_names = [f"feat_{i}" for i in range(n_features)]
        return X, y, feature_names

    # --- 10. SYNTHETIC MIXTURE ---
    elif dataset_name.lower() in ("synthetic_two_mixture", "synthetic_two_mechanism_mixture"):
        rng = np.random.RandomState(42)
        n_samples, n_features, block = 6000, 20, 5
        zA = rng.normal(0, 1, size=n_samples)
        zB = rng.normal(0, 1, size=n_samples)
        g = rng.binomial(1, 0.5, size=n_samples) 
        X = rng.normal(0, 1, size=(n_samples, n_features)) * 0.8
        for j in range(block): X[g == 0, j] = 1.5 * zA[g == 0] + rng.normal(0, 0.3, size=np.sum(g == 0))
        for j in range(block, 2*block): X[g == 1, j] = 1.5 * zB[g == 1] + rng.normal(0, 0.3, size=np.sum(g == 1))
        logits = np.zeros(n_samples)
        logits[g == 0] = 2.0 * zA[g == 0] + rng.normal(0, 0.3, size=np.sum(g == 0))
        logits[g == 1] = 2.0 * zB[g == 1] + rng.normal(0, 0.3, size=np.sum(g == 1))
        y = (logits > 0).astype(int)
        feature_names = [f"feat_{i}" for i in range(n_features)]
        return X, y, feature_names
    # --- 11. TCGA TUMOUR vs NORMAL (Xena TOIL RSEM TPM) ---
    elif dataset_name.lower() in ("tcga", "tcga_tumour_normal", "tcga-tumour-normal", "tcga_tumor_normal"):
        try:
            from tcga_xena_adapter import load_tcga_for_evoxplain
        except ImportError:
            raise ImportError(
                "[Data] tcga_xena_adapter.py not found. "
                "Ensure it is in the same directory as evoxplain_core_engine.py."
            )

        gz = tcga_gz_path or data_path
        if not gz:
            raise ValueError(
                "[Data] TCGA dataset requires --tcga_gz_path (path to tcga_RSEM_gene_tpm.gz)."
            )

        print(f"[Data] Loading TCGA expression matrix: {gz}")
        print(f"[Data] top_n={tcga_top_n}, standardize=True, log2_transform=False, label_source=barcode")

        X, y, feature_names = load_tcga_for_evoxplain(
            gz_path        = gz,
            label_source   = "barcode",
            top_n          = tcga_top_n,
            standardize    = True,
            log2_transform = False,   # Xena TOIL is already log2(TPM+0.001)
            drop_unknown   = True,
        )

        # --- Sanity checks ---
        print(f"[Data] TCGA raw loaded shape  : X={X.shape}, y={y.shape}")
        if X.shape[0] == 0:
            raise ValueError("[Data] TCGA: X is empty after loading. Check gz_path and adapter.")
        if np.isnan(X).any() or np.isinf(X).any():
            raise ValueError("[Data] TCGA: NaN or Inf detected in X after preprocessing.")
        unique_classes, class_counts = np.unique(y, return_counts=True)
        if len(unique_classes) < 2:
            raise ValueError(f"[Data] TCGA: fewer than 2 classes found after label inference: {unique_classes}")
        print(f"[Data] TCGA class distribution:")
        for cls, cnt in zip(unique_classes, class_counts):
            label_str = "Tumour" if cls == 1 else "Normal"
            pct = 100.0 * cnt / len(y)
            print(f"  Label {cls} ({label_str}): {cnt:,} samples ({pct:.1f}%)")
        if min(class_counts) / max(class_counts) < 0.15:
            print(f"[Data] WARNING: severe class imbalance detected. "
                  f"Consider stratified splits (already enforced) and class_weight='balanced' for LR.")
        print(f"[Data] TCGA final X shape     : {X.shape}")
        print(f"[Data] TCGA feature_names (10): {feature_names[:10]}")

        # Return early — adapter has already standardised, bypass global preprocessing block
        return X, y, feature_names

    # --- 12. TCGA BRCA Luminal vs Basal (Xena TOIL RSEM TPM + PAM50 subtypes) ---
    elif dataset_name.lower() in ("tcga_brca_luminal_vs_basal", "tcga-brca-luminal-vs-basal",
                                   "brca_luminal_basal", "brca_lum_bas"):
        try:
            from tcga_xena_adapter import load_tcga_for_evoxplain
        except ImportError:
            raise ImportError(
                "[Data] tcga_xena_adapter.py not found. "
                "Ensure it is in the same directory as evoxplain_core_engine.py."
            )

        gz = tcga_gz_path or data_path
        if not gz:
            raise ValueError(
                "[Data] tcga_brca_luminal_vs_basal requires --tcga_gz_path "
                "(path to tcga_RSEM_gene_tpm.gz)."
            )
        if not tcga_subtype_path:
            raise ValueError(
                "[Data] tcga_brca_luminal_vs_basal requires --tcga_subtype_path "
                "(path to PAM50 subtype annotation TSV/CSV)."
            )

        print(f"[Data] === TCGA BRCA Luminal vs Basal ===")
        print(f"[Data] Expression matrix : {gz}")
        print(f"[Data] Subtype labels    : {tcga_subtype_path}")
        print(f"[Data] top_n={tcga_top_n}, standardize=True, log2_transform=False")
        print(f"[Data] Label mapping: LumA→0, LumB→0, Basal→1 | Exclude: HER2, Normal-like, Unknown")

        # ── Step 1: load expression via existing adapter (tumour samples only) ──
        # We request all samples and will filter by PAM50 subtype below.
        # label_source='barcode' infers tumour/normal from barcode; we bypass
        # that label and replace with PAM50 — expression preprocessing is reused.
        X_full, _y_barcode, feature_names = load_tcga_for_evoxplain(
            gz_path        = gz,
            label_source   = "barcode",
            top_n          = tcga_top_n,
            standardize    = False,      # delay standardisation until after filtering
            log2_transform = False,
            drop_unknown   = False,      # keep all samples; we filter by subtype
        )
        # The adapter returns samples whose barcodes could be parsed.
        # We need sample IDs to align with subtype file.
        # Convention: adapter stores sample IDs as the row index of the expression
        # DataFrame before conversion. Re-read to get them.
        print(f"[Data] Expression matrix loaded: {X_full.shape[0]} samples × {X_full.shape[1]} genes")

        # ── Step 2: recover sample IDs from the adapter ──────────────────────
        # load_tcga_for_evoxplain does not return sample_ids; we re-open the gz
        # for the barcode list only (header row), which is fast.
        import gzip as _gzip
        print(f"[Data] Re-reading sample barcodes from gz header …")
        with _gzip.open(gz, 'rt') as _fh:
            _header = _fh.readline().strip().split('\t')
        # First column is gene identifier; remaining are sample barcodes.
        raw_barcodes = _header[1:]
        # TCGA barcodes are like TCGA-3C-AAAU-01A-11R-A41B-07
        # Trim to 15-character TCGA sample ID (positions 0-14) for matching.
        def _trim_barcode(bc):
            parts = bc.split('-')
            return '-'.join(parts[:4]) if len(parts) >= 4 else bc

        raw_trimmed = [_trim_barcode(b) for b in raw_barcodes]
        # The adapter drops columns silently; we have no index map unless we
        # redo the full parse.  Instead we use an ordered approach: rebuild
        # the expression matrix with barcode index from scratch using pandas.
        print(f"[Data] Parsing expression matrix with barcode index (this may take ~30s) …")
        _chunks = []
        with _gzip.open(gz, 'rt') as _fh:
            _expr_df = pd.read_csv(_fh, sep='\t', index_col=0)
        # _expr_df: genes × samples  →  transpose to samples × genes
        _expr_df = _expr_df.T
        _expr_df.index = [_trim_barcode(b) for b in _expr_df.index]
        print(f"[Data] Expression (samples × genes): {_expr_df.shape}")

        # ── Step 3: load subtype annotation ──────────────────────────────────
        _sub_path = Path(tcga_subtype_path)
        if not _sub_path.exists():
            raise FileNotFoundError(f"[Data] Subtype file not found: {_sub_path}")

        _sep = '\t' if str(_sub_path).endswith('.tsv') or str(_sub_path).endswith('.txt') else ','
        _sub_df = pd.read_csv(_sub_path, sep=_sep, dtype=str)
        print(f"[Data] Subtype file loaded: {_sub_df.shape[0]} rows, columns={list(_sub_df.columns[:8])}")

        # Auto-detect relevant columns (case-insensitive search)
        _cols_lower = {c.lower(): c for c in _sub_df.columns}

        # Sample ID column
        _id_col = None
        for _candidate in ('sample', 'samplename', 'sample_id', 'tcga_id',
                            'barcode', 'submitter_id', 'patient', 'id'):
            if _candidate in _cols_lower:
                _id_col = _cols_lower[_candidate]; break
        if _id_col is None:
            _id_col = _sub_df.columns[0]
            print(f"[Data] WARNING: could not auto-detect sample ID column; using '{_id_col}'")

        # PAM50 subtype column
        _sub_col = None
        for _candidate in ('pam50', 'pam50subtype', 'subtype', 'brca_subtype',
                            'oncotree_code', 'cancer_type_detailed', 'pathology_subtype'):
            if _candidate in _cols_lower:
                _sub_col = _cols_lower[_candidate]; break
        if _sub_col is None:
            raise ValueError(
                f"[Data] Cannot find PAM50 subtype column in {_sub_path}. "
                f"Available columns: {list(_sub_df.columns)}"
            )

        print(f"[Data] Using sample ID column : '{_id_col}'")
        print(f"[Data] Using subtype column   : '{_sub_col}'")

        _sub_df[_id_col] = _sub_df[_id_col].str.strip()
        _sub_df[_sub_col] = _sub_df[_sub_col].str.strip()

        # Trim subtype sample IDs to 15-char barcode style
        _sub_df['_sample_trimmed'] = _sub_df[_id_col].apply(_trim_barcode)
        _sub_df = _sub_df.drop_duplicates('_sample_trimmed').set_index('_sample_trimmed')

        print(f"[Data] Subtype distribution (raw):")
        for _st, _cnt in _sub_df[_sub_col].value_counts().items():
            print(f"  {_st}: {_cnt}")

        # ── Step 4: define PAM50 label mapping ───────────────────────────────
        # Flexible matching: accept common variants of subtype names
        _LABEL_MAP = {}
        for _st in _sub_df[_sub_col].unique():
            _st_norm = str(_st).strip().lower().replace('-', '').replace(' ', '').replace('_', '')
            if _st_norm in ('luminal_a', 'luminala', 'luma', 'luml_a', 'lum_a'):
                _LABEL_MAP[_st] = 0
            elif _st_norm in ('luminal_b', 'luminalb', 'lumb', 'luml_b', 'lum_b'):
                _LABEL_MAP[_st] = 0
            elif _st_norm in ('basal', 'basallike', 'basal-like', 'basallikebrca', 'tnbc'):
                _LABEL_MAP[_st] = 1
            # HER2, Normal-like, Unknown → excluded (not in map)

        print(f"[Data] Effective PAM50 label map: {_LABEL_MAP}")
        included_subtypes = set(_LABEL_MAP.keys())
        _sub_df = _sub_df[_sub_df[_sub_col].isin(included_subtypes)].copy()
        _sub_df['_label'] = _sub_df[_sub_col].map(_LABEL_MAP).astype(int)
        print(f"[Data] Samples retained after PAM50 filtering: {len(_sub_df)}")

        # ── Step 5: align expression and subtype ─────────────────────────────
        _common = _expr_df.index.intersection(_sub_df.index)
        if len(_common) == 0:
            raise ValueError(
                "[Data] Zero samples overlap between expression matrix and subtype file. "
                "Check barcode trimming and ID column."
            )
        print(f"[Data] Expression barcodes          : {len(_expr_df)}")
        print(f"[Data] Subtype barcodes (filtered)  : {len(_sub_df)}")
        print(f"[Data] Overlapping samples          : {len(_common)}")

        _expr_aligned = _expr_df.loc[_common].copy()
        _sub_aligned  = _sub_df.loc[_common].copy()

        # ── Step 6: filter to tumour samples only (code 01x in barcode pos 4) ─
        # TCGA barcode position 3 (0-indexed) contains sample type:
        # 01x = primary solid tumour. We only keep tumour samples to stay
        # consistent with the BRCA subtype biology (all PAM50 subtypes are tumour).
        def _is_tumour(bc):
            parts = bc.split('-')
            return len(parts) >= 4 and parts[3].startswith('0')
        _tumour_mask = pd.Series(_common).apply(_is_tumour).values
        _common_tumour = _common[_tumour_mask]
        if len(_common_tumour) < len(_common):
            print(f"[Data] Dropping {len(_common) - len(_common_tumour)} non-tumour barcodes "
                  f"(keeping {len(_common_tumour)} tumour samples)")
            _expr_aligned = _expr_df.loc[_common_tumour].copy()
            _sub_aligned  = _sub_df.loc[_common_tumour].copy()

        sample_ids = list(_expr_aligned.index)
        y = _sub_aligned['_label'].values.astype(int)

        # ── Step 7: select top-n variable genes ──────────────────────────────
        _X_raw = _expr_aligned.values.astype(np.float64)
        _gene_vars = np.var(_X_raw, axis=0)
        _top_idx = np.argsort(_gene_vars)[::-1][:tcga_top_n]
        _top_idx_sorted = np.sort(_top_idx)
        _X_filtered = _X_raw[:, _top_idx_sorted]
        feature_names = list(_expr_aligned.columns[_top_idx_sorted])

        # ── Step 8: standardise ──────────────────────────────────────────────
        from sklearn.preprocessing import StandardScaler as _SS
        _scaler = _SS()
        X = _scaler.fit_transform(_X_filtered)

        # ── Step 9: sanity checks ─────────────────────────────────────────────
        print(f"\n[Data] === TCGA BRCA Luminal vs Basal — Final Summary ===")
        print(f"[Data] Dataset         : tcga_brca_luminal_vs_basal")
        print(f"[Data] X shape         : {X.shape}")
        print(f"[Data] y shape         : {y.shape}")
        print(f"[Data] n_genes (top-n) : {len(feature_names)}")
        print(f"[Data] feature_names[:5]: {feature_names[:5]}")

        _unique_cls, _cls_counts = np.unique(y, return_counts=True)
        _cls_names = {0: 'Luminal (A+B)', 1: 'Basal-like'}
        for _cls, _cnt in zip(_unique_cls, _cls_counts):
            _pct = 100.0 * _cnt / len(y)
            print(f"[Data] Class {_cls} ({_cls_names.get(_cls, '?')}): {_cnt:,} samples ({_pct:.1f}%)")

        if len(_unique_cls) < 2:
            raise ValueError("[Data] BRCA: fewer than 2 classes after filtering — check subtype file.")

        _min_class = min(_cls_counts)
        if _min_class < 50:
            raise ValueError(
                f"[Data] BRCA: smallest class has only {_min_class} samples (<50). "
                f"Check subtype file and barcode alignment."
            )

        _imbalance_ratio = min(_cls_counts) / max(_cls_counts)
        if _imbalance_ratio < 0.15:
            print(f"[Data] WARNING: severe class imbalance (ratio={_imbalance_ratio:.2f}). "
                  f"Consider --lr_C_mode grid or class_weight='balanced'.")

        if np.isnan(X).any() or np.isinf(X).any():
            raise ValueError("[Data] BRCA: NaN or Inf detected in X after preprocessing.")

        print(f"[Data] NaN in X: {np.isnan(X).any()}  |  Inf in X: {np.isinf(X).any()}")
        print(f"[Data] === TCGA BRCA loader complete ===\n")

        # Return — preprocessing already done; bypass global preprocessing block
        return X, y, feature_names

    else:
        raise ValueError(f"Dataset {dataset_name} not supported.")

    # --- GLOBAL FEATURE DROPPING ---
    if drop_features:
        drop_list = [f.strip() for f in drop_features.split(",")]
        print(f"[Data] Explicitly dropping features: {drop_list}")
        X_df = X_df.drop(columns=drop_list, errors="ignore")

    # --- PREPROCESSING ---
    numeric_features = X_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = X_df.select_dtypes(exclude=[np.number]).columns.tolist()

    X_numeric = X_df[numeric_features].values if numeric_features else np.zeros((len(X_df), 0))
    if categorical_features:
        try:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        except TypeError:
            encoder = OneHotEncoder(sparse=False, handle_unknown="ignore")
        X_cat = encoder.fit_transform(X_df[categorical_features])
        cat_names = encoder.get_feature_names_out(categorical_features)
    else:
        X_cat = np.zeros((len(X_df), 0))
        cat_names = []
    
    X = np.hstack([X_numeric, X_cat])
    feature_names = numeric_features + list(cat_names)
    
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y, feature_names

# -----------------------------------------------------------------------------
# BOUNDARY SET LOGIC
# -----------------------------------------------------------------------------

def get_boundary_set(X, y, args, split_seed):
    import time, tempfile
    out_dir = Path(args.output_dir) / f"split{split_seed}"
    ensure_dir(out_dir)
    boundary_path = out_dir / "boundary_set.npz"

    if boundary_path.exists() and args.boundary_cache:
        for attempt in range(5):
            try:
                data = np.load(boundary_path, allow_pickle=True)
                _ = data["boundary_indices"]
                return data
            except (EOFError, zipfile.BadZipFile, KeyError, ValueError) as e:
                time.sleep(2 + attempt * 2)
        print(f"  [Boundary] Cached file corrupt after 5 retries. Regenerating...")

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=split_seed, stratify=y)

    seed_everything(args.boundary_seed)
    from sklearn.ensemble import RandomForestClassifier
    ref_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=args.boundary_seed)
    ref_model.fit(X_train, y_train)
    
    if args.boundary_source == "test":
        pool_indices = np.arange(len(y_test))
        probs = ref_model.predict_proba(X_test)[:, 1]
        y_pool = y_test
    else:
        pool_indices = np.arange(len(y_train))
        probs = ref_model.predict_proba(X_train)[:, 1]
        y_pool = y_train

    if args.boundary_method == "prob_band":
        mask = (probs >= args.boundary_prob_low) & (probs <= args.boundary_prob_high)
        selected_indices = pool_indices[mask]
        if len(selected_indices) < args.boundary_k:
            margins = np.abs(probs - 0.5)
            selected_indices = pool_indices[np.argsort(margins)[:args.boundary_k]]
        elif len(selected_indices) > args.boundary_k:
            rng = np.random.RandomState(args.boundary_seed)
            selected_indices = rng.choice(selected_indices, args.boundary_k, replace=False)

    selected_indices = np.sort(selected_indices)

    fd, tmp_path = tempfile.mkstemp(dir=out_dir, suffix=".npz")
    os.close(fd)
    np.savez(tmp_path, boundary_indices=selected_indices, boundary_y_true=y_pool[selected_indices])
    os.replace(tmp_path, boundary_path)

    return np.load(boundary_path, allow_pickle=True)

# -----------------------------------------------------------------------------
# TRAINING & EXPLAINING
# -----------------------------------------------------------------------------

def make_model(args, input_dim, rng_seed, C_override=None, l1_override=None):
    if args.model == "rf":
        from sklearn.ensemble import RandomForestClassifier
        mf = args.rf_max_features
        if mf not in ["sqrt", "log2"]:
            try:
                mf = float(mf)
                if mf > 1.0: mf = int(mf)
            except: pass
        return RandomForestClassifier(
            n_estimators=args.rf_n_estimators, max_depth=args.rf_max_depth,
            max_features=mf, random_state=rng_seed, n_jobs=1
        )
    elif args.model == "xgb":
        try: from xgboost import XGBClassifier
        except ImportError: raise ImportError("Please install xgboost: pip install xgboost")
        return XGBClassifier(n_estimators=args.rf_n_estimators, max_depth=args.rf_max_depth if args.rf_max_depth else 6,
                             learning_rate=0.1, random_state=rng_seed, n_jobs=1)
    elif args.model == "svm":
        from sklearn.svm import SVC
        return SVC(kernel='rbf', probability=True, random_state=rng_seed, C=1.0)
    elif args.model == "dnn":
        layers = [int(x) for x in args.dnn_layers.split(",")]
        return TorchDNN(input_dim=input_dim, hidden_layers=layers, lr=args.dnn_lr, epochs=args.dnn_epochs,
                        batch_size=args.dnn_batch_size, dropout=args.dnn_dropout, seed=rng_seed)
    elif args.model == "lr":
        from sklearn.linear_model import LogisticRegression
        penalty = args.lr_penalty if args.lr_penalty != "none" else None
        C_val = C_override if C_override is not None else args.lr_C
        
        # Determine l1_ratio if Elastic Net is used
        l1_val = l1_override if l1_override is not None else args.lr_l1_ratio
        
        # Force saga solver if elasticnet is requested or if just generally desired
        solver = "saga" if penalty == "elasticnet" or args.lr_penalty == "elasticnet" else "lbfgs"
        
        return LogisticRegression(
            C=C_val, 
            penalty=penalty, 
            solver=solver, 
            l1_ratio=l1_val, # Only used if penalty='elasticnet'
            max_iter=args.lr_max_iter, 
            random_state=rng_seed, 
            n_jobs=1
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")


def compute_attributions(model, X_bg, X_target, args, rng=None, attribution=None):
    """
    Computes feature attributions robustly across model types and methods.
    Normalizes output to a 2D array of shape (N, D) where D == X_target.shape[1].
    attribution: override args.attribution (used in multi-lens mode).
    """
    import numpy as np

    attr = attribution if attribution is not None else args.attribution

    # ---------------------------------------------------------
    # 1. GINI IMPORTANCE (Global - Tree Models Only)
    # ---------------------------------------------------------
    if attr == "gini":
        if args.model not in ["rf", "xgb"]:
            raise ValueError("Gini importance is only supported for tree models ('rf', 'xgb').")
        
        global_gini = model.feature_importances_ # Shape (D,)
        sv = np.tile(global_gini, (X_target.shape[0], 1))
        return sv

    # ---------------------------------------------------------
    # 2. INTEGRATED GRADIENTS (Captum - PyTorch DNN Only)
    # ---------------------------------------------------------
    elif attr == "ig":
        if args.model != "dnn":
            raise ValueError("Integrated Gradients (ig) is only supported for 'dnn' model.")
        
        try:
            import torch
            from captum.attr import IntegratedGradients
        except ImportError:
            raise ImportError("Please install captum: pip install captum")
            
        model.model.eval()
        ig = IntegratedGradients(model.model)
        
        X_target_tensor = torch.tensor(X_target, dtype=torch.float32).to(model.device)
        baseline = torch.zeros_like(X_target_tensor).to(model.device)
        
        # Attribute to positive class
        attributions, _ = ig.attribute(X_target_tensor, baseline, target=1, return_convergence_delta=True)
        sv = attributions.cpu().detach().numpy()
        
        if not args.shap_signed:
            sv = np.abs(sv)
            
        return sv

    # ---------------------------------------------------------
    # 3. SHAP (Default)
    # ---------------------------------------------------------
    elif attr == "shap":
        import shap
        import torch

        if args.model in ["rf", "xgb"]:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_target, check_additivity=False)
            sv = shap_vals[1] if isinstance(shap_vals, list) and len(shap_vals) > 1 else (shap_vals[0] if isinstance(shap_vals, list) else shap_vals)

        elif args.model == "lr":
            explainer = shap.LinearExplainer(model, X_bg)
            shap_vals = explainer.shap_values(X_target)
            sv = shap_vals[1] if isinstance(shap_vals, list) and len(shap_vals) > 1 else (shap_vals[0] if isinstance(shap_vals, list) else shap_vals)

        elif args.model == "svm":
            bg_summary = shap.kmeans(X_bg, 10)
            explainer = shap.KernelExplainer(model.predict_proba, bg_summary)
            shap_vals = explainer.shap_values(X_target, nsamples=100)
            sv = shap_vals[1] if isinstance(shap_vals, list) and len(shap_vals) > 1 else (shap_vals[0] if isinstance(shap_vals, list) else shap_vals)

        elif args.model == "dnn":
            import hashlib
            bg_size = min(len(X_bg), args.shap_background_size)
            if rng is None: rng = np.random.RandomState(0)
            bg_idx = rng.choice(len(X_bg), bg_size, replace=False)

            X_bg_tensor = torch.tensor(X_bg[bg_idx], dtype=torch.float32).to(model.device)
            X_target_tensor = torch.tensor(X_target, dtype=torch.float32).to(model.device)

            explainer = shap.DeepExplainer(model.model, X_bg_tensor)
            shap_vals = explainer.shap_values(X_target_tensor, check_additivity=False)
            h = hashlib.md5(bg_idx.tobytes()).hexdigest()[:8]
            print(f"[DEBUG] split={args.split_seed} bg_hash={h}")
            sv = shap_vals[1] if isinstance(shap_vals, list) and len(shap_vals) > 1 else (shap_vals[0] if isinstance(shap_vals, list) else shap_vals)
            if not isinstance(sv, np.ndarray): sv = sv.cpu().detach().numpy()

        else:
            raise ValueError("Unsupported model for SHAP")

        if not isinstance(sv, np.ndarray): sv = np.array(sv)

        if sv.ndim == 3:
            if sv.shape[2] == 2: sv = sv[:, :, 1]
            elif sv.shape[2] == 1: sv = np.squeeze(sv, axis=2)
            else: raise ValueError(f"SHAP output is 3D with {sv.shape[2]} classes.")

        if sv.ndim != 2: raise ValueError(f"SHAP output must be 2D. Got shape {sv.shape}.")
        if not args.shap_signed: sv = np.abs(sv)
        if sv.shape[1] != X_target.shape[1]: raise ValueError("SHAP mismatch! Dims do not align.")

        return sv

    # ---------------------------------------------------------
    # 4. LIME
    # ---------------------------------------------------------
    elif attr == "lime":
        try:
            from lime.lime_tabular import LimeTabularExplainer
        except ImportError:
            raise ImportError("Please install lime: pip install lime")

        if rng is None:
            rng = np.random.RandomState(0)

        n_samples = getattr(args, 'lime_n_samples', 1000)
        discretize = getattr(args, 'lime_discretize', False)

        explainer = LimeTabularExplainer(
            training_data=X_bg,
            mode="classification",
            feature_names=[str(i) for i in range(X_bg.shape[1])],
            discretize_continuous=discretize,
            random_state=rng.randint(0, 2**31 - 1)
        )

        def _predict_fn(x):
            if hasattr(model, "predict_proba"):
                return model.predict_proba(x)
            else:
                # DNN wrapper
                return model.predict_proba(x)

        sv = np.zeros((X_target.shape[0], X_target.shape[1]), dtype=np.float64)
        for i in range(X_target.shape[0]):
            exp = explainer.explain_instance(
                X_target[i],
                _predict_fn,
                num_features=X_target.shape[1],
                num_samples=n_samples,
                labels=(1,)
            )
            # exp.as_map() returns {label: [(feature_idx, weight), ...]}
            feat_map = dict(exp.as_map()[1])
            for f_idx, weight in feat_map.items():
                sv[i, f_idx] = weight

        return sv

    else:
        raise ValueError(f"Unknown attribution method: {attr}")

def run_chunk(args):
    seed_everything(args.base_seed + args.chunk_id)
    X, y, feature_names = load_dataset(args.dataset, args.drop_features, data_cache_dir=getattr(args, 'data_cache_dir', None),
                                        data_path=getattr(args, 'data_path', None), acs_states=getattr(args, 'acs_states', None),
                                        acs_year=getattr(args, 'acs_year', None),
                                        tcga_gz_path=getattr(args, 'tcga_gz_path', None),
                                        tcga_top_n=getattr(args, 'tcga_top_n', 1000),
                                        tcga_subtype_path=getattr(args, 'tcga_subtype_path', None))
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=args.split_seed, stratify=y)
    b_data = get_boundary_set(X, y, args, args.split_seed)
    b_indices = b_data["boundary_indices"]
    X_boundary = X_test[b_indices] if args.boundary_source == "test" else X_train[b_indices]

    run_ids, test_accs, expvecs, probs_test_list, run_C_values, run_l1_values = [], [], [], [], [], []
    start_run, end_run = args.chunk_id * args.chunk_size, args.chunk_id * args.chunk_size + args.chunk_size

    # C grid logic
    if args.model == "lr" and getattr(args, 'lr_C_mode', 'fixed') == "grid":
        C_grid = np.logspace(np.log10(args.lr_C_min), np.log10(args.lr_C_max), args.n_runs)
    else: C_grid = None

    # L1 ratio grid logic
    if args.model == "lr" and getattr(args, 'lr_l1_ratio_mode', 'fixed') == "grid":
        l1_grid = np.linspace(0, 1, args.n_runs)
    else: l1_grid = None

    # Multi-lens support: parse comma-separated attributions
    attribution_list = [a.strip() for a in args.attribution.split(",")]
    multi_lens = len(attribution_list) > 1
    print(f"[Chunk {args.chunk_id}] Starting runs {start_run} to {end_run-1} | Model: {args.model} | Attr: {attribution_list}")

    # Per-lens expvec accumulators
    expvecs_per_lens = {a: [] for a in attribution_list}

    for r in range(start_run, end_run):
        run_seed = args.base_seed + (args.split_seed * 10000) + r

        C_override = C_grid[r] if C_grid is not None else None
        l1_override = l1_grid[r] if l1_grid is not None else None

        model = make_model(args, X.shape[1], run_seed, C_override=C_override, l1_override=l1_override)

        X_fit, y_fit = X_train, y_train
        if args.train_resample == "bootstrap":
            rng_local = np.random.RandomState(run_seed)
            idx = rng_local.randint(0, X_train.shape[0], size=X_train.shape[0])
            X_fit, y_fit = X_train[idx], y_train[idx]

        model.fit(X_fit, y_fit)
        te_acc = model.score(X_test, y_test)
        probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(X_test))

        # --- ATTRIBUTION CALLS (one per lens, same model instance) ---
        for attr in attribution_list:
            if args.model == "dnn" and attr == "shap" and args.shap_bg_mode == "per_split":
                bg_rng_seed = args.shap_bg_seed + args.split_seed * 1000003
            else:
                bg_rng_seed = run_seed
            sv_local = compute_attributions(model, X_train, X_boundary, args,
                                            rng=np.random.RandomState(bg_rng_seed),
                                            attribution=attr)
            if args.explain_vector_agg == "median":
                vec = np.median(sv_local, axis=0)
            else:
                vec = np.mean(sv_local, axis=0)
            expvecs_per_lens[attr].append(vec)

        run_ids.append(r)
        test_accs.append(te_acc)
        expvecs.append(expvecs_per_lens[attribution_list[0]][-1])  # backward compat: first lens
        probs_test_list.append(probs)
        run_C_values.append(C_override if C_override is not None else args.lr_C)
        run_l1_values.append(l1_override if l1_override is not None else args.lr_l1_ratio)

    out_file = Path(args.output_dir) / f"split{args.split_seed}" / f"chunk_{args.chunk_id}.npz"
    save_dict = dict(dataset=args.dataset, split_seed=args.split_seed, chunk_id=args.chunk_id,
                     run_ids=np.array(run_ids), test_acc=np.array(test_accs),
                     expvec=np.array(expvecs),  # backward compat: first lens
                     probs_test=np.array(probs_test_list, dtype=np.float32),
                     feature_names=feature_names,
                     run_C_values=np.array(run_C_values, dtype=np.float64),
                     run_l1_values=np.array(run_l1_values, dtype=np.float64),
                     attribution_list=np.array(attribution_list))
    # Save per-lens expvecs
    for attr in attribution_list:
        save_dict[f"expvec_{attr}"] = np.array(expvecs_per_lens[attr])
    np.savez(out_file, **save_dict)
    print(f"[Chunk {args.chunk_id}] Saved {len(run_ids)} runs | lenses: {attribution_list}")

# -----------------------------------------------------------------------------
# CLUSTERING LOGIC (Shared)
# -----------------------------------------------------------------------------

def pick_best_k_kmeans(X, k_max=8, seed=0, silhouette_threshold=0.15, cosine_collapse_threshold=0.99, silhouette_metric="euclidean"):
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score
    
    n = X.shape[0]
    if n < k_max + 1: return 1, np.zeros(n, dtype=int), {"reason": "insufficient_samples"}
    if np.var(X) < 1e-9: return 1, np.zeros(n, dtype=int), {"reason": "zero_variance"}

    best_k, best_score, best_labels, scores = 1, -1.0, np.zeros(n, dtype=int), {}

    for k in range(2, k_max + 1):
        km = KMeans(n_clusters=k, random_state=seed, n_init="auto")
        labels = km.fit_predict(X)
        if len(np.unique(labels)) < 2 or np.min(np.bincount(labels)) < 2: continue

        centroids = km.cluster_centers_
        collapse = False
        for i in range(k):
            for j in range(i + 1, k):
                ci_norm, cj_norm = np.linalg.norm(centroids[i]), np.linalg.norm(centroids[j])
                if ci_norm < 1e-12 or cj_norm < 1e-12: continue
                if np.dot(centroids[i], centroids[j]) / (ci_norm * cj_norm) > cosine_collapse_threshold:
                    collapse = True
                    break
            if collapse: break
        if collapse: continue
            
        score = silhouette_score(X, labels, metric=silhouette_metric)
        scores[k] = float(score)
        if score > best_score: best_score, best_k, best_labels = score, k, labels

    if best_k > 1 and best_score < silhouette_threshold:
        return 1, np.zeros(n, dtype=int), {"reason": f"score_{best_score:.3f}_below_{silhouette_threshold}"}
    
    return best_k, best_labels, {"scores": scores, "best_score": best_score}

# -----------------------------------------------------------------------------
# AGGREGATION & REPORTING
# -----------------------------------------------------------------------------

def aggregate_split(args):
    split_dir = Path(args.output_dir) / f"split{args.split_seed}"
    chunk_files = sorted(split_dir.glob("chunk_*.npz"))
    if not chunk_files: raise FileNotFoundError(f"No chunks found in {split_dir}")

    all_expvec, all_expvec_raw, all_acc, all_run_ids, all_probs, all_C_values, all_l1_values, feature_names = [], [], [], [], [], [], [], None
    detected_lenses = None
    expvecs_per_lens = {}

    for cf in chunk_files:
        try: data = np.load(cf, allow_pickle=True)
        except: continue
        if 'probs_test' not in data: continue

        # Detect lenses from saved attribution_list or fallback
        if detected_lenses is None:
            if 'attribution_list' in data:
                detected_lenses = list(data['attribution_list'])
            else:
                detected_lenses = [args.attribution.split(",")[0].strip()]
            for a in detected_lenses:
                expvecs_per_lens[a] = []

        # Load per-lens expvecs if present, else fall back to expvec
        for a in detected_lenses:
            key = f"expvec_{a}"
            if key in data:
                expvecs_per_lens[a].append(data[key])
            else:
                expvecs_per_lens[a].append(data['expvec'])

        all_expvec.append(data['expvec'])
        all_expvec_raw.append(data['expvec_raw'] if 'expvec_raw' in data else data['expvec'])
        all_acc.append(data['test_acc'])
        all_run_ids.append(data['run_ids'])
        all_probs.append(data['probs_test'])
        if 'run_C_values' in data: all_C_values.append(data['run_C_values'])
        if 'run_l1_values' in data: all_l1_values.append(data['run_l1_values'])
        if feature_names is None: feature_names = data['feature_names']

    if not all_expvec: raise ValueError(f"No valid chunks found in {split_dir}")

    full_data = {
        "run_ids": np.concatenate(all_run_ids, axis=0),
        "test_acc": np.concatenate(all_acc, axis=0),
        "expvec": np.concatenate(all_expvec, axis=0),
        "probs_test": np.concatenate(all_probs, axis=0),
        "run_C_values": np.concatenate(all_C_values, axis=0) if all_C_values else None,
        "run_l1_values": np.concatenate(all_l1_values, axis=0) if all_l1_values else None
    }

    def _cluster_lens(X_exp):
        X_raw = X_exp.copy()
        X_proc = X_exp - np.mean(X_exp, axis=0) if args.center else X_exp
        if args.normalize == "l2":
            norms = np.linalg.norm(X_proc, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            X_normed = X_proc / norms
        else:
            X_normed = X_proc
        best_k, best_labels, metrics = pick_best_k_kmeans(
            X_normed, seed=args.boundary_seed, silhouette_threshold=0.15,
            cosine_collapse_threshold=args.cosine_collapse,
            silhouette_metric=args.silhouette_metric)
        centroids_normed = []
        for i in range(best_k):
            mask = (best_labels == i)
            centroids_normed.append(np.mean(X_normed[mask], axis=0) if np.sum(mask) > 0 else np.zeros(X_normed.shape[1]))
        return X_raw, X_normed, best_k, best_labels, np.array(centroids_normed)

    out_file = split_dir / f"aggregate_split{args.split_seed}.npz"
    save_dict = {
        "run_ids": full_data["run_ids"],
        "test_acc": full_data["test_acc"],
        "feature_names": feature_names,
        "probs_test": full_data["probs_test"] if full_data["probs_test"] is not None else np.array([]),
    }
    if full_data["run_C_values"] is not None: save_dict["run_C_values"] = full_data["run_C_values"]
    if full_data["run_l1_values"] is not None: save_dict["run_l1_values"] = full_data["run_l1_values"]
    if detected_lenses: save_dict["attribution_list"] = np.array(detected_lenses)

    # Cluster each lens separately
    primary_done = False
    for a in detected_lenses:
        X_lens = np.concatenate(expvecs_per_lens[a], axis=0)
        X_raw, X_normed, best_k, best_labels, centroids_normed = _cluster_lens(X_lens)
        save_dict[f"expvec_raw_{a}"] = X_raw
        save_dict[f"expvec_normed_{a}"] = X_normed
        save_dict[f"k_star_{a}"] = best_k
        save_dict[f"cluster_labels_{a}"] = best_labels
        save_dict[f"centroids_normed_{a}"] = centroids_normed
        # Backward compat: first lens populates the legacy keys
        if not primary_done:
            save_dict["expvec_raw"] = X_raw
            save_dict["expvec_normed"] = X_normed
            save_dict["k_star"] = best_k
            save_dict["cluster_labels"] = best_labels
            save_dict["centroids_normed"] = centroids_normed
            primary_done = True

    np.savez(out_file, **save_dict)
    print(f"[Aggregate] Saved to {out_file} | lenses: {detected_lenses}")

def generate_report(args):
    agg_file = Path(args.output_dir) / f"split{args.split_seed}" / f"aggregate_split{args.split_seed}.npz"
    if not agg_file.exists(): raise FileNotFoundError(f"Aggregate file not found: {agg_file}")

    data = np.load(agg_file, allow_pickle=True)
    features = data["feature_names"]
    accs     = data["test_acc"]

    # Detect lenses: prefer attribution_list saved in the npz, then --attribution, then legacy fallback
    if "attribution_list" in data:
        lenses = list(data["attribution_list"])
    elif args.attribution:
        lenses = [a.strip() for a in args.attribution.split(",")]
    else:
        lenses = ["default"]

    report_lines = [
        f"EVOXPLAIN REPORT - Split {args.split_seed}",
        "=" * 40,
        f"Global Accuracy : {np.mean(accs):.4f} (+/- {np.std(accs):.4f})",
        f"Lenses reported : {', '.join(lenses)}",
        "=" * 40,
    ]

    for lens in lenses:
        # Per-lens keys preferred; fall back to legacy keys for single-lens runs
        k_key  = f"k_star_{lens}"           if f"k_star_{lens}"           in data else "k_star"
        c_key  = f"centroids_normed_{lens}"  if f"centroids_normed_{lens}"  in data else "centroids_normed"
        lb_key = f"cluster_labels_{lens}"    if f"cluster_labels_{lens}"    in data else "cluster_labels"

        k         = int(data[k_key])
        centroids = data[c_key]
        labels    = data[lb_key]

        report_lines.append(f"\n{'=' * 40}")
        report_lines.append(f"LENS : {lens.upper()}  |  Optimal Basins (k*) : {k}")
        report_lines.append(f"{'=' * 40}")
        report_lines.append(f"  Global Accuracy : {np.mean(accs):.4f} (+/- {np.std(accs):.4f})")
        for i in range(k):
            basin_accs = accs[labels == i]
            report_lines.append(f"  > Basin {i} Accuracy : {np.mean(basin_accs):.4f}  (n={len(basin_accs)})")
        report_lines.append("-" * 40)
        for i, centroid in enumerate(centroids):
            report_lines.append(f"\n--- BASIN {i}  [{lens.upper()}] ---")
            for idx in np.argsort(np.abs(centroid))[::-1][:10]:
                direction = "(>)" if centroid[idx] > 0 else "(<)"
                report_lines.append(f"  {features[idx]:<25} : {centroid[idx]:+.4f}  {direction}")

    report_path = Path(args.output_dir) / f"split{args.split_seed}" / f"report_split{args.split_seed}.txt"
    with open(report_path, "w") as f: f.write("\n".join(report_lines))
    print("\n" + "\n".join(report_lines))

def analyze_subbasins(args):
    agg_file = Path(args.output_dir) / f"split{args.split_seed}" / f"aggregate_split{args.split_seed}.npz"
    data = np.load(agg_file, allow_pickle=True)
    X_raw_all, labels_all = data["expvec_raw"], data["cluster_labels"]
    
    for basin_id in ([args.target_basin] if args.target_basin is not None else np.unique(labels_all)):
        X_basin = X_raw_all[(labels_all == basin_id)]
        X_proc = X_basin - np.mean(X_basin, axis=0) if args.center else X_basin
        if args.normalize == "l2":
            norms = np.linalg.norm(X_proc, axis=1, keepdims=True)
            norms[norms==0] = 1.0
            X_normed = X_proc / norms
        else: X_normed = X_proc
            
        sub_k, _, sub_metrics = pick_best_k_kmeans(X_normed, seed=args.boundary_seed, silhouette_threshold=0.15, cosine_collapse_threshold=args.cosine_collapse, silhouette_metric=args.silhouette_metric)
        print(f"Basin {basin_id}: k={sub_k} (Silhouette: {sub_metrics.get('best_score', 0):.3f})")

def cross_split_stability(args):
    from scipy.optimize import linear_sum_assignment
    seeds = [int(s.strip()) for s in args.split_seeds.split(",")]
    # Which lens to use: --attribution_lens overrides, else first in --attribution
    lens = getattr(args, 'attribution_lens', None) or args.attribution.split(",")[0].strip()
    print(f"[CrossSplit] Using lens: {lens}")
    split_data = {}
    for seed in seeds:
        agg_file = Path(args.output_dir) / f"split{seed}" / f"aggregate_split{seed}.npz"
        data = np.load(agg_file, allow_pickle=True)
        # Use per-lens keys if available, else fall back to legacy keys
        k_key = f"k_star_{lens}" if f"k_star_{lens}" in data else "k_star"
        c_key = f"centroids_normed_{lens}" if f"centroids_normed_{lens}" in data else "centroids_normed"
        split_data[seed] = {
            "k_star": int(data[k_key]),
            "centroids_normed": data[c_key],
            "feature_names": list(data["feature_names"])
        }

    ref_names = split_data[seeds[0]]["feature_names"]
    for seed in seeds[1:]:
        if split_data[seed]["feature_names"] != ref_names: raise ValueError("Feature name mismatch")

    k_per_split = {str(s): split_data[s]["k_star"] for s in seeds}
    all_k1 = all(v == 1 for v in k_per_split.values())
    pairwise_cosine = {}

    if all_k1:
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                si, sj = seeds[i], seeds[j]
                ci, cj = split_data[si]["centroids_normed"][0], split_data[sj]["centroids_normed"][0]
                pairwise_cosine[f"{si}-{sj}"] = round(float(1.0 - cosine(ci, cj)), 6)

        cos_vals = list(pairwise_cosine.values())
        result = {"split_seeds": seeds, "k_star_per_split": k_per_split, "pairwise_cosine": pairwise_cosine, "min_cosine": round(min(cos_vals), 6), "mean_cosine": round(float(np.mean(cos_vals)), 6)}
    else:
        matched_results = {}
        for i in range(len(seeds)):
            for j in range(i + 1, len(seeds)):
                si, sj = seeds[i], seeds[j]
                C_i, C_j = split_data[si]["centroids_normed"], split_data[sj]["centroids_normed"]
                ki, kj = len(C_i), len(C_j)
                sim_matrix = np.zeros((ki, kj))
                for ri in range(ki):
                    for rj in range(kj): sim_matrix[ri, rj] = float(1.0 - cosine(C_i[ri], C_j[rj]))
                row_ind, col_ind = linear_sum_assignment(-sim_matrix)
                matches = [{"basin_split_a": int(ri), "basin_split_b": int(rj), "cosine": round(float(sim_matrix[ri, rj]), 6)} for ri, rj in zip(row_ind, col_ind)]
                avg_matched = round(float(np.mean([m["cosine"] for m in matches])), 6)
                pair_key = f"{si}-{sj}"
                matched_results[pair_key] = {"matches": matches, "avg_matched_cosine": avg_matched}
                pairwise_cosine[pair_key] = avg_matched

        cos_vals = list(pairwise_cosine.values())
        result = {"split_seeds": seeds, "k_star_per_split": k_per_split, "matched_pairwise": matched_results, "pairwise_avg_matched_cosine": pairwise_cosine, "min_cosine": round(min(cos_vals), 6), "mean_cosine": round(float(np.mean(cos_vals)), 6)}

    out_path = Path(args.output_dir) / "cross_split_stability.json"
    save_json(result, out_path)
    print(f"Saved: {out_path}")

# -----------------------------------------------------------------------------
# MAIN CLI
# -----------------------------------------------------------------------------

def build_arg_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="adult")
    p.add_argument("--model", type=str, default="dnn", choices=["rf", "dnn", "xgb", "svm", "lr"])
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--mode", type=str, required=True, choices=["chunk", "aggregate_split", "report", "analyze_subbasins", "cross_split_stability", "load_only"])
    
    # --- ATTRIBUTION ---
    p.add_argument("--attribution", type=str, default="shap",
                   choices=["shap", "ig", "gini", "lime",
                            "shap,ig", "shap,gini", "shap,lime",
                            "ig,lime", "shap,ig,lime"],
                   help="Attribution method(s). Comma-separated for multi-lens mode, e.g. 'shap,lime'.")
    p.add_argument("--lime_n_samples", type=int, default=1000,
                   help="Number of perturbation samples per instance for LIME (default 1000).")
    p.add_argument("--lime_discretize", type=int, default=0,
                   help="Whether LIME discretizes continuous features (0=False, 1=True). Default 0.")
    p.add_argument("--attribution_lens", type=str, default=None,
                   help="Which lens to use in cross_split_stability when multi-lens chunks exist. Defaults to first in --attribution.")
    
    p.add_argument("--split_seed", type=int, default=101)
    p.add_argument("--split_seeds", type=str, default=None)
    p.add_argument("--base_seed", type=int, default=42)
    p.add_argument("--train_resample", type=str, default="none", choices=["none", "bootstrap"])
    p.add_argument("--chunk_id", type=int, default=0)
    p.add_argument("--chunk_size", type=int, default=50)
    p.add_argument("--n_runs", type=int, default=200)
    p.add_argument("--drop_features", type=str, default=None)
    p.add_argument("--data_cache_dir", type=str, default=None)
    p.add_argument("--data_path", type=str, default=None)
    # --- TCGA-specific args ---
    p.add_argument("--tcga_gz_path", type=str, default=None,
                   help="Path to tcga_RSEM_gene_tpm.gz (required when --dataset tcga)")
    p.add_argument("--tcga_top_n", type=int, default=1000,
                   help="Number of top variable genes to retain from TCGA expression matrix.")
    p.add_argument("--tcga_subtype_path", type=str, default=None,
                   help="Path to PAM50 subtype annotation file (TSV or CSV) required for "
                        "--dataset tcga_brca_luminal_vs_basal. Must contain a sample ID column "
                        "and a PAM50/subtype column. Luminal A+B → class 0, Basal-like → class 1; "
                        "HER2-enriched, Normal-like, and unknown subtypes are excluded.")
    p.add_argument("--acs_states", type=str, default="CA")
    p.add_argument("--acs_year", type=str, default="2018")
    p.add_argument("--boundary_source", type=str, default="test")
    p.add_argument("--boundary_method", type=str, default="prob_band")
    p.add_argument("--boundary_prob_low", type=float, default=0.45)
    p.add_argument("--boundary_prob_high", type=float, default=0.55)
    p.add_argument("--boundary_k", type=int, default=200)
    p.add_argument("--boundary_seed", type=int, default=123)
    p.add_argument("--boundary_cache", type=int, default=1)
    p.add_argument("--dnn_layers", type=str, default="100,50")
    p.add_argument("--dnn_lr", type=float, default=0.001)
    p.add_argument("--dnn_epochs", type=int, default=50)
    p.add_argument("--dnn_batch_size", type=int, default=64)
    p.add_argument("--dnn_dropout", type=float, default=0.2)
    p.add_argument("--rf_n_estimators", type=int, default=100)
    p.add_argument("--rf_max_depth", type=int, default=None)
    p.add_argument("--rf_max_features", type=str, default="sqrt")
    
    # --- LR & Elastic Net Args ---
    p.add_argument("--lr_C", type=float, default=1.0)
    p.add_argument("--lr_C_mode", type=str, default="fixed", choices=["fixed", "grid"])
    p.add_argument("--lr_C_min", type=float, default=0.01)
    p.add_argument("--lr_C_max", type=float, default=100.0)
    p.add_argument("--lr_penalty", type=str, default="l2", choices=["l1", "l2", "elasticnet", "none"])
    p.add_argument("--lr_l1_ratio", type=float, default=0.5, help="ElasticNet mixing parameter (0=L2, 1=L1). Only used if penalty=elasticnet.")
    p.add_argument("--lr_l1_ratio_mode", type=str, default="fixed", choices=["fixed", "grid"], help="Grid search l1_ratio from 0 to 1 across runs.")
    p.add_argument("--lr_max_iter", type=int, default=1000)

    p.add_argument("--shap_type", type=str, default="tree")
    p.add_argument("--shap_signed", type=int, default=1)
    p.add_argument("--shap_background_size", type=int, default=100)
    p.add_argument("--explain_vector_agg", type=str, default="mean", choices=["mean", "median"])
    p.add_argument("--normalize", type=str, default="l2")
    p.add_argument("--center", type=int, default=0)
    p.add_argument("--target_basin", type=int, default=None)
    p.add_argument("--cosine_collapse", type=float, default=0.99)
    # --- SHAP background determinism (DNN only) ---
    p.add_argument("--shap_bg_mode", type=str, default="per_split",
                   choices=["per_run", "per_split"],
                   help="How SHAP background RNG is seeded for DNN DeepExplainer. "
                        "'per_split' fixes bg across runs within a split (recommended). "
                        "'per_run' uses run_seed (legacy behaviour).")
    p.add_argument("--shap_bg_seed", type=int, default=12345,
                   help="Base seed for SHAP background sampling when shap_bg_mode='per_split'.")
    # --- Silhouette metric control ---
    p.add_argument("--silhouette_metric", type=str, default="euclidean",
                   choices=["euclidean", "cosine"],
                   help="Distance metric for silhouette score in clustering. Default 'euclidean'.")
    return p

def main():
    log_environment()
    args = build_arg_parser().parse_args()
    if args.mode == "chunk": run_chunk(args)
    elif args.mode == "aggregate_split": aggregate_split(args)
    elif args.mode == "report": generate_report(args)
    elif args.mode == "analyze_subbasins": analyze_subbasins(args)
    elif args.mode == "cross_split_stability": cross_split_stability(args)
    elif args.mode == "load_only":
        # Dry-run: load and validate the dataset, then exit. No training.
        print("[load_only] Dry-run mode: loading dataset and exiting.")
        X, y, feature_names = load_dataset(
            args.dataset, args.drop_features,
            data_cache_dir=getattr(args, 'data_cache_dir', None),
            data_path=getattr(args, 'data_path', None),
            acs_states=getattr(args, 'acs_states', None),
            acs_year=getattr(args, 'acs_year', None),
            tcga_gz_path=getattr(args, 'tcga_gz_path', None),
            tcga_top_n=getattr(args, 'tcga_top_n', 1000),
            tcga_subtype_path=getattr(args, 'tcga_subtype_path', None),
        )
        print(f"[load_only] X shape       : {X.shape}")
        print(f"[load_only] y shape       : {y.shape}")
        print(f"[load_only] n_features    : {len(feature_names)}")
        print(f"[load_only] feature_names : {feature_names[:10]}")
        unique, counts = np.unique(y, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"[load_only] class {u}: {c:,} samples")
        print("[load_only] NaN in X:", np.isnan(X).any())
        print("[load_only] Inf in X:", np.isinf(X).any())
        print("[load_only] Done.")
    else: raise ValueError(f"Unknown mode {args.mode}")

if __name__ == "__main__":
    main()
