"""
Boom Challenge — Trajectory Unknown
========================================
Author  : Goutam Roy
Profile : https://www.freelancer.com/u/Goutam895
Type    : Individual (Solo)

Model   : Ensemble — XGBoost (55%) + Random Forest (45%)
Features: 8 base + 11 physics-derived = 19 total
"""

import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

# ══════════════════════════════════════════════════════════
# CONFIGURATION
# ══════════════════════════════════════════════════════════

BASE = 'Boom-Challenge-Datasets-main'

FEATURES_BASE = [
    'energy', 'angle_rad', 'coupling', 'strength',
    'porosity', 'gravity', 'atmosphere', 'shape_factor'
]

TARGETS = [
    'P80', 'fines_frac', 'oversize_frac',
    'R95', 'R50_fines', 'R50_oversize'
]

XGB_WEIGHT = 0.55
RF_WEIGHT  = 0.45

# ══════════════════════════════════════════════════════════
# PHYSICS FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════

def add_physics_features(df):
    """
    11 physics-informed interaction features derived from
    impact mechanics principles.

    Range-driving:
      energy_per_gravity   = energy / gravity
      dispersal_index      = energy / (gravity * strength)
      coupling_efficiency  = energy * coupling / gravity
      atmosphere_damping   = atmosphere * energy

    Size-driving:
      energy_x_coupling    = energy * coupling
      energy_x_angle       = energy * angle_rad
      energy_squared       = energy ** 2

    Material properties:
      gravity_x_strength   = gravity * strength
      porosity_x_strength  = porosity * strength
      material_resistance  = strength * (1 - porosity)
      shape_x_coupling     = shape_factor * coupling
    """
    df = df.copy()
    df['energy_per_gravity']  = df['energy'] / df['gravity']
    df['dispersal_index']     = df['energy'] / (df['gravity'] * df['strength'])
    df['coupling_efficiency'] = df['energy'] * df['coupling'] / df['gravity']
    df['atmosphere_damping']  = df['atmosphere'] * df['energy']
    df['energy_x_coupling']   = df['energy'] * df['coupling']
    df['energy_x_angle']      = df['energy'] * df['angle_rad']
    df['energy_squared']      = df['energy'] ** 2
    df['gravity_x_strength']  = df['gravity'] * df['strength']
    df['porosity_x_strength'] = df['porosity'] * df['strength']
    df['material_resistance'] = df['strength'] * (1.0 - df['porosity'])
    df['shape_x_coupling']    = df['shape_factor'] * df['coupling']
    return df


# ══════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════

print("=" * 55)
print("Boom Challenge — Trajectory Unknown")
print("Author: Goutam Roy | @Goutam895")
print("=" * 55)

print("\n[1/5] Loading data...")
train_raw = pd.read_csv(f'{BASE}/forward_prediction/train.csv')
labels    = pd.read_csv(f'{BASE}/forward_prediction/train_labels.csv')
test_raw  = pd.read_csv(f'{BASE}/forward_prediction/test.csv')

train = add_physics_features(train_raw)
test  = add_physics_features(test_raw)

FEATURES = FEATURES_BASE + [
    c for c in train.columns if c not in FEATURES_BASE
]

X      = train[FEATURES].values
y      = labels[TARGETS].values
X_test = test[FEATURES].values

with open(f'{BASE}/inverse_design/constraints.json') as f:
    cfg = json.load(f)
bounds_cfg = cfg['input_bounds']

print(f"  Train : {X.shape}")
print(f"  Test  : {X_test.shape}")
print(f"  Base features   : {len(FEATURES_BASE)}")
print(f"  Physics features: {len(FEATURES) - len(FEATURES_BASE)}")
print(f"  Total features  : {len(FEATURES)}")


# ══════════════════════════════════════════════════════════
# 2. VALIDATION
# ══════════════════════════════════════════════════════════

print("\n[2/5] Validating on 80/20 split...")

X_tr, X_val, y_tr, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

val_xgb = MultiOutputRegressor(XGBRegressor(
    n_estimators=300, max_depth=6, learning_rate=0.08,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=3, gamma=0.1, reg_alpha=0.1,
    random_state=42, verbosity=0))

val_rf = MultiOutputRegressor(RandomForestRegressor(
    n_estimators=200, max_depth=12, min_samples_leaf=2,
    max_features=0.7, random_state=42, n_jobs=-1))

val_xgb.fit(X_tr, y_tr)
val_rf.fit(X_tr, y_tr)

val_pred = (XGB_WEIGHT * val_xgb.predict(X_val) +
            RF_WEIGHT  * val_rf.predict(X_val))
r2_val   = r2_score(y_val, val_pred, multioutput='raw_values')

print(f"  Ensemble = XGBoost x{XGB_WEIGHT} + RandomForest x{RF_WEIGHT}")
print("  Validation R2 (unseen 20% data):")
for t, s in zip(TARGETS, r2_val):
    print(f"    {t:<18}: {s:.4f}")
print(f"  Average R2: {r2_val.mean():.4f}")


# ══════════════════════════════════════════════════════════
# 3. FINAL MODELS — trained once on full data
# ══════════════════════════════════════════════════════════

print("\n[3/5] Training final models on full dataset...")

xgb_model = MultiOutputRegressor(XGBRegressor(
    n_estimators=300, max_depth=6, learning_rate=0.08,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=3, gamma=0.1, reg_alpha=0.1,
    random_state=42, verbosity=0))

rf_model = MultiOutputRegressor(RandomForestRegressor(
    n_estimators=200, max_depth=12, min_samples_leaf=2,
    max_features=0.7, random_state=42, n_jobs=-1))

xgb_model.fit(X, y)
rf_model.fit(X, y)

with open('model.pkl', 'wb') as f:
    pickle.dump({
        'xgb': xgb_model,
        'rf': rf_model,
        'xgb_weight': XGB_WEIGHT,
        'rf_weight': RF_WEIGHT,
        'features': FEATURES
    }, f)
print("  model.pkl saved!")


def ensemble_predict(X_input):
    return (XGB_WEIGHT * xgb_model.predict(X_input) +
            RF_WEIGHT  * rf_model.predict(X_input))


# ══════════════════════════════════════════════════════════
# 4. FORWARD PREDICTION
# ══════════════════════════════════════════════════════════

print("\n[4/5] Generating forward predictions...")

pred_test = ensemble_predict(X_test)
sub = pd.DataFrame(pred_test, columns=TARGETS)
sub['fines_frac']    = sub['fines_frac'].clip(lower=0)
sub['oversize_frac'] = sub['oversize_frac'].clip(lower=0)
sub.insert(0, 'scenario_id', range(len(sub)))
sub.to_csv('prediction_submission.csv', index=False)
print(f"  Saved prediction_submission.csv ({len(sub)} rows)")
print(f"  Negative fractions: 0 (clipped) ✓")


# ══════════════════════════════════════════════════════════
# 5. INVERSE DESIGN
# ══════════════════════════════════════════════════════════

print("\n[5/5] Inverse design...")

# A: filter training rows by output constraints
preds_all = ensemble_predict(train[FEATURES].values)
mask = (
    (preds_all[:, 0] >= 96) & (preds_all[:, 0] <= 101) &
    (preds_all[:, 3] <= 175)
)
valid_inputs = train_raw[FEATURES_BASE][mask].copy()
print(f"  Candidate rows: {mask.sum()}")

# B: clip base features to declared input bounds
for col in FEATURES_BASE:
    valid_inputs[col] = valid_inputs[col].clip(
        lower=bounds_cfg[col]['min'],
        upper=bounds_cfg[col]['max']
    )

# C: re-derive physics features after clipping
valid_inputs = add_physics_features(valid_inputs)

# D: sort by energy (lower = better small-impact score)
valid_inputs = valid_inputs.sort_values('energy').reset_index(drop=True)

# E: re-verify with ensemble after clipping
preds_check = ensemble_predict(valid_inputs[FEATURES].values)
valid_mask  = (
    (preds_check[:, 0] >= 96) & (preds_check[:, 0] <= 101) &
    (preds_check[:, 3] <= 175)
)
design_df = valid_inputs[valid_mask].head(20).reset_index(drop=True)

# F: final strict re-verify
fp = ensemble_predict(design_df[FEATURES].values)
truly_valid = (fp[:, 0] >= 96) & (fp[:, 0] <= 101) & (fp[:, 3] <= 175)
design_df   = design_df[truly_valid].reset_index(drop=True)

out_df = design_df[FEATURES_BASE].copy()
out_df.insert(0, 'submission_id', range(len(out_df)))
out_df.to_csv('design_submission.csv', index=False)

fp2 = ensemble_predict(design_df[FEATURES].values)
vc  = sum(1 for p, r in zip(fp2[:, 0], fp2[:, 3])
          if 96 <= p <= 101 and r <= 175)
bounds_ok = all(
    (out_df[col] >= bounds_cfg[col]['min']).all() and
    (out_df[col] <= bounds_cfg[col]['max']).all()
    for col in FEATURES_BASE
)

print(f"  P80  : [{fp2[:,0].min():.4f}, {fp2[:,0].max():.4f}]"
      f"  (target: 96-101) ✓")
print(f"  R95  : max {fp2[:,3].max():.4f}  (target: <=175) ✓")
print(f"  Valid: {vc}/{len(out_df)} ✓")
print(f"  Bounds OK: {bounds_ok} ✓")
print(f"  Saved design_submission.csv ({len(out_df)} scenarios)")

print("\n" + "=" * 55)
print("All done!")
print("=" * 55)
