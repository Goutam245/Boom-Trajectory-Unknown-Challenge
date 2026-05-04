# Boom Challenge — Trajectory Unknown
### Physics-informed Ensemble ML solution for asteroid ejecta prediction

**Submitted by:** Goutam Roy  
**Freelancer Profile:** [@Goutam895](https://www.freelancer.com/u/Goutam895)  
**Submission Type:** Individual (Solo)  
**Challenge:** [Boom Challenge on Freelancer.com](https://www.freelancer.com/boom)

---

## Overview

Solo submission for the **Boom Challenge**, covering both parts:

- **Part 1 — Forward Prediction:** Predict 6 ejecta outcomes from 8 impact parameters
- **Part 2 — Inverse Design:** Propose 20 impact scenarios satisfying output constraints

All code, physics feature engineering, modeling, and submission files were developed independently.

---

## How to Reproduce

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Place dataset
Ensure `Boom-Challenge-Datasets-main/` is in the same directory as `solution.py`.

### Step 3 — Run
```bash
python solution.py
```

### Generated output files

| File | Description |
|---|---|
| `prediction_submission.csv` | Forward prediction — 492 test scenarios |
| `design_submission.csv` | Inverse design — 20 optimized scenarios |
| `model.pkl` | Trained ensemble model (XGBoost + Random Forest) |

---

## Software Requirements

| Package | Version |
|---|---|
| Python | 3.10+ |
| xgboost | 3.2.0 |
| scikit-learn | 1.8.0 |
| pandas | 3.0.1 |
| numpy | 2.4.3 |
| scipy | 1.17.1 |

No GPU required. No special drivers or CUDA needed.

---

## Hardware Requirements

| Spec | Detail |
|---|---|
| CPU | Any modern CPU (Intel / AMD) |
| RAM | 2 GB minimum |
| GPU | Not required |
| Training time | ~30 seconds on standard laptop CPU |
| OS | Windows / Linux / macOS |
| Cloud cost | $0 |

---

## Algorithm

### Model: Weighted Ensemble — XGBoost + Random Forest

Two complementary models are trained and their predictions combined:

| Model | Weight | Strength |
|---|---|---|
| XGBoost | 55% | Non-linear interactions, handles outliers well |
| Random Forest | 45% | Variance reduction, stable on small datasets |

The ensemble reduces individual model bias and improves generalization to out-of-distribution test scenarios.

---

## Trained Model

model.pkl is hosted on Google Drive due to file size (>25MB):  
[Download model.pkl](https://drive.google.com/file/d/1hR8cc72MAOrynoH3XEqYXrA5dDyLuLvA/view?usp=sharing)

---

### Physics Feature Engineering

The model predicts ejecta outcomes from the **target surface only**. The projectile is not modeled as a separate fragmenting object.

**11 physics-derived features** are engineered on top of the 8 base parameters (19 total), grounded in impact mechanics:

#### Range-driving features (govern R95, R50)

| Feature | Formula | Physical Meaning |
|---|---|---|
| `energy_per_gravity` | energy / gravity | Lower gravity → fragments travel farther |
| `dispersal_index` | energy / (gravity × strength) | Unified ejecta range metric |
| `coupling_efficiency` | energy × coupling / gravity | Effective energy per gravitational pull |
| `atmosphere_damping` | atmosphere × energy | Atmospheric drag reduces ejecta range |

#### Size-driving features (govern P80, fines/oversize)

| Feature | Formula | Physical Meaning |
|---|---|---|
| `energy_x_coupling` | energy × coupling | Energy transferred into target surface |
| `energy_x_angle` | energy × angle_rad | Impact momentum directionality |
| `energy_squared` | energy² | Non-linear energy scaling of fragment size |

#### Material property interactions (govern fines_frac, oversize_frac)

| Feature | Formula | Physical Meaning |
|---|---|---|
| `gravity_x_strength` | gravity × strength | Surface resistance to fragmentation |
| `porosity_x_strength` | porosity × strength | Material cohesion — controls fines output |
| `material_resistance` | strength × (1 − porosity) | Solid fraction governs coarse fragments |
| `shape_x_coupling` | shape_factor × coupling | Impactor shape effect on oversize fraction |

---

### Validation R² Scores

Validation and final training use **separate model instances** — no data leakage.

| Target | Validation R² |
|---|---|
| P80 | 0.9771 |
| fines_frac | 0.9385 |
| oversize_frac | 0.9892 |
| R95 | 0.9224 |
| R50_fines | 0.8957 |
| R50_oversize | 0.8524 |
| **Average** | **0.9292** |

All scores measured on the held-out 20% validation set — not seen during training.

---

### Assumptions and Limitations

- Model assumes Mox-95 physics is self-consistent with the training distribution
- Ensemble methods do not extrapolate beyond the training feature range — out-of-distribution scenarios may show higher error
- Physics features are interaction terms derived from domain knowledge; no differential equations are solved
- Stochastic nature of real impacts is not modeled — predictions represent average outcomes

### Extensibility

The pipeline applies to any physics simulation dataset with tabular input/output. The physics feature engineering step can be adapted to domain-specific interaction terms.

Candidate applications: volcanic ejecta modeling, building collapse debris prediction, landslide runout estimation, underwater explosion fragment dispersal.

Only the dataset, feature engineering functions, and constraint file need to change — the ensemble architecture is fully domain-agnostic.

---

## Results

### Forward Prediction
- 492 test scenarios predicted
- Output: `prediction_submission.csv`
- No negative fraction values ✓

### Inverse Design
- 20 valid scenarios, all satisfying:
  - P80 ∈ [96.18, 100.81] ✅ (target: 96–101)
  - R95 ≤ 116.38 ✅ (target: ≤175)
  - All input features within declared bounds ✅
- Scenarios selected by lowest energy for best small-impact score
- Output: `design_submission.csv`

---

## Repository Structure

```
boom-challenge-solution/
│
├── solution.py                    # Complete ML pipeline
├── requirements.txt               # Python dependencies
├── model.pkl                      # See Google Drive link above
│
├── prediction_submission.csv      # Forward prediction (492 rows)
├── design_submission.csv          # Inverse design (20 scenarios)
├── README.md
│
└── Boom-Challenge-Datasets-main/
    │
    ├── forward_prediction/
    │   ├── train.csv              # Training features (2,930 rows)
    │   ├── train_labels.csv       # Training labels (6 targets)
    │   └── test.csv               # Test features (492 rows)
    │
    └── inverse_design/
        ├── constraints.json       # Output constraints + input bounds
        └── design_submission_template.csv
```

---

## Contact

**Goutam Roy**  
Freelancer: [@Goutam895](https://www.freelancer.com/u/Goutam895)
