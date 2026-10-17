# DATA CARD — SHIKRA

## 1. NASA C-MAPSS Turbofan Degradation Dataset

| Field | Value |
|---|---|
| **Source URL** | https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip |
| **Download date** | *recorded by `make data`* |
| **Licence** | NASA public dataset. Cite: Saxena et al. (2008). "Damage propagation modeling for aircraft engine run-to-failure simulation." *ICPHM 2008*. |
| **Subsets used today** | FD001 and FD003 (pooled for model training) |
| **FD002 / FD004** | Reserved for future multi-condition extension |

### SHA-256 checksums (filled by `make data`)

| File | SHA-256 |
|---|---|
| train_FD001.txt | *TBD* |
| test_FD001.txt | *TBD* |
| RUL_FD001.txt | *TBD* |
| train_FD002.txt | *TBD* |
| test_FD002.txt | *TBD* |
| RUL_FD002.txt | *TBD* |
| train_FD003.txt | *TBD* |
| test_FD003.txt | *TBD* |
| RUL_FD003.txt | *TBD* |
| train_FD004.txt | *TBD* |
| test_FD004.txt | *TBD* |
| RUL_FD004.txt | *TBD* |

---

## 2. Synthetic Wing Parameters (SHIKRA-Wing v0)

> **All parameters below are SYNTHETIC and for research/demonstration only.**
> They are not derived from or validated against any classified or official IAF data.

### Fleet

| Parameter | Value | Notes |
|---|---|---|
| Squadrons | 3 | Named S1, S2, S3 |
| Aircraft per squadron | 18 | Tail IDs T01–T54 |
| Engines per aircraft | 2 | |
| Sortie demand | 0.8 sorties/MC tail/day | 1 sortie = 1 engine cycle on both engines |

### Maintenance

| Parameter | Value | Notes |
|---|---|---|
| Hangar slots | 3 per squadron | |
| Planned engine change | 2 days | |
| Unscheduled engine change | 5 days | |
| Engine repair (BRD/HAL) | Lognormal, mean 60 days, CV 0.4 | Synthetic |
| Engine spares | 1 per squadron (base) + 3 (depot) | Synthetic |
| Depot→base transfer | 3 days | Synthetic |

### LRU Parameters (all SYNTHETIC)

| LRU ID | Description | Weibull β | Weibull η (sorties) | Source | Cost (₹ lakh) | Repair TAT (days) | Lead time (days) | Base stock | Depot stock |
|---|---|---|---|---|---|---|---|---|---|
| fuel_pump | Fuel pump | 2.0 | 400 | domestic | 8 | 15 | 45 | 2 | 4 |
| hyd_pump | Hydraulic pump | 1.8 | 350 | domestic | 12 | 20 | 50 | 2 | 4 |
| radar_tx | Radar transmitter | 1.5 | 600 | foreign | 85 | 30 | 120 | 1 | 2 |
| ins_unit | Inertial navigation unit | 2.5 | 700 | foreign | 95 | 25 | 150 | 1 | 2 |
| gen_ac | AC generator | 2.0 | 450 | domestic | 18 | 18 | 40 | 2 | 4 |
| ecs_valve | ECS valve | 1.2 | 300 | foreign | 22 | 35 | 180 | 2 | 3 |
| eng_ctrl | Engine controller | 3.0 | 900 | foreign | 110 | 45 | 160 | 1 | 2 |
| fire_detect | Fire detection unit | 1.5 | 500 | domestic | 6 | 12 | 30 | 3 | 5 |
| apu_start | APU starter | 2.2 | 150 | foreign | 30 | 28 | 90 | 2 | 3 |
| brake_ctrl | Brake controller | 1.8 | 250 | domestic | 9 | 10 | 35 | 3 | 5 |

### Pipeline and Simulation

| Parameter | Value |
|---|---|
| pipeline_mult | 1.0 |
| Simulation horizon | 365 days |
| Replications | 30 (seeds 1–30) |

### Calibration change log

*Changes to shared wing parameters made during G0.4 calibration will be logged here.*

---

## 3. Model Parameters

| Parameter | Value |
|---|---|
| Unit split seed | 2026 |
| Train/calib/replay | 60% / 20% / 20% of pooled FD001+FD003 train units |
| RUL cap | 125 cycles |
| LightGBM quantiles | 0.05, 0.10, 0.50, 0.90, 0.95 |
| Seeds per quantile | 5 (seeds 0–4) |
| Conformal coverage | 80% and 90% bands |
| Rolling windows | 5, 10, 20, 30 cycles |
