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
| train_FD001.txt | 963b5e22825b34d8b21c69e1aeb4af3e647050eb672ee8834ba4b5d91d2de0f8 |
| test_FD001.txt | 3cda7109ce17bafb5443f2ac926cfcf88154b941b8c4cf95eb55d1ddd6f52851 |
| RUL_FD001.txt | a19c8ec94931949d0485bdc35118206e9c81c4547b422efb9cf86f4ceddbceca |
| train_FD002.txt | dac6c4dbc4e7c1bdeb5747da3d313d05c395bb99801b44a002b26a2ba13d788f |
| test_FD002.txt | de7b5bf7e998a985c378488480528b7c02cff1406a46740def362dda8d9b4e02 |
| RUL_FD002.txt | c851dd96a6ea6998d3c4a8f834d3c8013aa90e93a6ed950dc826ad0655b2906b |
| train_FD003.txt | 2abbe9968cc5e8eb091980f51b20f62bb4127336d3482cb52071d53bf23329e2 |
| test_FD003.txt | 299babd63c8d987cef079c4a425429f33b3a34797d803bbe2ad48c29dbd0d790 |
| RUL_FD003.txt | df1e0566306b174a2de41c67a3e7a51877889598b78643fc3e5685259091b7cb |
| train_FD004.txt | 27ef6160b6a1dcb2613a88de9c239f763b223f02cdc41dc5cdedc5dc189b6218 |
| test_FD004.txt | 1dc675fff0624bac10786927c6715b37d1297657137400d2b1a3138d777a3ba5 |
| RUL_FD004.txt | 196b836b85a95ac7fdbbf29c5fdf1657382eafa445644d114ffaaf50dc2975e1 |

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
| pipeline_mult | 1.7 |
| Simulation horizon | 365 days |
| Replications | 30 (seeds 1–30) |

### Calibration change log

- **Iteration 1**: Set `pipeline_mult: 1.9` (previously `1.0`). Increases repair turnaround times, depot-to-base transfer times, and LRU procurement lead times by 1.9×, bringing 30-seed reactive serviceability from ~76.1% to 61.6%, satisfying Gate G0.4 (target 55%–65%).
- **Supply halt repair restart**: In A12 supply shock, held-repair return day set to `return_day = max(day, self.shock_end) + repair_tat`, and shock window bounded by `self.shock_start <= day < self.shock_end` (v1 returned all held repairs on the day the halt lifted, which produced an instant recovery and an overshoot; found in verification).
- **Model of record re-freeze**: model of record re-frozen 17 Oct; previous replay model was not committed and was overwritten by a retrain; all results regenerated from the committed model.

### Pre-registered SHIKRA v2 (lead-time-aware pre-positioning)

Pre-positioning horizon h = (depot→base transfer days + planned change days) × pipeline_mult × sortie_demand, in cycles. Applies identically to shikra and oracle. R_star unchanged. `shikra_v2` and `oracle_v2` are the headline policies; `shikra` and `oracle` (v1) represent ablations without lead-time-aware prepositioning.

### Pre-registered v2 simulation

v2 simulation:

- random initial ages (engines start at a uniform random cycle of their trajectory; LRUs start with sorties flown uniform in [0, first lifetime]), drawn from a dedicated RNG stream, identical across policies;
- 365-day warm-up discarded; all KPIs measured over days 365–729;
- squadron symmetry fixed;
- pipeline_mult recalibrated so reactive serviceability lands at 55–65% (shared parameter only, at most 3 iterations, each logged).

Whatever G0.5 shows afterwards is reported as is.

#### Recalibration log (v2 simulation, max 3 tries)
- **Try 1 (`pipeline_mult: 1.9`)**: reactive serviceability = 53.4% (below 55% target; slope = +0.002586).
- **Try 2 (`pipeline_mult: 1.5`)**: reactive serviceability = 60.9% (satisfies G0.4, but slope = -0.006162).
- **Try 3 (`pipeline_mult: 1.7`)**: reactive serviceability = 56.5% (satisfies G0.4 55%–65%; daily MC trajectory slope = -0.000662, 95% CI [-0.001882, +0.000558], slope CI includes 0). Calibrated value set to 1.7.

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

---

## 4. Ingestion & Data Quality (T1.8)

> **All mock exports are SYNTHETIC and do not represent Indian Air Force (IAF) data.**

- **Data Origin**: Four synthetic extracts (`emms_workorders.csv`, `immols_stock.csv`, `hums_engine.csv`, `brd_repairs.csv`) are generated by `scripts/make_mock_exports.py` (seed 2026), derived read-only from `config/wing.yaml` and a single simulation run (`shikra_v2`, seed 1).
- **Sensor Telemetry**: HUMS sensor readings (`s2, s3, s4, s7, s11, s12, s15`) represent synthetic noise and baseline values, not true engine physical degradation.
- **Fault Detection Scope**: Detection covers the three modelled fault types only (missing primary keys, mismatched serial numbers, and stale stock rows older than 30 days). Perfect recall reported in `results/dq_detection.csv` is therefore by construction against this closed fault model.
- **Stale Stock Policy**: Stale IMMOLS stock records (>30 days old) are loaded into `lru_stock` with `stale = 1` instead of being quarantined, preserving the latest known stock state for operations while flagging it for data refresh.

---

## 5. Spares Allocation (T1.6)

> **All parameters and inventory levels are SYNTHETIC and for research/demonstration only.**

### Budget and Valuation
- **Baseline Budget**: **₹2508.0 lakh**, calculated as the total capital of LRU spares in `config/wing.yaml`:
  $$\text{Budget} = \sum_{i \in \text{LRUs}} \text{cost\_lakh}_i \times (3 \times \text{stock\_base}_i + \text{stock\_depot}_i)$$
- **Zero-Sum Reallocation**: VARI-METRIC reallocates this fixed budget without requiring additional capital expenditure.

### Assumptions
- **One Installed Unit per Tail**: Each of the 54 aircraft tails (3 squadrons × 18 aircraft) contains exactly 1 installed unit of each LRU type (fleet size $N_{\text{tails}} = 54$).
- **Symmetric Squadrons**: The 3 squadrons are identical in fleet size (18 tails each), operational tempo (0.8 sorties/day), and maintenance capabilities. Base inventory is allocated symmetrically across all three squadrons in increments of 1 unit per squadron (3 units per step), maintaining equal base stock and conforming to the engine's `stock_base` configuration format without requiring simulation engine modifications.
- **Stationary Poisson Demand**: LRU failures are modelled as Poisson arrivals with fleet rate:
  $$\lambda_i = \frac{N_{\text{tails}} \times \text{sortie\_demand}}{\eta_i \cdot \Gamma\left(1 + \frac{1}{\beta_i}\right)}$$
  where $\beta_i$ and $\eta_i$ are the synthetic Weibull shape and scale parameters from `config/wing.yaml`.
- **Exclusion of Engines**: Engine spare levels (1 per base, 3 at depot) are held fixed and excluded from the optimization because `config/wing.yaml` does not specify engine acquisition or overhaul unit costs.

### Engine-to-METRIC Mapping
The two-echelon VARI-METRIC optimization model maps to the simulation engine mechanics as follows:
- **Repair Routing**: In `shikra/sim/engine.py`, failed LRUs are routed to the central depot for repair ($r_i = 1.0$).
- **Depot Repair Pipeline**: Depot repair turnaround time is $T_i^{\text{depot}} = \text{repair\_tat\_days}_i \times \text{pipeline\_mult}$. Under infinite repair capacity (M/G/$\infty$ queue), the expected number of items in depot repair is $D_i = \lambda_i \times T_i^{\text{depot}}$.
- **Depot Delay & Base Resupply**: The physical transit time from depot to base is $O_i = \text{transfer\_days} \times \text{pipeline\_mult}$ (nominal 3 days $\times$ `pipeline_mult`). When depot stock is depleted, replacement LRUs wait in a first-come-first-served (FCFS) queue. By Little's Law, the expected depot delay is $\text{EBO}_{\text{depot}}(s_{0,i}) / \lambda_i$.
- **Base Pipeline & Expected Backorders (EBO)**: The effective resupply time to each base is $T_{j,i}^{\text{base}} = O_i + \frac{\text{EBO}_{\text{depot}}(s_{0,i})}{\lambda_i}$. The expected base pipeline is $\mu_{j,i} = \frac{\lambda_i}{3} \times T_{j,i}^{\text{base}}$. Expected backorders at base $j$ for LRU $i$ with stock $s_{j,i}$ are calculated via exact Poisson expectation:
  $$\text{EBO}(s, \mu) = \sum_{x > s} (x - s) \frac{\mu^x e^{-\mu}}{x!} = \mu \left(1 - \sum_{k=0}^{s-1} \frac{\mu^k e^{-\mu}}{k!}\right) - s \left(1 - \sum_{k=0}^{s} \frac{\mu^k e^{-\mu}}{k!}\right)$$
- **Expected NMCS Aircraft**: Fleet expected Not Mission Capable due to Supply (NMCS) aircraft is evaluated as $\sum_{i} \sum_{j=1}^3 \text{EBO}_{j,i}$.

