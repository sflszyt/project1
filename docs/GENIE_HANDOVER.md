# Genie handover: current MVP contract and Databricks migration

Updated 2026-10-04. Audience: Databricks Genie and the migration engineer. [MVP_SPEC](MVP_SPEC.md) is the Chinese specification and summarizes prior validation boundaries; [README](../README.md) lists runtime packages and workspace preparation. This handover describes the portable 01–04 migration package. No Databricks deployment or successful remote bundle validation is claimed.

## 1. Pipeline and the upstream boundary

Layer 1 fits one LightGBM propensity model per product. A tuning model selects iterations; a separate champion is refit on its declared scope. Optional calibration uses a reserved block. Final Test supplies reports and a fixed release gate. Layer 2 predicts purchase-configuration attributes; scoring returns eligible weighted Top-k recommendations and reasons.

**Notebook 00 only prepares existing synthetic input locally. It is not a real-data preparation option.** The production job starts at **01**. Upstream feature/label engineering must create the real modeling table, matching dictionary and purchase table before 01; 01 validates them and never repairs stale splits. Do not redirect 00 to real tables or enable its synthetic guard for production.

| Step | Inputs | Outputs / responsibility |
|---|---|---|
| 01 | modeling table, dictionary, `raw.l2_purchase_config` | full preflight; `ml.l1_raw_scores`, `ml.l1_training_runs`; MLflow training artifacts |
| 02 | training runs, champion holdout scores, modeling table, dictionary, purchases | evidence, calibration if enabled, Test report/gate; `ml.l1_evaluation`, `ml.model_selection`; release bundle |
| 03 | selected release, modeling table, purchases | `ml.l2_evaluation`, `ml.layer2_models`, Layer 2 artifacts |
| 04 | modeling table, model selection, Layer 2 models; validated artifacts | four `scores.*` tables and scoring manifest |

## 2. Repository and runtime

Notebooks are Databricks `.py` source format. Keep `notebooks/` and `conf/` together; imports use the notebook directory and `repo_root()` finds `conf/products.yaml`. There is no wheel or `src/` package. `common.py` handles platform/Delta/preflight; `l1_contract.py` handles config/date/artifact contracts; `modeling.py` handles modeling and explanation functions. The Windows-only Spark worker, synthetic preparation notebook 00, local runners, tests, reference originals and development records are excluded from this migration tree. The Databricks branch uses the runtime Spark session without the Windows worker.

Local `.venv/`, `tools/`, `spark-warehouse/`, `mlruns/`, and `artifacts/` are ignored local resources and do not automatically follow a worktree. Spark reads/filters/writes; pandas/LightGBM fit on the driver. The historical full synthetic run used a 64 GB machine; size the real driver from actual data, rather than treating that as a production capacity guarantee.

| Concern | Local | Databricks target |
|---|---|---|
| Profile | local profile (not distributed) | `conf/env/dev.yaml`, MIGRATION placeholders remain |
| Tables | `{catalog}__{schema}.{table}` | `{catalog}.{schema}.{table}`, optional `table_overrides` |
| Parameters | `--name value` or `--name=value`, then `PRODREC_NAME`, then default | widgets populated by job parameters |
| Spark | local session and Delta catalog rehydration | active cluster session; runtime/profile must agree |
| MLflow | local SQLite and artifact paths | Databricks tracking, configured experiment root |
| File bundle | `artifacts/<release_id>/` | configured UC Volume path |
| 00 | synthetic preparation allowed | `allow_synthetic_data: false`, excluded from job |

Target DBR 18 LTS ML/runtime libraries must be verified on the chosen workspace. Historical local versions include Python 3.12, Spark 4.1.1, LightGBM 4.6, MLflow 3.8, sklearn 1.6. Do not infer successful remote compatibility from local execution.

## 3. Real input contract

`ml.modeling_table` has unique, non-null `client_id × snapshot_date` month-end keys; the 143 registered columns in `conf/feature_groups.csv`; `y_product1..17`; binary, non-null `elig_product1..17`; `split`; and `snapshot_month` (reporting only). `CLIENT_JOINDATE` is registered but excluded from the 142 model features. Labels are binary for all mature rows, including gap/unused, and NULL for immature rows. `y=1, elig=0` exists; eligible filtering is mandatory, never treat those rows as negatives.

`ml.data_dictionary` must match every modeling column exactly: `column_name`, Spark `data_type`, and role `key / feature / excluded_feature / label / eligibility / split / aux`. 01 checks the complete unsampled input before creating its MLflow experiment, sampling, or fitting. It validates schema, keys, month ends, every product's labels/eligibility, and exact configured date-to-split assignments.

The field contract uses generic customer-portal and lead names: `customer_portal_activated`, `cli_customer_portal_login_*`, `portal_eStatement_*`, `portal_product_page_*`, `n_portal_login_*`, and the `leads` / `lead` families. Use `conf/feature_groups.csv` for exact names; upstream modeling-table columns and dictionary `column_name` values must match together. Models, feature specs, calibrators and releases built with a previous field schema cannot be reused directly: retrain, evaluate and publish a new release under this contract. Do not patch old artifact names or hashes to bypass compatibility checks.

`raw.l2_purchase_config` has one qualifying purchase per row: `purchase_id, client_id, l2, application_date, issue_date, feature_snapshot_date, sp_rp, par_nonpar, mass_higher, buying_for, elig_at_feature_snapshot`. Synthetic `feature_snapshot_date` is the last month end before application. Label horizon is `(s, add_months(s, 3)]`; readiness is direct `add_months(s, 5)` with one day clipping, not chained +3 then +2. `synthetic_truth.*` is never a pipeline input.

Map real physical names in `table_overrides`; the production input preparation itself is upstream work, not an implemented extra branch in 00.

## 4. Chronology and three complete presets

All steps load the same **complete** training YAML using `training_config` (default `conf/training.yaml`). Unknown/missing policy keys fail; this is not an overlay. Databricks job parameters populate widgets in all four tasks. For interactive execution, set `env=dev` explicitly; notebook defaults remain local. Relative config paths resolve against the repository root.

Only `timeline.mode=retrospective_holdout` is implemented, at extraction **2026-10-01**. H=3 and maturity=2 are fixed by the labels. Separate gap parameters are `train_calibration_gap_snapshots`, `calibration_test_gap_snapshots`, and `inner_gap_snapshots`; allocation counts are `test_snapshots`, `calibration_snapshots`, `inner_valid_snapshots`. Default gaps/counts are 3, minimum Train is 12. `train_max_snapshots` caps recent Train (null means uncapped), marking older mature rows `unused`. Insufficient history fails rather than trimming requests; sparse histories use observed dates.

`refit_scope=all_train|tuning_fit` controls champion fitting, default all_train including inner gap/validation; neither scope includes reserved calibration or Test. `tuning_fit_as_of`, `base_fit_as_of`, `calibrator_fit_as_of`, `evaluation_as_of` default to extraction when null, must be ordered for active stages and cannot exceed extraction.

Label non-overlap is **previous last + H ≤ next first**. Label availability is separately **max ready_on ≤ stage as_of**, including validation labels used to select iterations. Test and evidence must be ready by evaluation_as_of. A deployment simulation would additionally need fit_as_of≤prediction origin; this implementation does not establish that and does not force blanket five-month gaps.

For the existing 36 monthly snapshots 2023-10-31..2026-09-30, the contract derives:

| Block | calibrated / raw-retained | raw-reallocated |
|---|---|---|
| Train | 2023-10-31..2025-04-30, 19 | 2023-10-31..2025-10-31, 25 |
| tuning_fit | 2023-10-31..2024-10-31, 13 | 2023-10-31..2025-04-30, 19 |
| inner_gap | 2024-11-30..2025-01-31, 3 | 2025-05-31..07-31, 3 |
| inner_valid | 2025-02-28..04-30, 3 | 2025-08-31..10-31, 3 |
| outer gap1 | 2025-05-31..07-31, 3 | 0 |
| calibration allocation | 2025-08-31..10-31, 3 | 0 |
| outer gap2 | 2025-11-30..2026-01-31, 3 | same; now Train→Test |
| Final Test | 2026-02-28..04-30, 3 | same |
| immature | 2026-05-31..09-30, 5 | same |

- **`conf/training.yaml`**: enabled calibration, evidence source calibration; fitting only on the reserved block.
- **`conf/training.raw-retained.yaml`**: disabled, unchanged allocation, calibration window reserved for evidence/raw diagnostics only.
- **`conf/training.raw-reallocated.yaml`**: disabled with explicit calibration=0, gap1=0, evidence=inner_valid; expanded Train. Disabling calibration alone never reallocates rows.

All four effective as_of values in these presets are 2026-10-01. Default tuning/base readiness is 2025-09-30; reallocated readiness is 2026-03-31. Evidence readiness is 2026-03-31 and Test readiness 2026-09-30 in all three; enabled calibrator readiness is 2026-03-31. Default base readiness is later than calibration's first origin; calibrator readiness is later than Test's first origin: these are retrospective holdouts, not historical deployment results.

## 5. Model roles, evidence, reports and ranking

01 stores `tuning/model.txt` and `tuning/feature_columns.json` separately from champion `model.txt` and `feature_columns.json`. Their category dictionaries are fitted only on each model's actual fitting rows; baseline preprocessing is fit-only too. Both models use the selected iteration count. `inner_valid` score rows come from tuning, with role/artifact/spec/config/iteration provenance. They are iteration-selection diagnostics, not unbiased final performance. Calibration/Test and serving use champion only; the logistic baseline never replaces it.

Evidence counts distinct purchase_id with matching L2, eligible purchase flag and application date in `(block_first, block_last+H]`, using the training client sample. `evidence_source/evidence_event_count` determine tier; `calibration_event_count` counts the actual allocated calibration block (0 if absent); `test_event_count` is reporting only. Sparse-history counting remains the existing first-to-last event envelope, not an exact union of observed windows.

Default distinct evidence event thresholds are own E≥200, limited 100≤E<200, insufficient E<100. These configurable engineering settings do not guarantee statistical sufficiency. With calibration enabled, tier automatically selects the method; the configuration schema has no manual Platt/intercept selector. Own fits Platt without constraining its slope, then checks b>0; b≤0 triggers an intercept refit on the same calibration block (`intercept / calibrated_intercept_fallback`). Otherwise own is `platt / calibrated_platt`, limited is `intercept / calibrated_intercept`, and insufficient is `none / not_calibrated` and excluded. Inner_valid evidence is permitted only with disabled calibration; final Test can never choose tiers or methods. Changing the evidence allocation may change eligibility, but these presets happen to use the same Aug–Oct 2025 evidence dates, so their counts may coincide.

Disabled calibration performs **no fit**, sets method/status to `disabled` for every tier, serves NULL `calibrated_purchase_probability`, and forces raw ranking. The identity pickle and calibrator_run_id identify a compatible artifact/evaluation run, not a fitted calibrator. Reports contain only real score kinds: unavailable calibrated metrics are JSON `null` / console `N/A`, including enabled-but-insufficient products. No-cal allocation yields no calibration-block metric rows; selection calibration_end is NULL (some manifest presentation dates are stringified).

Final Test is report/gate only: AUROC, PR-AUC, log-loss, Brier, O/E, slope, lift@1/5/10%, reliability, per-snapshot O/E/slope; client-clustered bootstrap 200 for 95% AUROC/O/E CIs on the gate field. Gates remain AUROC≥0.60, lift10≥1.5, O/E [0.8,1.25], slope [0.7,1.3]. Missing/nonfinite values fail. Ranking eligibility requires own/limited, passing gates, and status other than not_calibrated; disabled never bypasses insufficient exclusion.

Scoring uses one release-level field times business_weight, descending per client, ties by L2 name, continuous ranks from 1. Eligible products can number fewer than top_k (default 3). Independently trained raw propensities are **not guaranteed comparable calibrated purchase probabilities** across products; raw cross-product ordering is a policy with this limitation.

03 retains pooled head/rule/share/fixed, validity masks, constraints (`on_missing: allow` currently), and margin abstention. Its saved `layer2_spec.base_feature_spec` learns categories from its own pooled Train purchases; 04 uses it exactly. Reasons use each L1 champion's spec and grouped pred_contrib; recommendation text excludes sensitive/non-explainable groups, while the audit contribution table includes flags for all groups.

<a id="layer1-math"></a>
### 5.1 Layer 1: target, learning and evaluation mathematics

For customer i, product ℓ and snapshot t, train/evaluate only `eligᵢ,ℓ,t=1` rows. The mature label `yᵢ,ℓ,t=1` means a qualifying purchase in `(t,t+3 months]`; otherwise it is 0. An additional two-month maturity delay gives direct `ready_on=t+5 months`. Inputs are snapshot features. Repeated customer/snapshot rows are not independent purchase events: one purchase may label several snapshots. Fit rows, customers and distinct purchase_id counts are different quantities. Client sampling retains histories; it does not balance positives and negatives. The current retrospective_holdout is not historical deployment replay.

With `objective=binary` and the unoverridden LightGBM default `sigmoid=1`, the additive-tree raw margin and raw probability are:

```text
Fℓ(x) = Fℓ,0 + η Σₘ₌₁ᴹ fℓ,m(x)；sℓ(x) = sigmoid(Fℓ(x))
loss(y,F) = −y ln(s) − (1−y) ln(1−s)
          = ln(1+exp(F)) − yF
gradient g = s−y；Hessian h = s(1−s)
```

Each tree adds a leaf update to the margin; trees' probabilities are not averaged. η is learning_rate and M is the selected iteration count. Saved models already include initialization/shrinkage; do not apply learning_rate again when loading. `lambda_l2` penalizes leaf values: for an ordinary leaf, the local second-order term is `G·w+½(H+λ)w²`, with `G=Σg, H=Σh`, and stationary update `w=−G/(H+λ)`. This illustrates regularization in a local leaf approximation, not LightGBM's entire constrained/sampled tree-building algorithm or sklearn's linear-model penalty objective.

The [default YAML](../conf/training.yaml) uses binary_logloss, num_leaves=31, learning_rate=0.05, min_data_in_leaf=200, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1, max_rounds=2000 and early_stopping_rounds=50. There are no class weights, SMOTE or scale_pos_weight. Tree row/feature subsampling is not label balancing. See [LightGBM 4.6 parameters](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html).

01 fits tuning on tuning_fit, chooses best_iteration on inner_valid, then fits a **separate champion** for fixed iterations on all_train (default) or tuning_fit. Reserved calibration/Test are excluded; disabling calibration does not disable this refit. Each actual fitting population supplies its category levels; numerics are float32 and unknown categories become NaN. Inner validation is an iteration-selection diagnostic, not unbiased final evaluation. Logistic baseline preprocessing is fit-only and it remains comparison-only: better baseline results produce WARN, never an automatic champion switch.

For a fixed product's eligible evaluation rows, p is the actual score kind under evaluation (raw or genuinely calibrated):

```text
AUROC = P(p⁺ > p⁻) + ½ P(p⁺ = p⁻)
k = ceil(0.10 n)
lift@10% = [sum of y in top k / k] / [Σy/n]
O/E = Σy / Σp
Brier = mean((p−y)²)
log-loss = −mean(y ln(p)+(1−y) ln(1−p))
```

AUROC needs both classes. Lift selects descending p, preserving input row order on ties, and needs a nonzero overall positive rate; this is distinct from the L2-name tie rule for cross-product ranking. O/E>1 indicates aggregate underprediction; O/E<1 indicates overprediction. O/E near 1 alone does not ensure accuracy in every probability band. Lower Brier/log-loss is better; implemented log-loss clips at 1e-15. Existing CIs cluster bootstrap by client, accounting for repeated customer rows rather than treating them as independent purchases.

**Illustrative arithmetic only:** ten already-descending probabilities `p=[0.90,0.80,0.70,0.60,0.50,0.40,0.30,0.20,0.10,0.05]` with `y=[1,0,1,0,0,1,0,0,0,0]` have 17 wins among 3×7=21 positive-negative pairs, no ties: AUROC=0.809524. The top ceil(1)=1 row is positive, so lift@10%=1/0.3=3.333333. Σp=4.55 and Σy=3 give O/E=0.659341; Brier=0.185250 and log-loss=0.533367. Good ordering can coexist with aggregate overprediction.

Evaluation calibration slope fits diagnostic `P(y=1)=sigmoid(α+β·logit(clip(p,1e-6,1−1e-6)))` on the **evaluated block**, using `LogisticRegression(penalty=None)`, and reports β and α (ideal reference 1 and 0). It does not alter serving predictions or select a calibrator on Test. This unpenalized diagnostic differs from the weak-L2 Platt fit on the separate calibration block below; one-class or degenerate inputs can make the diagnostic undefined/unstable.

For Top-3, first restrict to customer eligibility and product tier/gate eligibility; then sort `selected_probability × business_weight`, descending, breaking ties lexicographically by L2 name. The weight expresses policy, not a fitted expected-revenue or causal-uplift model. Probability comparability needs validation. Layer 2 guidance is generated after selection and does not rerank products.

Sources: [01](../notebooks/01_train_layer1.py), [modeling.py](../notebooks/modeling.py) (`train_l1_pair`, `binary_metrics`, `calibration_slope`), [04](../notebooks/04_batch_scoring.py). [Chinese specification](MVP_SPEC.md#layer1-math) · [Offline explanation](modeling_flow.html#layer1-math).

<a id="calibration-math"></a>
### 5.2 Calibration mathematics and what ordering preservation means

The [canonical Chinese explanation](MVP_SPEC.md#calibration-math) and [offline diagram](modeling_flow.html#calibration-math) describe the same implementation. Let `s` be the frozen champion's original score for an eligible row, `y∈{0,1}` its mature label, and `q` the clipped calibration input:

```text
q = clip(s, 10⁻⁶, 1−10⁻⁶) = min(max(s, 10⁻⁶), 1−10⁻⁶)
odds(q) = q/(1−q)                 logit(q) = ln(q/(1−q))
sigmoid(z) = 1/(1+exp(−z))
sigmoid(logit(q)) = q             logit(sigmoid(z)) = z
```

The inverse identities hold in exact real arithmetic. Odds compare event to non-event probability: 10% has odds 1/9, while 50% has odds 1. Clipping avoids infinite logit at 0 or 1; it can change the original score and cannot repair missing/invalid input.

**Intercept-only fixes b=1** and fits only `a`. Taking the odds of its sigmoid output gives the full conversion:

```text
z = a + logit(q)                  p = sigmoid(z)
p/(1−p) = exp(z) = exp(a) · q/(1−q)
p = exp(a) · q / (1−q + exp(a) · q)
```

It multiplies odds, not probability or a fixed number of percentage points. **All parameters and numbers below are illustrative mathematics, not fitted model parameters or observed acceptance results.** With `a=ln(2)`, 10%→18.18% and 50%→66.67%. With `a=0`, the output is `q`, equal to `s` only where clipping did not change it.

**Platt fits a and b** in `p=sigmoid(a+b·logit(q))`, equivalently `p/(1−p)=exp(a)·[q/(1−q)]ᵇ`. The intercept sets baseline odds: at q=0.5, p=sigmoid(a), with odds exp(a). The slope scales log-odds differences: `logit(p₁)−logit(p₂)=b·[logit(q₁)−logit(q₂)]`. Intercept-only is the prediction-form special case b=1, not the same fitting procedure. For a=0, b=0.5, 10%→25%, 50%→50%, 90%→75%; movement toward 0.5 is specific to this example, not a universal result for arbitrary a,b. The post-fit positive-slope check in 02 is not constrained optimization.

**Fit objective and library defaults.** Let `n` be the number of eligible calibration **fit rows**, including positive and negative labels. It differs from distinct evidence event count E: a purchase can label multiple snapshots. For pᵢ=sigmoid(zᵢ), mean binary cross-entropy is:

```text
L = −(1/n) Σᵢ [yᵢ ln(pᵢ) + (1−yᵢ) ln(1−pᵢ)]
  =  (1/n) Σᵢ [ln(1+exp(zᵢ)) − yᵢ zᵢ]
Stable per-row form: max(zᵢ,0) + ln(1+exp(−|zᵢ|)) − yᵢ zᵢ
```

The stable form avoids overflowing exp(z). Intercept fitting uses equivalent `logaddexp(0,z)−y*z`, fixes b=1 and minimizes mean BCE using bounded scalar search with bounds (-20,20). It rejects nonconvergence or an estimate within 1e-6 of either boundary; this is not an unconditional unbounded optimum claim.

Platt calls `LogisticRegression(C=platt_C, solver="lbfgs", max_iter=1000, tol=1e-10)` without sample or class weights. In verified local **scikit-learn 1.6.1**, the unoverridden defaults are `penalty="l2"` and `fit_intercept=True`. For this one-feature, unweighted fit, the objective is `J(a,b)=L(a,b)+b²/(2nC)`, with **a unpenalized**. Default C=1,000,000 makes regularization very weak, not mathematically zero. The code/config shorthand “effectively unpenalised” must not be read as exact unpenalized maximum likelihood. See the official [1.6 binary objective](https://scikit-learn.org/1.6/modules/linear_model.html#binary-case) and [LogisticRegression defaults](https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.LogisticRegression.html). This is a version-specific local reference, not a promise about an unverified Databricks Runtime; check its actual dependencies.

Both calibrators fit only frozen champion predictions/labels from independent calibration, never Train, and never optimize/select on Final Test. Both require positive and negative labels: a one-class fit raises an error rather than automatically becoming a successful intercept fallback. Missing/invalid scores or labels violate the 02 input contract, and missing required blocks/rows are rejected. Constant or nearly constant scores lack discrimination; calibration cannot create ordering. Clipping, extreme coefficients and floating-point saturation may produce ties or degenerate diagnostics. These boundaries are explanations, not claims that every such case has passed end-to-end testing. `none / not_calibrated` means enabled calibration lacked evidence; `disabled / disabled` means the global switch suppressed fitting for all tiers. Neither is a fitted probability conversion; insufficient remains excluded under both, with existing gates and business eligibility unchanged.

**Within-product ordering.** Differentiate with respect to the clipped input q, not the original s:

```text
dp/dq = p(1−p) · b · 1/[q(1−q)]
      = b p(1−p)/[q(1−q)] > 0
```

This holds in exact arithmetic for interior q, finite a and finite positive b; numerical use additionally needs nonsaturated arithmetic. Intercept-only satisfies it with b=1. Distinct original scores can clip to the same q, and finite-precision sigmoid can introduce further ties. For fixed evaluation rows/labels of one product, **identical pairwise strict ordering and ties** preserve AUROC, since it counts positive-negative wins and ties. Top-fraction ordering/membership is also preserved with unchanged tie handling. Do not claim unconditional raw AUROC or membership invariance when new ties appear. 02 checks actual Test Spearman correlation ≥1−1e-9; degenerate cases such as undefined constant-sequence correlation can be rejected. This guard does not establish probability accuracy.

**Across-product ordering for one customer.** Each product has its own function. The following four hypothetical eligible products have the same eligible set and equal business weights W=1 before/after, with every b=1:

| Product | Raw s=q | a | Calibrated p | Raw rank → calibrated rank |
|---|---|---|---|---|
| A | 40% | ln(1/4) | 14.29% | 1 → 4 |
| B | 30% | ln(2) | 46.15% | 2 → 1 |
| C | 20% | 0 | 20.00% | 3 → 3 |
| D | 10% | ln(4) | 30.77% | 4 → 2 |

The order changes from A→B→C→D to B→D→C→A, and Top-3 membership changes from {A,B,C} to {B,D,C}, even though each product's function is strictly increasing in q. This proves a possible ordering/membership change, not better recommendations or calibration. Cross-product calibrated probabilities still need consistent targets/data and validation of accuracy/comparability; monotonicity alone proves neither, and raw scores have no such guarantee.

Source mapping: [modeling.py](../notebooks/modeling.py) (`clipped_logit`, calibrators, tier selection), [02](../notebooks/02_calibrate_and_validate.py) (fitting, Spearman and gates), [configuration contract](../notebooks/l1_contract.py), [default preset](../conf/training.yaml). Latest 10% synthetic calibrated acceptance exercised three limited/intercept products with the others insufficient; the raw path was tested. Current full-volume own/Platt end-to-end acceptance has not run. Full-volume metrics in MVP_SPEC §14 remain historical. Numerical example checks are documentation checks, not new model, pipeline or Databricks acceptance.

<a id="layer2-math"></a>
### 5.3 Layer 2: attribute distributions conditional on purchase

Training rows are eligible **purchase events** joined to their feature snapshots. The target is `P(A=v | purchased product ℓ, x, eligible)`, an observed attribute-choice pattern among purchasers. Nonbuyers are not negative examples here. This is neither another purchase propensity nor `P(purchase AND configuration)`, and it does not estimate the benefit of changing a configuration.

Each of sp_rp, par_nonpar and buying_for has a pooled LightGBM head with categorical l2 added to the snapshot features. Configured class order is `[SP,RP]`, `[Par,NonPar]`, and `[self,spouse,child,parent]`, respectively. A binary head's output r means the **second value**, RP or NonPar, and is expanded to `[1−r,r]`. Binary heads use BCE; the four-value buying_for head uses multiclass softmax and categorical cross-entropy:

```text
Binary: r = sigmoid(F(x,ℓ)); L = −mean(y ln(r)+(1−y) ln(1−r))
Multiclass: r_v = exp(F_v)/Σ_u exp(F_u)
L = −(1/n) Σᵢ Σ_v 1[Aᵢ=v] ln(rᵢ,v)
```

The head training pool includes valid Train purchases from **all L2s with at least two valid values**, even L2s that will serve share fallback. It early-stops on the last three **observed application months** in that pool and refits all pooled Train rows at fixed selected iterations. The pooled Train supplies Layer 2's own category dictionary. Layer 2 dates are independent: currently Train≤2026-01-31, Test 2026-02-01..07-31, not a clone of the L1 split. There is no implemented Layer 2 calibrator.

Serving method is decided separately per L2/attribute. One valid value means fixed; no valid value means not_applicable. With multiple values, a head requires **every valid value** to have at least 200 valid Train purchase rows; otherwise it uses Laplace share. This min_rows_per_value=200 is unrelated to the L1 own calibration tier's 200 distinct evidence events. For K valid values, share is `r_v=(n_v+1)/(Σvalid n_u+K)` and invalid values get 0. Counts `[8,1,0]` yield `[9/12,2/12,1/12]=[0.750000,0.166667,0.083333]`. Fixed is a one-hot value. The multi-value mass_higher rule returns HigherEnd for wealth_seg3 in UHNW/HNW/Affluent, otherwise Mass, **including missing wealth_seg3**. Rule/fixed/share perform no learned loss optimization; validity/constraints still apply.

Product valid values are attribute-wise projections of the validity table (all configured values if an attribute column is absent). Customer constraints require Married for spouse, age<50 for parent, age>25 for child. All three currently have on_missing=allow: an unknown input does not rule out that value, rather than proving the condition true. With m_v indicating both product validity and customer permission, finite probabilities become:

```text
r′_v = m_v r_v / Σ_u(m_u r_u)
margin = top r′ − runner-up r′
```

Code uses `np.where(mask,r,0)` before summing, so an excluded NaN can be removed. A zero or nonfinite **retained total** yields not_applicable; not every NaN in the original vector necessarily invalidates the row. Head/share abstain only when margin<0.20; exact equality passes by policy (the implementation compares actual floating-point values). Fixed/rule are not margin-abstained. One remaining nonzero value normalizes to 1 with margin 1; top-value ties use configured value order before abstention is checked.

Illustrative buying_for probabilities `[0.45,0.35,0.15,0.05]` with mask `[1,0,1,0]` retain 0.60 and normalize to `[0.75,0,0.25,0]`. Margin 0.50 returns self. If `[0.45,0.40,0.10,0.05]` is fully allowed, margin 0.05 makes head/share abstain. The three-value share example above, fully allowed, has margin 0.583333 and passes. These are math examples, not fitted head results.

Guidance is per attribute: multiplying separate head probabilities does not establish a valid or optimal joint configuration. Attribute-wise validity projections are not a joint solver. Guidance comes **after Top-3 product selection** and does not rerank products. Keep observed purchaser choices, purchase propensity and causal benefit distinct.

Sources: [03](../notebooks/03_layer2_and_reasons.py) (`l2_plan`, `train_head`), [modeling.py](../notebooks/modeling.py) (`class_probabilities`, `restrict_probabilities`, `layer2_guidance`), [Layer 2 config](../conf/layer2.yaml), [validity table](../conf/l2_config_validity.csv). [Chinese](MVP_SPEC.md#layer2-math) · [Offline](modeling_flow.html#layer2-math).

<a id="shap-math"></a>
### 5.4 SHAP and reasons: scope, units and display filtering

`modeling.feature_contributions` calls native `LightGBM Booster.predict(X,pred_contrib=True)`. For binary models, the returned shape is n×(d+1): d feature contributions and a trailing Booster bias. The helper returns only the feature columns to reasons, but the bias is required for the full decomposition. No shap.TreeExplainer, user background DataFrame, interventional background, probability/logloss model_output or Layer 2 explainer is constructed.

Native TreeSHAP uses saved tree path/count information: see [LightGBM 4.6 Booster.predict](https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.Booster.html#lightgbm.Booster.predict) and [official tree.h](https://github.com/lightgbm-org/LightGBM/blob/v4.6.0/include/LightGBM/tree.h). Do not invent a population-average customer, current scoring batch or user-selected reference cohort as this call's background. Bias φ₀ is the model's **raw-output baseline**, not mean purchase probability and not defined as zero.

For the current sigmoid=1 binary L1 champion, additivity is on raw log-odds (margin):

```text
Fℓ(x) = φ₀,ℓ + Σⱼ φⱼ,ℓ(x)
sℓ(x) = sigmoid(Fℓ(x))
Illustrative waterfall: −2.0 → +0.6 → +0.3 → −0.1
F = −2+0.6+0.3−0.1 = −1.2; s = sigmoid(−1.2) = 23.15%
```

A positive contribution raises the margin; +0.3 is log-odds, **not +30 percentage points**. Sigmoid is nonlinear, so transforming individual terms does not give additive probability contributions. Each product has its own bias/model; SHAP magnitudes across products are not probability-benefit comparisons. X uses the champion's saved feature_columns order, category levels and float32 numerics; unknown categories become NaN. Features are LightGBM inputs, not logistic-baseline one-hot columns. The existing 03 smoke takes up to 100 Test purchase rows and checks feature-sum+bias against raw_score=True with max error<1e-6. This describes existing implementation/coverage; no smoke or training was rerun for this documentation.

Group totals are `Φ_g=Σⱼ∈g φⱼ`. Reasons retain only positive, explainable, nonsensitive groups, sorted descending, at most top_n_reasons=3, with stable configuration-order ties. A positive group's representative is its largest positive-direction contributor; a negative audit group's representative is its most negative contributor. That representative is not the group total. Audit rows may retain all groups and sensitive/explainable flags, optionally feature contributions. Displayed positive top-three reasons omit bias, negative and filtered groups: **they are not a full additive decomposition**. An omitted negative group does not mean its features were unused. Business grouping establishes neither feature independence nor permission to surface filtered attributes.

The implemented explanation is exclusively for the **pre-calibration L1 champion**. It does not decompose calibrated probability, business weights, eligibility/gates, cross-product rank, Layer 2 heads/rules or attribute choices. Preserving one product's customer order under monotonic calibration does not make these SHAP reasons a complete explanation for Top-3: those other stages remain outside the explainer, and different product calibrators can change cross-product order.

Without input-changing clipping or numerical saturation, algebra permits `logit(p)=a+bF=(a+bφ₀)+Σⱼ bφⱼ`. This affine logit transformation is **NOT IMPLEMENTED as an explanation output** and is not a computed calibrated-probability SHAP decomposition. After sigmoid, per-feature probability effects are not additive; clipping also prevents substituting F for logit(q).

SHAP attributes an association in a fixed model under its allocation assumptions, not causality, actionability, a guaranteed counterfactual or benefit from changing a feature. Correlated inputs and background/path assumptions affect allocation. Neither causal identification, feature independence nor fairness is guaranteed.

Sources: [modeling.py](../notebooks/modeling.py) (`feature_contributions`, `group_contributions`, `explain_rows`), [03](../notebooks/03_layer2_and_reasons.py) additivity check, [04](../notebooks/04_batch_scoring.py), [reason groups](../conf/reason_groups.yaml), [feature groups](../conf/feature_groups.csv). [Chinese](MVP_SPEC.md#shap-math) · [Offline](modeling_flow.html#shap-math).

## 6. Artifacts, table versions and output commit

`training_provenance.json` records effective YAML/hash, exact dates/readiness/as_of, refit scope, model role/path/iteration, feature-spec hashes, model SHA-256 and training input names/versions. `calibration_provenance.json` adds the original evaluation/calibrator run, calibrator hash, evidence and 02 input pins. Selection and release/scoring manifests carry this lineage; both files and MLflow loaders check original identities and hashes. Old provenance-free or mixed/stale artifacts require a correctly trained fresh release, not relabeling.

Input identity currently means **logical name → physical table name + Delta version**, not a Delta table UUID. 04 captures all three input versions before any input read and uses the pinned versions for selection, Layer 2 models and scoring snapshot. It never falls back to latest on a failed pin. These are per-table pins, **not an atomic multi-table snapshot**; 01/02 also record their own pins. Do not claim the same mechanism covers all 03 reads.

File layout includes `<release>/<l2>/{model.txt, feature_columns.json, calibrator.pkl, training_provenance.json, calibration_provenance.json}`, `<release>/layer2/` specs/heads and `manifest.json`. 02 refuses an existing release by default; although an explicit allow_overwrite option exists, normal validation and config changes require unique releases. 03 adds Layer 2 to that release; simultaneous writers to one release are unsupported.

| Output | Grain / contract |
|---|---|
| `scores.customer_l2_scores` | run_date × client × L2; probabilities, evidence/tier/gates, ranking, guidance/reasons, versions and score provenance |
| `scores.customer_top_recommendations` | rank≤top_k subset |
| `scores.customer_reason_contributions` | Top-k × reason_group, sensitivity/explainability flags; optional feature contributions |
| `scores.scoring_runs` | one committed run_date marker with manifest summary |

04 chooses the latest snapshot≤run_date within 35 days. Ineligible rows have NULL scores/rank and `not_eligible`; otherwise excluded products use `not_ranking_eligible`. It deletes the previous date's scoring_runs marker, writes/rechecks the three data tables, then writes the marker last. Consumers must join it. Output partitions use replaceWhere, writer-level mergeSchema, and partition column last; read by name.

Release file publication uses staging plus rename with interrupted-swap recovery. Delta selection precedes bundle publication; a failure can leave a partial release and files loader refuses mismatches. UC Volume rename behavior remains to be validated; do not claim a transaction spans Delta tables and files.

## 7. Exact job and interactive parameters

The job has only job-level parameters (no task base_parameters):

| Name | Default | Meaning |
|---|---|---|
| `training_config` | `conf/training.yaml` | same complete configuration for 01–04 |
| `env` | `${bundle.target}` | environment profile |
| `release_id` | `mvp-{{job.run_id}}` | shared release; required by 03/04 on Databricks |
| `run_date` | empty→today | 04 scoring date |
| `l2_list` | pilot | 01/02/04 selection; all or comma-separated also accepted |

Interactive notebook parameters also include 01 sample_fraction and 02 allow_overwrite. There is **no job-level sample_fraction parameter** in current YAML. Any sampled interactive run must set the existing widget in its notebook; a sampled job would require an explicit future job-config change.

Write-producing validation must use an isolated environment and unique release. The migration package does not include the local runner or synthetic preparation. See [MVP_SPEC §15](MVP_SPEC.md#configuration-usage) for complete-config selection.

## 8. Migration checklist and remaining validation

1. Resolve existing MIGRATION placeholders: workspace target, catalog/table mapping, exact DBR runtime/node type, experiment root, artifacts Volume and intended access. This document does not change these settings.
2. Confirm the dated L1 extraction/as-of values (currently 2026-10-01) and independent L2 application-date windows in `conf/layer2.yaml`. Produce real modeling/dictionary/purchase inputs upstream with the chosen config. Validate the full 01 contract; never substitute 00 for production preparation.
3. Confirm the packages listed in [README](../README.md) in the exact chosen ML Runtime, keep shared modules importable, and confirm driver capacity; verify file publication semantics on the intended Volume in a separately approved test release.
4. Run real remote bundle validation before any authorized deployment. Prior local bundle-schema checks returned 0 errors; remote validation/deployment remain uncompleted, not silently accepted.
5. Execute staged real-data acceptance with new releases; verify provenance, model-source score/rank equality, negative stale-artifact behavior, NULL semantics, all gates/exclusions, and marker-last output. Scoring-only orchestration must supply the compatible frozen release; current job retrains 01–04 and does not implement a separate production scheduler/evaluator.

Prior synthetic validation: 68 broad checks before report correction, 39 focused afterward, and 5 reporting cases repeated independently. Two isolated 10%/6-pilot flows completed; score/rank equality and saved-model reproduction were checked with the source-timing and sampling limits recorded in [MVP_SPEC §12](MVP_SPEC.md). These are prior results, not new packaging or real-environment acceptance. Full raw-retained flow, current full-volume acceptance and remote execution have not run. Historical Ruff has 51 existing diagnostics, 0 added by the business fixes.

Deferred: **Point 5 production evaluator/monitoring**, rolling calibration/OOF, isotonic, Optuna, UC alias, post-evaluation refit on all mature data, seasonality features, joint configuration guidance, negative downsampling and cross-product Hit@k baseline. The existing Train-only champion refit is implemented and is distinct from the deferred final refit.
