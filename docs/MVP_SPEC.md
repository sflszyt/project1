# 10 月 Notebook MVP 规格（当前契约，2026-10-04）

本文件是当前业务规格的唯一说明入口，整合 2026-10-03 共识与已验收的 2026-10-04 修正。实现依据为实际 `conf/`、`notebooks/01..04`、共享模块和 `resources/mvp.job.yml`；历史验收数字单列于 §14，不代表本次重新执行。英文迁移契约见 [GENIE_HANDOVER](GENIE_HANDOVER.md)，运行库与工作区准备见 [README](../README.md)。

## 1. 目标与范围

以几个 notebook 和 YAML 完成 Layer 1 训练、可选校准、时间留出评价、Layer 2 配置指导、reasons 与批量 Top-k 打分。17 个 L2 为 `product1..product17`；6 pilot 为 product1/5/11/8/16/14。`l2_list` 接受 `pilot`、`all` 或逗号分隔列表；没有单独的 `--all` 开关。tier 根据实际抽样、窗口和事件数确定，不按产品写死。

本地 Windows 使用已有 Python/JDK/Hadoop；Databricks 面向 DBR 18 LTS ML、Unity Catalog、MLflow 3 与 Asset Bundles，实际远端接入尚待验证。不建 `src/` 平台、不打 wheel、不恢复归档重型框架。

## 2. 输入契约

| 逻辑名 | 粒度与内容 | 当前职责 |
|---|---|---|
| `raw.client_snapshot_monthly` | client × 月末，143 特征 + `y_product1..17` + `elig_product1..17` | 本地合成准备源，00 读取 |
| `raw.l2_purchase_config` | 一次合格购买：`purchase_id, client_id, l2, application_date, issue_date, feature_snapshot_date, sp_rp, par_nonpar, mass_higher, buying_for, elig_at_feature_snapshot` | 01/02 的事件计数、03 的购买属性训练 |
| `ml.modeling_table` | 快照宽表 + `split` + `snapshot_month` | 01/03/04 的建模/打分输入；02 校验时间契约 |
| `ml.data_dictionary` | 每列的 `column_name, data_type, role` | 01 读取并与全表 schema 核对 |

真实上游须在 **01 前**提供后三张所需表；若使用自己的物理名，用环境配置 `table_overrides` 映射。00 只准备合成输入，不能指向真实表作为真实预处理选项。`synthetic_truth.*` 不得由 pipeline 读取。

`y=1` 表示购买申请日期落在 `(snapshot_date, add_months(snapshot_date, 3)]`。`ready_on = add_months(snapshot_date, 5)`，直接相加 H+m，一次按目标月天数裁剪；晚于 extraction 的标签为 NULL，所有成熟行（包括 gap/unused）须为 0/1。`elig` 永不 NULL，须为二值；存在 `y=1, elig=0`，这些行不能当负例，训练/校准/评估按 `elig=1` 过滤，打分对不合资格行置空。

快照键 `(client_id, snapshot_date)` 唯一、非空且日期为月末。特征列须等于 `conf/feature_groups.csv` 的 143 个登记列；`CLIENT_JOINDATE` 属输入契约但排除建模，实际模型为 142 特征。`client_id, snapshot_date, snapshot_month, y_*, elig_*, p_true_*` 不作特征。字典角色精确匹配 `key / feature / excluded_feature / label / eligibility / split / aux` 和 Spark 类型。

当前字段采用通用客户门户和线索命名：`customer_portal_activated`、`cli_customer_portal_login_*`、`portal_eStatement_*`、`portal_product_page_*`、`n_portal_login_*` 以及 `leads` / `lead` 字段族。精确名称以 `conf/feature_groups.csv` 为准；上游建模表和字典 `column_name` 必须同步匹配。字段 schema 更新后，旧模型、feature spec、校准器和 release 不可直接复用，须重新训练、评估并生成新 release；不能仅修改旧 artifact 中的名称或哈希。

现有合成历史为 36 个快照 2023-10-31..2026-09-30；按当前 extraction，最后成熟快照是 2026-04-30，2026-05-31..09-30 五个快照标签为 NULL。一次购买可能形成多个正标签，事件数不能用正标签行数替代。

## 3. 布局与入口

| 路径 | 职责 |
|---|---|
| `notebooks/common.py` | 参数/配置、平台、Spark/Delta、preflight、MLflow、文件包发布 |
| `notebooks/l1_contract.py` | 完整 YAML 校验、实际快照日期推导、成熟度、L1 provenance 校验 |
| `notebooks/modeling.py` | pandas/LightGBM、校准、指标、事件计数、Layer 2 与 reasons |
| `notebooks/01_train_layer1.py` | tuning + champion/refit + logistic baseline |
| `notebooks/02_calibrate_and_validate.py` | evidence、可选校准、Final Test 报告/gate、release |
| `notebooks/03_layer2_and_reasons.py` | Layer 2 与文件包增量 |
| `notebooks/04_batch_scoring.py` | 固定输入、加载、打分和最后提交标记 |
| `conf/training*.yaml` | 三份独立完整配置，非 overlay |
| `conf/products.yaml`, `layer2.yaml`, feature/reason CSV/YAML | 产品、权重、属性、特征和理由策略 |

迁移包不包含本地合成准备器 00、Windows Spark worker、runner、测试、原始参考文档、数据和模型产物。notebook 同目录导入 `common.py`、`l1_contract.py`、`modeling.py`，`repo_root()` 向上寻找 `conf/products.yaml`。Databricks 分支直接使用 Runtime Spark，不需要本地 `.venv/` 或 `tools/`。

## 4. 公共接口与参数

环境显式传入 `load_env(env)`；`load_conf('training')` 按所选 `training_config` 读取并校验完整 YAML。`get_param(name, default)` 在本地按 CLI → `PRODREC_<NAME>` → 默认取值，空字符串返回默认；Databricks 读取 widgets/job parameters。相对配置路径相对 repository root。

`common.derive_split_dates(training_conf, snapshot_dates)` 委托 `l1_contract.derive_dates`，返回 Train、两段 outer gap、calibration、test、unused、immature，以及 tuning_fit、inner_gap、inner_valid、refit、evidence、availability。`read_table(..., version_as_of=整数)` 使用 Delta `versionAsOf`，版本不可用时报错，不回落最新版本。`delta_version` 捕获实际已提交版本。

`validate_training_input` 校验完整未抽样建模表与字典，`sample_clients` 再按 client 哈希保留整条时间序列。`write_table` 支持分区替换与 writer 级 `merge_schema`。完整命令见 §15。

<a id="timeline"></a>
## 5. 时间分配、标签可用性与事件数

当前唯一模式是 `timeline.mode: retrospective_holdout`，提取日为 **2026-10-01**。它回答“提取时可用数据上的时间留出表现”，不声称历史预测起点已具备这些标签或可部署。

| 配置键 | 当前三个 preset 的值或含义 |
|---|---|
| `timeline.horizon_months / maturity_months` | 固定输入标签契约 H=3、m=2，其他值拒绝 |
| `timeline.*_as_of` | `tuning_fit_as_of / base_fit_as_of / calibrator_fit_as_of / evaluation_as_of`；null 均解析为 extraction |
| `split.test_snapshots` | 3 |
| `split.calibration_snapshots` | calibrated / raw-retained 为 3；raw-reallocated 为 0 |
| `split.train_calibration_gap_snapshots` | 前两者 3；raw-reallocated 必须显式 0 |
| `split.calibration_test_gap_snapshots` | 3；cal=0 时成为 Train→Test gap |
| `split.inner_valid_snapshots / inner_gap_snapshots` | 各 3，位于 Train 尾部 |
| `split.min_train_snapshots` | 12；请求块不足时报错，不默默缩短 |
| `split.train_max_snapshots` | 默认 null；正整数限制最近 Train，且不能小于 min_train；更早成熟行标 `unused` |
| `split.refit_scope` | `all_train`（默认）或 `tuning_fit`；champion 实际拟合范围 |

先以实际观测月末日期判断成熟度，再从末尾分配 Test、gap2、calibration、gap1，剩余成熟日期分配 Train（受 cap 限制）。Train 尾部再拆 inner_valid 和 inner_gap。稀疏历史不虚构月份；gap 数量表示观测快照数，是否隔离标签由真实日期决定。

两个不同条件分别检查：

- **标签不重叠**：`add_months(前块末快照, H) <= 后块首快照`。适用于 tuning_fit→inner_valid、Train→calibration→Test；无 calibration 时检查 Train→Test。
- **标签已可用**：参与阶段标签的 `max(ready_on) <= 对应 as_of`；tuning 包括用于迭代选择的 inner_valid 标签，base 使用实际 refit，calibrator 使用保留校准块，Test 与 evidence 在 evaluation_as_of 可用。

各 as_of 不晚于 extraction；tuning≤base≤evaluation，启用校准时还须 base≤calibrator≤evaluation。若要声称部署模拟，还需相应 `fit_as_of <= prediction origin`；当前没有实现该模式。不能用 H+m 一律替代边界 H，也不要求所有 gap 都是 5 个月。

日期运算是直接 `add_months(s, H+m)`，不是先 H 再 m：例如 2026-11-30 +5 月 = 2027-04-30；先 +3 到 2027-02-28 再 +2 会得到不同的 04-28，不采用后者。

下表从当前三份 YAML 和合成历史的实际月末日期推导；每个范围端点均为月末：

| 阶段 | calibrated / raw-retained | raw-reallocated |
|---|---|---|
| Train | 2023-10-31..2025-04-30（19） | 2023-10-31..2025-10-31（25） |
| tuning_fit | 2023-10-31..2024-10-31（13） | 2023-10-31..2025-04-30（19） |
| inner_gap | 2024-11-30..2025-01-31（3） | 2025-05-31..07-31（3） |
| inner_valid | 2025-02-28..04-30（3） | 2025-08-31..10-31（3） |
| gap1 | 2025-05-31..07-31（3） | 无（0） |
| calibration | 2025-08-31..10-31（3） | 无（0） |
| gap2 | 2025-11-30..2026-01-31（3） | 同左 |
| Final Test | 2026-02-28..04-30（3） | 同左 |
| immature | 2026-05-31..09-30（5） | 同左 |
| evidence | calibration：2025-08-31..10-31 | inner_valid：2025-08-31..10-31 |
| tuning/base ready_on（默认 all_train） | 2025-09-30 | 2026-03-31 |
| calibrator ready_on | calibrated：2026-03-31；raw：不拟合 | 不拟合 |
| evidence / Test ready_on | 2026-03-31 / 2026-09-30 | 同左 |

三份 preset 的四个有效 as_of 都是 2026-10-01。默认 Train 末标签到 2025-09-30 才成熟，晚于 calibration 首日 2025-08-31；calibration 末标签到 2026-03-31 才成熟，晚于 Test 首日 2026-02-28。这正是回顾性留出与历史部署可用性的区别。

**事件口径**：`modeling.count_events` 数同一 L2、`elig_at_feature_snapshot=1`、申请日期落在 `(块首快照, 块末快照+H]` 的 distinct `purchase_id`，02 沿用 01 的 client 抽样比例。`evidence_source/evidence_event_count` 决定 tier；`calibration_event_count` 始终计实际 calibration 分配（无块为 0）；`test_event_count` 仅报告，绝不能作为 evidence 或选校准方法。

保留限制：上述首尾“事件包络”在稀疏历史中会包含缺失快照之间的日期，不等于逐个实际标签窗口的精确并集。本次没有改计数政策。当前 calibrated 与 raw-reallocated 的 evidence 日期恰相同，数量可以相同；其他配置可能改变 evidence 时期、数量及资格。

## 6. 本地合成准备边界（不随迁移包分发）

读取已有合成 `raw.client_snapshot_monthly`，按选定 YAML 生成 `split` 与报告用 `snapshot_month`，验证键、登记特征和标签后写 `ml.modeling_table` 与 `ml.data_dictionary`。`split ∈ {train, gap, calibration, test, immature, unused}`；outer gap 合为 `gap`，inner gap 仍属 Train。

`allow_synthetic_data: false` 时拒绝运行。00 不是生成真实特征/标签的入口，也不在 Databricks job；真实上游负责同契约产表，01 负责完整 preflight，不修复旧 split。

## 7. Notebook 01：tuning 与 champion 分开

preflight 在创建 MLflow experiment、抽样与拟合前检查全表：列/字典名称、角色和类型，唯一非空键、月末、全部产品标签与 eligibility、精确 date→split 对应、成熟度及时间边界。此前修正 1 持续有效。

每个 L2 只使用 eligible Train。tuning 在 tuning_fit 拟合 LightGBM，在 inner_valid 早停选择 `best_iteration`；然后另建 champion，以固定迭代数按 `refit_scope` 重拟合。`all_train` 包括 inner gap 和 inner_valid；`tuning_fit` 仅该块。两者始终排除保留 calibration 和 Final Test。

tuning 类别字典只从 tuning_fit 学习；champion 字典只从实际 refit 行学习。logistic baseline 的重尾检测、填充、标准化、one-hot 等使用其 fit 样本（受 `baseline_sample_rows` 上限）；仅比较报告，优于 LightGBM 时 WARN，不替换 champion。LightGBM 不采用 class weights、SMOTE 或 scale_pos_weight。

| 用途 | 模型/artifact | 转换/spec | 预测含义 |
|---|---|---|---|
| inner_valid | `tuning/model.txt`，role `tuning` | `tuning/feature_columns.json` | 真实 tuning 留出预测；迭代选择诊断，非无偏最终评价 |
| calibration、test、04 服务 | `model.txt`，role `champion_refit` | `feature_columns.json` | 冻结 champion 预测 |

同一 L2 的两模型保存在同一训练 MLflow run 的不同路径；`model_run_id` 是 run 标识，不代替模型角色。保存 `training_provenance.json`、feature importance、baseline 及 LightGBM MLflow model。`ml.l1_raw_scores` 每行带 run、role、artifact、best_iteration、配置/spec 哈希；`ml.l1_training_runs` 保存原始训练 provenance。

<a id="layer1-math"></a>
### 7.1 Layer 1 数学：学什么、如何学习、怎样评价

#### 预测对象与样本单位

对客户 i、产品 ℓ、快照 t，只有 `eligᵢ,ℓ,t=1` 的行进入该产品的训练与评价。标签 `yᵢ,ℓ,t=1` 表示在 `(t, t+3个月]` 内有合格购买，否则成熟标签为 0；另等 2 个月才认为标签完整，即直接计算 `ready_on=t+5个月`。模型估计的是该产品 eligible 客户在这个窗口内的购买倾向，输入 x 是当时的快照特征，不包含未来购买属性。

训练单位是客户×快照行，并非互相独立的购买事件：同一客户会重复出现，一次购买也可使多个快照为正。拟合行数、客户数、distinct purchase_id 三者不同；client 抽样保留客户的时间序列，不是正负样本均衡。当前 `retrospective_holdout` 只声称提取时可用数据上的回顾性时间留出，不能解释成历史部署回放；§5 的窗口与成熟度契约保持不变。

#### LightGBM 的 margin、损失和树更新

每个产品有自己的二分类 boosted-tree 模型。当前 `objective=binary` 且没有覆盖 LightGBM 的默认 `sigmoid=1`，所以原始概率 s 与 raw margin F 的关系为：

```text
Fℓ(x) = Fℓ,0 + η Σₘ₌₁ᴹ fℓ,m(x)；sℓ(x) = sigmoid(Fℓ(x))
ℓ(y,F) = −y ln(s) − (1−y) ln(1−s)
       = ln(1+exp(F)) − yF
g = ∂ℓ/∂F = s−y；h = ∂²ℓ/∂F² = s(1−s)
```

f 是一棵树给当前行所属叶子的更新，η 是 learning_rate，M 是选定轮数；各轮在已有 margin 上增加修正，并非对树的概率取平均。初始项与缩放由 LightGBM 保存，不能在读取模型后再重复乘 learning_rate。二元交叉熵惩罚高信心的错误预测；梯度与 Hessian 给出当前误差的方向及局部曲率。

`lambda_l2` 对叶值做 L2 正则。用一个普通叶子的局部二阶近似说明其作用：若叶内 `G=Σg, H=Σh`，关于叶更新 w 的项为 `G·w + ½(H+λ)w²`，相应驻点是 `w=−G/(H+λ)`；更大的 λ 会压缩同样 G/H 下的更新。这是解释叶更新的局部近似，不是把 sklearn 线性模型的惩罚目标当成 LightGBM 的完整训练算法；实际还受树分裂、叶约束、采样和 shrinkage 等影响。

当前 [training.yaml](../conf/training.yaml) 为 `binary_logloss`、`num_leaves=31`、`learning_rate=0.05`、`min_data_in_leaf=200`、`feature_fraction=0.8`、`bagging_fraction=0.8`、`bagging_freq=1`、`lambda_l2=1`、最多 2000 轮、50 轮 early stopping。特征/行采样用于树训练，不是 class weights；未使用 class weights、SMOTE 或 scale_pos_weight。默认参数含义参见 [LightGBM 4.6 参数](https://lightgbm.readthedocs.io/en/v4.6.0/Parameters.html)。

01 在 tuning_fit 训练、inner_valid 选 best_iteration，再另建固定轮数的 champion；默认 all_train 重拟合，或按配置只用 tuning_fit，始终排除 calibration/Test。关闭校准也不会取消这个 Train 内 refit。类别水平分别从各模型的实际拟合人口学习；未知类别编码为 NaN，数值为 float32。inner_valid 已参与轮数选择，其指标是 tuning 诊断，不是无偏最终成绩。logistic baseline 使用自己的 fit-only 预处理，只作比较；更优时 WARN，绝不自动换掉 champion。

#### 指标各回答什么问题

在固定某一产品的 eligible 评价行上，p 表示本次评价的实际概率字段（raw 或真实 calibrated）。AUROC 衡量正例是否排在负例之前，lift 衡量高分人群的正例集中度；O/E、Brier 与 log-loss 检查概率刻度，而不是只看名次：

```text
AUROC = P(p⁺ > p⁻) + ½ P(p⁺ = p⁻)
k = ceil(0.10 n)
lift@10% = [Σ(前 k 行的 y)/k] / [Σy/n]
O/E = Σy / Σp
Brier = (1/n) Σ(p−y)²
log-loss = −(1/n) Σ[y ln(p)+(1−y) ln(1−p)]
```

AUROC 需要同时有正、负例；只有一类时不能赋一个有意义的 AUROC。lift 先按 p 降序，同分保留评价行顺序，分母正例率必须非零；这与跨产品排名按 l2 名字典序平分是不同规则。O/E>1 表示该批观测正例数多于预测总和（整体低估），O/E<1 表示整体高估，但 O/E≈1 本身不能证明每个概率区间都准确。Brier 和 log-loss 越低越好；实际 log-loss 以 `1e-15` 裁剪端点避免 log(0)。重复客户的依赖也解释了为何现有 CI 按 client 聚类 bootstrap，不能将每行视为独立购买。

**数学示例，非模型成绩**：十行已按概率降序排列，`p=[0.90,0.80,0.70,0.60,0.50,0.40,0.30,0.20,0.10,0.05]`，`y=[1,0,1,0,0,1,0,0,0,0]`。正负例配对共 3×7=21 对，正例胜出 17 对且无并列，所以 AUROC=17/21=0.809524。top 10% 取 ceil(1)=1 行，命中率 1，整体率 0.3，lift=3.333333；Σp=4.55、Σy=3，O/E=0.659341，Brier=0.185250，log-loss=0.533367。这个例子能把正例排得较前，却总体高估概率；排序好不等于校准好。

评价中的 calibration slope 是在**所评价块**上另拟合诊断回归 `P(y=1)=sigmoid(α+β·logit(clip(p,1e-6,1−1e-6)))`，实现为 `LogisticRegression(penalty=None)`，报告 β 和 α；理想刻度参照 β≈1、α≈0。它不更新服务模型，也不利用 Test 选择校准器。这个无惩罚诊断不能与 §8.1 在独立 calibration 块拟合、带弱 L2 的 Platt 参数 a,b 混同；单类或退化输入也会使诊断无定义或不稳定。

#### 概率如何变成 Top-3

先确定该客户的 eligibility 与产品 tier/gate 放行集合，再在集合中计算 `ranking_scoreᵢ,ℓ = selected_probabilityᵢ,ℓ × business_weightℓ`，降序取 Top-3，同分按 l2 名字典序。权重表达业务排序策略；这个乘积不是已建模的期望收入，也不是联系客户后的因果增益。不同产品的概率可比性仍需验证；Layer 2 的属性指导是在选中之后生成，不回头重排产品。

实现对应：[01](../notebooks/01_train_layer1.py)、[modeling.py](../notebooks/modeling.py) 的 `train_l1_pair / binary_metrics / calibration_slope`、[04](../notebooks/04_batch_scoring.py)。同义入口：[英文](GENIE_HANDOVER.md#layer1-math)、[离线图解](modeling_flow.html#layer1-math)；概率刻度变换见 [校准数学](#calibration-math)。

## 8. Notebook 02：证据、可选校准与最终留出

加载并固定输入版本，核对原训练 run、完整有效 YAML/hash、时间分配、模型/spec 文件及上游物理表名/版本。只消费 champion 的 calibration/test score，不把 tuning 预测拿来拟合校准器。

<a id="calibration-policy"></a>
| evidence tier | distinct evidence 事件数 E（默认阈值） | 启用校准时的 method / status |
|---|---|---|
| own | E≥200 | `platt / calibrated_platt`；拟合后若 b≤0，回退 `intercept / calibrated_intercept_fallback` |
| limited | 100≤E<200 | `intercept / calibrated_intercept`，b=1 |
| insufficient | E<100 | `none / not_calibrated`，不参加排名 |

启用校准必须保留非空独立 calibration 块且 `evidence_tiers.source=calibration`。方法由 tier 自动选择，当前配置 schema 没有手动指定 Platt/intercept 的开关。200/100 是可配置的工程阈值，不是数学上样本充分、必收敛或必校准准确的保证。两种方法都先裁剪输入；公式、拟合目标和失败边界见[下方数学说明](#calibration-math)。

关闭校准时所有 tier 的产品均为 `method=disabled, status=disabled`，不调用校准 fit；identity pickle 仅兼容加载，服务 calibrated 概率为 NULL，排名强制 raw。关闭开关不会改变时间分配。raw-retained 保留 calibration 作为 evidence-only 窗口并可报告其 raw 指标；raw-reallocated 显式 cal=0、gap1=0、source=inner_valid。启用校准时不允许 inner_valid evidence，任何模式都不允许 Test evidence。

Final Test 只生成一次性留出报告与既定放行门，不调参数、不挑模型/校准器。指标包括 AUROC、PR-AUC、log-loss、Brier、O/E、slope、lift@1/5/10%、reliability；按快照报告 O/E 与 slope；按 client 聚类 bootstrap（默认 200、95% CI）用于 gate 字段的 AUROC/O/E。实际校准时验证单产品排序单调。

只有实际存在的 score_kind 才有评价行。未校准（含 enabled 但 insufficient）的 calibrated 指标在 JSON 摘要为 `null`，控制台为 `N/A`，绝不以 raw 冒充；真实 calibrated 数值照常报告。无 calibration 块无该块的评价行，count=0、selection 的 calibration_end=NULL；某些 manifest 日期展示字段仍会字符串化空值，消费者应按字段语义读取。

放行门保持：AUROC≥0.60、lift@10%≥1.5、O/E∈[0.8,1.25]、slope∈[0.7,1.3]；非有限/缺失值不通过。按发布排名字段判门（calibrated 且存在则用它，否则 raw）。`ranking_eligible_flag = tier∈{own,limited} AND gate_passed AND status!=not_calibrated`；disabled 不绕过 insufficient 排除。

写 `ml.l1_evaluation`、`ml.model_selection`，后者含 evidence 来源/计数、calibration/test 计数、tier/status/gate、ranking_field、run IDs、配置哈希和 provenance。02 的 run ID 仍叫 `calibrator_run_id`，raw 模式下只标识评估/identity artifact run，不表示拟合过校准器。

发布 `artifacts/<release_id>/<l2>/{model.txt, feature_columns.json, calibrator.pkl, training_provenance.json, calibration_provenance.json}` 及 release manifest。已有 release 默认拒绝；02 虽保留显式 `allow_overwrite=true`，常规验收/配置切换必须使用新唯一 ID，不能用覆盖或改标签来迁就陈旧产物。

<a id="calibration-math"></a>
### 8.1 校准数学：从原始分数到概率，再到排序

#### 输入、odds、logit 与 sigmoid

设 `s` 是冻结 champion 对 eligible 行输出的原始分数，`y∈{0,1}` 是成熟购买标签。校准不是重训 LightGBM，而是对每个产品学习一个分数变换。原始分数与实际送入校准公式的值必须区分：

```text
ε = 10⁻⁶
q = clip(s, ε, 1−ε) = min(max(s, ε), 1−ε)
odds(q) = q / (1−q)
logit(q) = ln(q / (1−q))
sigmoid(z) = 1 / (1+exp(−z))
sigmoid(logit(q)) = q；logit(sigmoid(z)) = z（实数精确运算）
```

odds 是“发生概率 / 不发生概率”：概率 10% 对应 odds=1/9，50% 对应 odds=1。logit 把 `(0,1)` 映到实数轴，sigmoid 做逆变换。裁剪避免 0/1 的无限 logit；它不等于原始 `s`，也不修复缺失或无效输入。

#### intercept-only：固定 b=1，乘的是 odds

只学习截距 `a`，保留输入 log-odds 的斜率 1。由 sigmoid 的 odds 等于其自变量的指数，可逐步得到：

```text
z = a + logit(q)；p = sigmoid(z)
p / (1−p) = exp(z) = exp(a) · q / (1−q)
p = exp(a) · q / (1−q + exp(a) · q)
```

所以 `exp(a)` 乘的是 odds，不是概率，也不是给每个概率增加固定百分点。**以下所有参数和数字均为数学示例，不是已拟合模型参数或验收结果。** 取 `a=ln(2)`，odds 翻倍：10%→18.18%，50%→66.67%；概率既没有统一翻倍，也没有统一增加相同百分点。`a=0` 时返回 `q`，只有未被裁剪时才等于 `s`。

#### Platt：同时调整截距和 log-odds 的跨度

```text
z = a + b · logit(q)；p = sigmoid(z)
p / (1−p) = exp(a) · [q / (1−q)]ᵇ
```

`a` 改变基准 odds；在 `q=0.5` 时，输出为 `sigmoid(a)`，odds 为 `exp(a)`。`b` 改变 log-odds 上的跨度，任意两行有 `logit(p₁)−logit(p₂)=b·[logit(q₁)−logit(q₂)]`。预测公式中取 `b=1` 就得到 intercept-only；这不意味着两个实际拟合过程相同。

例如 `a=0, b=0.5` 时，10%→25%、50%→50%、90%→75%，向 50% 收缩。这一解释依赖本例的截距与斜率，不能泛化为任意 `a,b` 都使概率靠近 50%。实际 own 分支先做**未约束斜率**的拟合，再检查 `b>0`；`b≤0` 才在同一 calibration 块重新拟合 intercept-only，不是优化器内施加正斜率约束。

#### 参数如何学习：仅使用独立 calibration

令 `n` 为该产品 eligible calibration 的**拟合行数**，`pᵢ=sigmoid(zᵢ)`。`n` 包含正、负标签行，不是上表的 distinct evidence 事件数 `E`；一次购买可以影响多个快照标签。二元交叉熵（BCE，也称 log-loss）为：

```text
L = −(1/n) Σᵢ [yᵢ ln(pᵢ) + (1−yᵢ) ln(1−pᵢ)]
  =  (1/n) Σᵢ [ln(1+exp(zᵢ)) − yᵢ zᵢ]
每行的稳定算法：max(zᵢ,0) + ln(1+exp(−|zᵢ|)) − yᵢ zᵢ
```

稳定算法避免直接计算大 `exp(z)` 溢出；实际 intercept 实现用等价的 `logaddexp(0,z)−y*z`。它固定 `b=1`，在边界 `(-20,20)` 内数值最小化平均 BCE；不收敛或结果距任一边界小于 `1e-6` 时拒绝。应理解为带既定数值搜索范围的估计，而不是无条件存在的无界最优解。

实际 Platt 调用 `LogisticRegression(C=platt_C, solver="lbfgs", max_iter=1000, tol=1e-10)`，没有 sample weights 或 class weights。在已核对的本机 **scikit-learn 1.6.1** 中，未显式覆盖的默认值是 `penalty="l2"`、`fit_intercept=True`。单一输入特征 `logit(q)`、无权重时，其目标函数专门化为：

```text
J(a,b) = L(a,b) + b² / (2 n C)；截距 a 不受惩罚
默认 C = 1,000,000
```

默认正则很弱，但数学上不为零；源码/配置里“effectively unpenalised”的简写不等于精确无正则最大似然估计。该说明依据 [sklearn 1.6 官方二元目标函数](https://scikit-learn.org/1.6/modules/linear_model.html#binary-case)及 [LogisticRegression 参数文档](https://scikit-learn.org/1.6/modules/generated/sklearn.linear_model.LogisticRegression.html)，不是对未验证 Databricks Runtime 版本的保证；迁移时仍须核对实际依赖。

拟合输入仅为独立 calibration 上的冻结 champion 预测与标签；不在 Train 上拟合校准器，也不在 Final Test 上优化参数或挑选方法。两种 fit 均要求正、负标签同时存在，单一类别会报错，不会自动变成一次成功的 intercept 回退。缺失/无效标签或分数不属于合法输入；02 校验原始分数与标签，缺失所需块/行也会拒绝。常数或几乎常数分数缺乏区分信息，校准无法创造排序；裁剪、极端参数和数值饱和可能形成并列或使诊断退化。这里说明数学/实现边界，没有宣称这些情形均已做端到端验收。

#### 单产品的排序为何可能保持，跨产品为何可以改变

对已裁剪的内部输入 `q∈(0,1)` 求导。有限 `a`、有限正 `b` 下，精确实数运算有：

```text
dp/dq = [p(1−p)] · b · [1/(q(1−q))]
      = b p(1−p) / [q(1−q)] > 0
```

所以正斜率变换对 `q` 严格单调；intercept-only 的 `b=1` 也满足。实际计算还要求未数值饱和。**这不是对原始 `s` 的无条件严格单调保证**：不同 `s` 可被 clip 到同一端点，浮点 sigmoid 也可产生新并列。固定单产品的同一批评价行/标签，若变换前后所有两两大小关系及并列完全相同，则 AUROC 不变，因为 AUROC 只取决于正负例之间的胜负与并列；沿用相同平分处理时，top-fraction 顺序和选中集合也不变。新增并列时不能无条件声称原始 AUROC 或集合不变。02 的实际 Test Spearman 检查要求 `ρ≥1−1e-9`；常数序列的未定义相关等退化可能被拒绝，这个检查也不是概率准确性的证明。

客户在单一产品内排序，与同一客户在不同产品间排序，是两种问题。各产品有各自 `a,b`，因此各自保持客户顺序，仍能改变跨产品顺序。下例固定一个客户、四个均有排名资格的产品、相同 eligibility 和业务权重 `W=1`；所有 `b=1`，参数纯为说明：

| 示例产品 | 原始 s=q | 截距 a | 校准 p | raw 顺位 → calibrated 顺位 |
|---|---|---|---|---|
| A | 40% | ln(1/4) | 14.29% | 1 → 4 |
| B | 30% | ln(2) | 46.15% | 2 → 1 |
| C | 20% | 0 | 20.00% | 3 → 3 |
| D | 10% | ln(4) | 30.77% | 4 → 2 |

raw 顺序 A→B→C→D，校准后 B→D→C→A；Top-3 成员从 {A,B,C} 变为 {B,D,C}。例子只证明顺序/成员可以改变，不证明推荐收益或校准质量必然改善。跨产品 calibrated 概率是否准确、可比，仍需数据与目标口径一致及实际验证；单调性本身不能证明这些性质，raw 也没有这种保证。实际资格、gate、业务权重和平分规则仍按本节及 §10，不因上述数学结论放宽。

实现对应：[modeling.py](../notebooks/modeling.py) 的 `clipped_logit`、两种 calibrator、`select_calibrator`；[02](../notebooks/02_calibrate_and_validate.py) 的 `fit_calibrator`、Spearman/gate；[配置校验](../notebooks/l1_contract.py)；[默认配置](../conf/training.yaml)。可离线阅读的同义图解见 [图 3 数学说明](modeling_flow.html#calibration-math)，英文见 [Genie 数学说明](GENIE_HANDOVER.md#calibration-math)。

**覆盖边界**：最近的 10% 合成 calibrated 验收是三个 limited/intercept 产品、其余 insufficient，raw 路径已测试；当前 full-volume own/Platt 端到端未运行。§14 的全量数字仍属历史记录。本节示例的数值校验只是文档检查，不是新模型、流水线或 Databricks 验收。

## 9. Notebook 03：Layer 2 与 reasons

eligible 购买事件按 `(client_id, feature_snapshot_date=snapshot_date)` join 快照特征。`conf/layer2.yaml` 独立控制申请日期 Train/Test 和 inner_valid_months：当前 Train≤2026-01-31、Test 为 2026-02-01..07-31，Train 尾部 3 个申请月作验证；购买时属性已知，不加 L1 的 horizon gap。

`sp_rp / par_nonpar / buying_for` 为 pooled LightGBM head（l2 作 categorical）；某产品取值样本不足回退 Train 历史占比，单一有效值为 fixed。`mass_higher` 由 wealth_seg3 规则给出。validity mask、重新归一化、margin abstain 保持原策略；buying_for 的 `constraints[].on_missing` 为 allow/block，当前配置明确 allow。

Layer 2 以自己的 pooled Train 购买行拟合类别字典，保存 `layer2_spec.base_feature_spec`，04 按同一规格使用，不能借用某个产品的 L1 类别子集。L1 解释仍以对应 champion spec 做 `pred_contrib`，按 reason group 汇总，推荐理由只取正贡献且非 sensitive、可 explainable 的前 N 组（默认 3）。此前修正 3 的空/未知类别安全展示保持有效。

写 `ml.l2_evaluation`、`ml.layer2_models`，MLflow `l2_training` 和文件包 `layer2/`。03 对已选 release 增量添加 Layer 2，保留原 L1 契约；04 校验 Layer 2 YAML 与 saved spec 一致。每个属性独立指导，不声称联合组合最优。

<a id="layer2-math"></a>
### 9.1 Layer 2 数学：购买条件下的属性分布

#### 目标与 head 的训练人口

Layer 1 回答“会不会买这个产品”，Layer 2 使用 eligible **购买事件**与对应快照特征，学习 `P(A=v | 已购买产品 ℓ, x, eligible)` 这种购买者条件下的属性选择模式。未购买行不是 Layer 2 的负例；这个分布不是又一份购买概率，也不估计 `P(购买且选择某配置)`。把购买者中的相关模式用于当前客户指导，不等于预测改变配置后的购买收益。

`sp_rp / par_nonpar / buying_for` 各有一个跨产品 pooled LightGBM head，输入原特征加 categorical `l2`。配置顺序明确为 `sp_rp=[SP,RP]`、`par_nonpar=[Par,NonPar]`、`buying_for=[self,spouse,child,parent]`。二分类的正类是**第二个配置值**，即 RP 或 NonPar；模型输出 r 后组装 `[1−r,r]`。前两者用 binary BCE，四值 buying_for 使用 multiclass softmax 与类别交叉熵：

```text
二值：r = sigmoid(F(x,ℓ))；L = −mean[y ln(r)+(1−y) ln(1−r)]
多值：r_v = exp(F_v) / Σ_u exp(F_u)
      L = −(1/n) Σᵢ Σ_v 1[Aᵢ=v] ln(rᵢ,v)
```

训练池包含该属性有至少两个有效值的所有 L2 的有效 Train 购买行，即使某 L2 最终使用 share，仍可为 pooled head 提供训练行。head 在这个训练池最后 **3 个实际观测申请月**上 early-stop，再以选定轮数在完整 pooled Train 重拟合。Layer 2 类别字典来自自己的 pooled Train；它有独立的申请日期 Train/Test（当前 Train≤2026-01-31，Test 2026-02-01..07-31），不复制 L1 的留出窗口，也没有实现 Layer 2 calibrator。

#### 逐产品决定 head、share、fixed、rule

服务计划再按 L2 检查该属性的有效值：单一值为 fixed；多个值时，head 需**每个有效值**至少 200 条有效 Train 购买行，否则使用 Laplace 历史占比。这里 `min_rows_per_value=200` 是属性样本条件，与 L1 校准 own tier 的 200 个 distinct evidence 事件是两回事。若产品无有效值，则为 not_applicable。

对 K 个有效值及计数 n_v，share 使用 `r_v=(n_v+1)/(Σ有效值 n_u+K)`，无效值为 0。这是加一平滑计数，不训练一个优化器。例如有效值计数 `[8,1,0]` 时得到 `[9/12,2/12,1/12]=[0.750000,0.166667,0.083333]`，零计数也有小的非零占比。fixed 给唯一值概率 1；mass_higher 的多值 rule 对 `wealth_seg3∈{UHNW,HNW,Affluent}` 给 HigherEnd，否则给 Mass，**缺失也走 Mass**。rule/fixed/share 都没有学习损失最小化步骤；它们仍要经过后面的有效性约束。

#### 先 mask 再归一化，最后决定是否 abstain

每个产品的属性有效值由 validity 表按属性投影得到（未提供该属性列时使用配置值全集）；同时按客户特征施加 buying_for 约束：spouse 要 Married、parent 要年龄<50、child 要年龄>25。`on_missing` 决定缺失输入 allow/block，当前三条均明确 allow，含义是未知不据此排除，不表示条件已被事实验证。

令 m_v 表示产品有效性与客户约束都允许该值。对有限概率，服务变换是：

```text
r′_v = m_v r_v / Σ_u(m_u r_u)
margin = 最大 r′ − 第二大 r′
```

实际实现用 `np.where(mask, r, 0)` 清零后才求和：被 mask 排除的 NaN 可以被移除，并非输入任意一项 NaN 都必定使整行无效。**保留下来的总量**为 0 或非有限时输出 not_applicable；总量可用时归一化。head/share 的 margin<0.20 才 abstain；相等的 0.20 按规则通过（浮点计算按实际存储值比较），fixed/rule 不受 margin abstain。只剩一个非零值时归一化为 1、其余为 0，margin=1；并列最大值先按配置值顺序选取，再判断 abstain。

例如 buying_for 的原概率按配置顺序为 `[0.45,0.35,0.15,0.05]`，当前客户允许 self/child、不允许 spouse/parent，mask=`[1,0,1,0]`。保留总量为 0.60，归一化为 `[0.75,0,0.25,0]`，margin=0.50，输出 self。另一个有限分布 `[0.45,0.40,0.10,0.05]` 若全部允许，margin=0.05，head/share 应 abstain；前述三值 share 示例若均允许，margin=0.750000−0.166667=0.583333，通过。这些都是纯数学示例，不是实测 head 概率。

当前按**每个属性独立**给指导；把各 head 的概率直接相乘不等于学到了有效联合配置，更不能证明该组合最优。validity 的属性投影也不是联合组合求解器。04 在产品 Top-3 确定后才计算指导，不据此重排产品。属性选择、购买概率和因果收益应分别解读。

实现对应：[03](../notebooks/03_layer2_and_reasons.py) 的 `l2_plan / train_head`、[modeling.py](../notebooks/modeling.py) 的 `class_probabilities / restrict_probabilities / layer2_guidance`、[layer2.yaml](../conf/layer2.yaml)、[validity](../conf/l2_config_validity.csv)。同义入口：[英文](GENIE_HANDOVER.md#layer2-math)、[离线图解](modeling_flow.html#layer2-math)。

<a id="shap-math"></a>
### 9.2 SHAP 与 reasons：解释的是哪个输出

#### 实际调用、baseline 与单位

当前 `modeling.feature_contributions` 调用的是原生 `LightGBM Booster.predict(X, pred_contrib=True)`。二分类返回 n×(d+1)：前 d 列对应原始模型特征贡献，末列是 Booster 返回的 bias。helper 检查形状后只把前 d 列交给 reasons；bias 仍是理解完整分解所必需的基准。这里没有构造 `shap.TreeExplainer`、用户 background DataFrame、interventional background、probability/logloss 的 model_output，或 Layer 2 explainer。

原生 TreeSHAP 使用模型树中保存的路径/计数信息处理特征贡献，依据 [LightGBM 4.6 Booster.predict 文档](https://lightgbm.readthedocs.io/en/v4.6.0/pythonapi/lightgbm.Booster.html#lightgbm.Booster.predict)与 [官方 tree.h 实现](https://github.com/lightgbm-org/LightGBM/blob/v4.6.0/include/LightGBM/tree.h)。不能把没有传入的“总体平均客户”、当前打分批次或用户选定客户群说成这个调用的 background。末列 φ₀ 是该模型的 **raw-output baseline**，不是平均购买概率，也不是约定为 0。

对当前默认 sigmoid=1 的 L1 二分类 champion，完整加法分解在 raw log-odds（margin）尺度成立：

```text
Fℓ(x) = φ₀,ℓ + Σⱼ φⱼ,ℓ(x)
sℓ(x) = sigmoid(Fℓ(x))
```

正贡献把 margin 向上推，负贡献把它向下推。纯数学瀑布例子为 `−2.0 → +0.6 → +0.3 → −0.1`：`F=−2+0.6+0.3−0.1=−1.2`，`s=sigmoid(−1.2)=23.15%`。其中 +0.3 是 log-odds 贡献，绝非 +30 个百分点；sigmoid 非线性意味着概率增量依赖已有 margin，不能逐项 sigmoid 后相加。各产品有各自模型和 bias，跨产品 SHAP 数字不能当成概率收益比较。

解释输入严格沿用对应 champion 的 `feature_columns` 顺序、已保存类别水平与 float32 数值；未知类别为 NaN。贡献对应 LightGBM 的原始输入特征，不是 logistic baseline 的 one-hot 列。03 既有 smoke 从 Test 购买行取最多 100 行，检查“特征贡献和+bias”对 `raw_score=True` 的最大误差<1e-6；这里只描述已有实现与既有流水线覆盖，本次没有再跑 smoke/训练。

#### 从完整贡献到可展示理由

`reason_group` 的聚合是 `Φ_g=Σⱼ∈g φⱼ`。推荐理由只保留 `Φ_g>0`、explainable 且非 sensitive 的组，降序取 top_n_reasons（默认 3），同分按配置组顺序。正贡献组的代表特征取组内最大正方向贡献；完整审计中负组则取最负方向特征。代表特征仅用于展示，不能代替全组和。

审计表可保留所有组及 sensitive/explainable 标志，可选记录 feature contributions；对外正向前三理由省略 bias、负组和被过滤组，**不是完整可加分解**。没有展示某负组不表示该特征未被模型使用；分组只是业务展示映射，不是特征独立性或理论独立性的保证，也不授权展示原本被过滤的敏感属性。

#### 校准之后，这些贡献还能说明什么

当前返回的解释只分解**校准前的 L1 champion**。它不分解 calibrated probability、business_weight、eligibility/gate、跨产品 rank、Layer 2 head/rule 或属性选择；不能把 reason_1..3 描述为完整的“为何 Top-3”证明。校准保留单产品内部客户次序，也不能补上这些未解释环节，而且各产品的变换不同，可能改变客户的跨产品次序。

若完全没有 clipping 改变输入且没有数值饱和，可作下列代数推导：

```text
logit(p) = a+bF = (a+bφ₀) + Σⱼ bφⱼ
```

这是 logit 尺度的仿射变换，**未实现为当前解释输出**；不能据此宣称已计算 calibrated-probability SHAP。再经 sigmoid 后，特征的概率贡献不满足这种逐项相加关系；发生 clip 时也不能直接把 logit(q) 替换成 F。

SHAP 在既定模型与贡献分配假设下解释关联和模型归因，不证明因果、行动建议、改变一个特征就能获得相应收益，或反事实保证。相关特征及 background/路径分配假设会影响功劳如何分摊；此实现没有提供特征独立、因果识别或公平性保证。

实现对应：[modeling.py](../notebooks/modeling.py) 的 `feature_contributions / group_contributions / explain_rows`、[03](../notebooks/03_layer2_and_reasons.py) 的加法检查、[04](../notebooks/04_batch_scoring.py)、[reason_groups.yaml](../conf/reason_groups.yaml)、[feature_groups.csv](../conf/feature_groups.csv)。同义入口：[英文](GENIE_HANDOVER.md#shap-math)、[离线图解](modeling_flow.html#shap-math)。

## 10. Notebook 04：服务、固定输入与提交

按 `env.model_source: mlflow|files` 加载同一 champion/calibrator/feature specs/provenance 和 Layer 2；两条路径均验证模型、校准器哈希及原 run 一致性。无 provenance、配置不符、混用 tuning/champion、spec/artifact 不符会拒绝，需重新按正确配置训练并发布新 release。

在读取任何业务输入前，逐表捕获 `ml.modeling_table`、`ml.model_selection`、`ml.layer2_models` 的 Delta version，再按 `versionAsOf` 读取，manifest 记同一版本；捕获或读取失败不能回落 latest。**当前输入标识为逻辑名→物理表名 + delta_version；没有单独 Delta table UUID，也没有跨表原子快照保证。** 01/02 另保存其训练/校准输入 pins；不能把模型 run ID 当表版本，也不宣称 03 全部读取均已有同样 pin 机制。此前修正 4 保持有效。

打分快照为≤`run_date` 且在 `lookback_days=35` 内的最近日期。eligible 产品计算 raw；实际已校准才计算 calibrated。`elig=0` 的两个概率和 rank 均 NULL，原因 `not_eligible`。产品不获排名资格则 rank=NULL，原因 `not_ranking_eligible`。

发布级 `ranking_field` 全表一致；`ranking_score=该字段×business_weight`。只在客户的 eligible 排名集合中按分数降序，平分按 l2 名字典序，rank 从 1 连续唯一。默认 Top-3，合资格项少则不足 3。raw 为分别训练模型的 propensity，跨产品用 raw 排名是一项策略，不保证概率可比或已校准。

| 输出表 | 粒度 / 关键内容 |
|---|---|
| `scores.customer_l2_scores` | run_date × client × L2；概率、tier/gate、evidence/calibration 计数、排名、指导、reasons、model/calibrator/layer2/release 版本、training_config_hash 和 model_role；spec/model 哈希与迭代等详细 provenance 在 manifest 中 |
| `scores.customer_top_recommendations` | 上表 rank≤top_k 子集 |
| `scores.customer_reason_contributions` | Top-k × reason group；含 sensitive/explainable 标志，审计可含被推荐文本过滤的组；可选 feature_contributions |
| `scores.scoring_runs` | 每 run_date 的提交标记及 manifest 摘要 |

Top-k 才生成 Layer 2 和 reasons。输出按 run_date `replaceWhere`，writer 级 mergeSchema，分区列最后；下游按列名取值。04 **先删旧 scoring_runs → 写三张数据表并核对 → 最后写 scoring_runs**；中途失败该日期不算提交，下游必须 join 提交标记。manifest 记录 run_date/snapshot、release、score 来源、表 pins、模型 provenance、实际字段/权重、staleness、行数和 rank 检查。

文件包 staging→rename swap 含中断恢复；同 release 并发写不支持。02 的 Delta selection 先于文件包写入，失败可能部分发布，files loader 会拒绝不一致包；跨表和文件系统不构成原子事务。UC Volume rename 语义仍待真实环境验证。

## 11. Databricks 迁移

`resources/mvp.job.yml` 只含 01→02→03→04，job 级参数为 `training_config`（默认 `conf/training.yaml`）、`env`（`${bundle.target}`）、`release_id`（`mvp-{{job.run_id}}`）、`run_date`（空→当天）、`l2_list`（pilot）。不混用 task base_parameters；job 当前没有 sample_fraction 参数。03/04 在 Databricks 必须有 release_id。

上游产出合契约的真实 `ml.modeling_table / ml.data_dictionary / raw.l2_purchase_config`，并按相同完整 training YAML 生成 split。01 全表验证，00 仅合成准备。迁移时确认 catalog/映射、runtime/node type、MLflow 路径、Volume 路径和依赖版本；当前文件仍有 MIGRATION 占位。

远端 `databricks bundle validate` 尚未通过，部署未执行。历史本地 `bundle schema` 的 0 错误只证明 schema 检查，不代替远端验证。迁移清单见 [GENIE_HANDOVER](GENIE_HANDOVER.md)。

## 12. 验证与当前验收

开发机保留 regression/focused 测试和验收日志，迁移包不包含测试、数据或个人审计记录。写入型验证应在隔离环境和独立 release 上执行。

已有验证：修正 1/3/4 的 34 项回归；时间/模型契约修正后 68 项 broad 检查通过；报告修正后 39 项 focused 检查通过（含新增 5 项 reporting），独立复审重跑这 5 项并接受。不是一次最终 73 项完整 suite 的运行，也不代表本次迁移包装重新运行了这些检查。

两个隔离 10% 合成客户、6 pilot 流程完成 calibrated 与 raw-reallocated 00→04；每种模式的 64,896 行中 `client_id,l2,raw_propensity_score,calibrated_purchase_probability,rank` 精确一致，保存模型/spec 的抽样预测最大差为 0。calibrated 的 01/02 使用较早源码快照，后来共享代码修正由 raw 流程、focused 检查及最终 calibrated 04/保存 artifact 重现覆盖，不能声称两个流程均在最终源码重新跑完。报告修正后未再训全流程或重跑此前 34 项 Spark suite。

raw-retained 全流程、train cap / tuning_fit refit 独立全流程、当前全量及新的 own/Platt 验收、远端 Databricks 尚未执行。历史 Ruff 有 51 项既有诊断，业务修正新增 0；不能称全项目 lint clean。逐表 pins、文件发布与 Volume rename 等边界仍以 §10–11 为准。

## 13. 明确推迟

Point 5 生产效果评价器、monitoring/drift、滚动校准和 rolling OOF 均未实现。未来评价需匹配当时保存的预测与成熟标签；当前图解不再把未来滚动方案当成已有 pipeline。

另推迟 isotonic、Optuna、UC registry/alias、全成熟数据最终 refit、季节特征、联合配置指导、option B 下采样、Hit@k 跨 L2 基线对比。这里“最终 refit”指评估后再把保留窗口并入训练，区别于 01 已实现的 Train 内 champion refit。`calibration.isotonic` 和 `final_refit_calibrator` 当前只能为 false，启用会报错。

## 14. 历史质检与 2026-10-03 release 基线

以下原始数字保留供追溯，不作为当前未提交修正的全量重验声明；当前事件语义含 §5 的稀疏包络限制。


LightGBM（40% 客户训练，Test 2026-02..04）：AUROC product1 0.750 / product5 0.693 / product8 0.752 / product17 0.692（CI 0.57–0.81）；lift@10% 3.2 / 2.5 / 3.7；slope 0.96–1.02；O/E 0.98 / 0.94 / 0.88。真值上限 0.81 / 0.71 / 0.83。17 个 L2 全量训练 < 5 分钟。

Calibration 块 distinct 事件数有两种口径，**pipeline 用 §5 口径**（`application_date ∈ (块首, 块尾 + H]`，即校准器实际看到的事件；2026-10-03 决定 A）：p1 2016、p5 1305、p11 1590、p8 832、p16 337、p14 72。质检阶段按 `feature_snapshot_date ∈ 块` 数的是另一口径（约为前者的 3/5）：p1 1197、p5 818、p11 1006、p8 493、p16 190、p13 183、p7 166、p14 49、p4 40、p17 21、p10 10——只作参考，不用于 tier。

第 ② 批验收实测（release `mvp-acceptance`，6 个 pilot 全量，Test 块）：AUROC p1 0.757 / p5 0.701 / p8 0.761 / p11 0.804 / p16 0.725 / p14 0.743（CI 0.68–0.80）；Platt 斜率 0.96–1.04；O/E（calibrated）0.93–1.04；p14 insufficient 不参与排名。01 约 195 s，02 约 77 s。

<a id="configuration-usage"></a>
## 15. 完整配置与可执行参数

三份文件：[calibrated](../conf/training.yaml)、[raw-retained](../conf/training.raw-retained.yaml)、[raw-reallocated](../conf/training.raw-reallocated.yaml)。它们都是完整 YAML，校验器拒绝未知/缺失 policy 键及非法组合。不开启 calibration 不会重分配；显式零 calibration 必须同时设 gap1=0、disabled 和 inner_valid evidence。

Databricks 使用现有 job 参数 `training_config=conf/training.raw-retained.yaml`（或另外两份文件），通过 widgets 到全部任务；同时设置 `env=dev`、适用的 `run_date`、`l2_list` 和新唯一 `release_id`。交互执行也必须设置 `env=dev`，因为 notebook 的默认 env 仍为 local。配置相对路径以仓库根目录解析。

该参数不是 YAML overlay，也不负责重写真实上游 split。更换 YAML 后，上游准备、01→04 和发布必须一致；仅服务既有兼容 release 时，04 也须选其原完整 YAML。迁移包不提供本地 00 或 PowerShell runner，运行库、工作区占位和上游接入步骤见 [README](../README.md)。远端验证、部署及真实数据验收尚待迁移阶段实施。
