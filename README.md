# Prod_Recom — 产品推荐引擎 MVP

港险客户 × 17 个产品类别（L2）：按产品训练购买倾向模型，生成配置指导、解释和批量 Top-k 推荐。当前是 `lean-v0` 的 notebook + YAML 实现；本地合成数据已验收，Databricks 真实接入尚待执行。

此仓库工作树仅包含 Databricks 迁移所需的 01→04、共享模块、完整配置与文档。数据、模型产物、本地环境/runner、测试、参考原件及个人工作记录不在迁移包内；上述范围仅描述当前迁移目录；历史版本的清理需单独核验。

| 入口 | 用途 |
|---|---|
| [MVP_SPEC](docs/MVP_SPEC.md) | 当前业务规格、时间/模型/发布契约及已有验证边界 |
| [流程图](docs/modeling_flow.html) | 可离线浏览的四段中文流程 |
| [GENIE_HANDOVER](docs/GENIE_HANDOVER.md) | Databricks 上游、表、参数与迁移步骤（英文） |
| [环境模板](conf/env/dev.yaml) | catalog、表名映射、MLflow experiment 和 UC Volume |
| [Bundle](databricks.yml) / [Job](resources/mvp.job.yml) | dev 目标与顺序执行的 01→04 作业 |

数学阅读入口：[Layer 1：目标、训练与指标](docs/MVP_SPEC.md#layer1-math) · [校准：公式与排序](docs/MVP_SPEC.md#calibration-math) · [Layer 2：条件属性与约束](docs/MVP_SPEC.md#layer2-math) · [SHAP：贡献单位与解释边界](docs/MVP_SPEC.md#shap-math)。

[离线 Layer 1](docs/modeling_flow.html#layer1-math) · [离线校准](docs/modeling_flow.html#calibration-math) · [离线 Layer 2](docs/modeling_flow.html#layer2-math) · [离线 SHAP](docs/modeling_flow.html#shap-math) 保留四段流程，并提供可复算示例；这些例子不是新模型验收成绩。

## Databricks 迁移准备

1. 保持仓库目录结构导入 Workspace/Git folder 或通过 Asset Bundle 同步；`notebooks/` 与 `conf/` 必须有同一根目录。共享 `.py` 文件必须作为 Workspace 文件可导入，不能只复制四个 notebook。无需本机 `.venv`、JDK/Hadoop 或 PowerShell 脚本。
2. 在 `databricks.yml` 填目标 host，在 `resources/mvp.job.yml` 确认工作区支持的精确 DBR 18 LTS ML 版本、node type 和 driver 容量。当前单节点、driver 上 pandas/LightGBM 拟合方式不是生产容量保证。
3. 填 `conf/env/dev.yaml` 的 catalog、必要 `table_overrides`、已创建的 UC Volume 路径及 experiment root；确认执行身份具备输入读取、输出表写入、实验和 Volume 访问权限。保持 `allow_synthetic_data: false`。
4. 上游在 01 前提供 `ml.modeling_table`、`ml.data_dictionary`、`raw.l2_purchase_config`，并按选定完整 YAML 生成相符的 split。143 个登记特征见 `conf/feature_groups.csv`，其中 142 个用于模型；字典表由真实上游提供，原始参考 Excel 不是运行依赖。
   当前字段采用通用客户门户和线索命名（如 `customer_portal_activated`、`portal_product_page_12m`、`n_leads_12m`）；以 CSV 为完整字段契约，上游建模表列名和字典 `column_name` 必须同步一致。旧字段 schema 的模型、feature spec、校准器及 release 不可直接复用；须按新契约重新训练、评估并生成新 release，本地已有产物不在此步骤中修改。
5. 三份 training YAML 的 extraction/as-of 默认 **2026-10-01**，Layer 2 的训练/测试日期在 `conf/layer2.yaml` 独立设置。迁移时共同确认日期、实际历史长度与成熟标签，不能把固定示例日期直接当新的生产窗口。
6. 在已配置且获准的目标环境运行 `databricks bundle validate -t dev`，通过后再按已批准流程部署和真实数据验收。本地 YAML/schema 检查不能替代远端验证；此仓库未证明真实环境已就绪。

作业参数为 `env=dev`、`training_config`、`l2_list=pilot|all|逗号列表`、`run_date` 和唯一 `release_id`。默认 release 为 `mvp-{{job.run_id}}`，`run_date` 为空时为当天，需有最近 35 天内的快照。三个配置为完整文件而非 overlay，01→04 必须使用同一份配置和 release。交互执行时显式设置 `env=dev` widget：notebook 本身仍保留本地默认值。当前 job 没有 `sample_fraction` 参数。

### 从 GitHub 到 dev 作业

**首次准备**

在获准的位置取得私有仓库的 `main` 分支（需要已有仓库访问权限）：

```shell
git clone --branch main https://github.com/sflszyt/project1.git
cd project1
```

也可在 Workspace/Git folder 导入同一仓库和分支；下列 CLI 命令均须在能运行 Databricks CLI、包含 `databricks.yml` 的仓库根目录执行。
使用已批准、已配置的 CLI 认证，并确认对应目标工作区；如需指定已有 profile，在每条 Databricks 命令末添加 `-p "<已配置的 profile 名>"`，替换占位符。凭据留在 Git 外。
按上文完成模板、依赖、上游数据及日期检查；部署/执行身份还须按目标环境访问策略具备同步 Workspace 文件、创建/更新 job、创建及使用所配置集群的权限。UC、实验和 Volume 访问仍按上文核对。

**后续验证与部署**

配置或源码更新后，先执行第一条；只有验证通过并符合已批准部署流程时才执行第二条：

```shell
databricks bundle validate -t dev
databricks bundle deploy -t dev
```

`deploy` 同步源码并创建/更新 job；它不会创建 UC catalog、schema、Volume 或上游特征表，也不会授予访问权限。部署本身不启动 01→04。

**可选：实际运行**

完成目标环境检查、数据就绪确认及适用流程要求的审批后，以下命令才会启动实际的 01→04 训练和打分，并写入模型与结果；参数及默认日期见上文：

```shell
databricks bundle run -t dev prodrec_mvp
```

## 运行库要求

使用包含 Spark/Delta、Databricks widgets/DBUtils 和 Workspace MLflow tracking 的 **ML Runtime**。以下 Python 包必须能在四个任务的同一环境导入；表中版本仅是已使用的本地参考，不是目标 DBR 的锁定文件，也不保证其他版本兼容。

| 包 / import | 本地参考版本 | 用途 |
|---|---|---|
| PyYAML / `yaml` | 6.0.2 | 全部 YAML 配置 |
| LightGBM / `lightgbm` | 4.6.0 | L1/L2 训练、文件模型、pred_contrib |
| NumPy / `numpy` | 2.1.3 | 数值计算 |
| pandas | 2.2.3 | driver 数据处理 |
| SciPy / `scipy` | 1.15.3 | 优化、校准和统计 |
| scikit-learn / `sklearn` | 1.6.1 | baseline、预处理与指标 |
| MLflow / `mlflow` | 3.8.1 | 实验、模型和 artifact API |
| PyArrow / `pyarrow` | 21.0.0 | Spark↔pandas Arrow 转换 |
| PySpark / `pyspark` | 4.1.1 | 本地参考；Databricks 必须沿用 Runtime 自带 Spark/Delta |
| IPython | Runtime 提供 | 获取 notebook 的 dbutils；另有 DBUtils fallback |
| Databricks SDK / `databricks-sdk` | 与目标 Runtime 兼容 | MLflow 实验目录创建的条件分支（`WorkspaceClient`） |

在选定 DBR 上先核对包版本与导入；若缺包，通过已批准的 cluster/job library 管理补齐，保持任务间版本一致。不要为复现本地版本覆盖 Runtime 自带的 PySpark/Delta。MLflow 3 API、scikit-learn `OneHotEncoder(sparse_output=...)`、LightGBM callback/模型加载，以及 Volume 文件发布均须在目标环境验收。模型/校准器的 Python pickle 依赖相同共享模块和兼容包版本。

## 已实现的流程

1. **输入**：本地 00 仅将已有合成快照准备成 `ml.modeling_table` / `ml.data_dictionary`，不生成真实数据。Databricks 上游须在 01 前提供这两张表及 `raw.l2_purchase_config`；00 不进 job。
2. **01**：先全表 preflight，再按 `elig=1` 训练。tuning 模型在 inner_valid 选 `best_iteration`；独立 champion 按 `refit_scope` 重拟合。inner_valid 导出 tuning 预测，calibration/test 导出 champion 预测。
3. **02**：按选定 evidence 窗口数 distinct 购买事件，确定 tier；启用校准时仅用保留的 calibration 块拟合。Final Test 只报告和放行，不选择模型或证据。缺失 calibrated 指标显示 JSON `null` / 控制台 `N/A`。
4. **03**：购买行上的 pooled Layer 2 head / rule / share / fixed，保留 validity、缺失约束及 margin abstain；保存自身训练类别字典。解释使用各 L1 champion 的特征规格。
5. **04**：固定三个输入表的 Delta 版本，加载一致 release，取 `run_date` 前 35 天内最近快照。仅合资格行按发布级字段 × 业务权重排序，Top-k 生成指导/解释；最后写 `scores.scoring_runs` 提交标记，下游须 join 它。

## 三种配置与时间

默认 extraction 为 **2026-10-01**，模式为 **retrospective_holdout**。以现有 36 个连续月末快照推导：

| 完整 YAML | Train / tuning fit / inner_valid | calibration 分配 | evidence 来源 | Final Test |
|---|---|---|---|---|
| `conf/training.yaml` | 19 / 13 / 3 | 3，拟合校准器 | calibration | 3 |
| `conf/training.raw-retained.yaml` | 19 / 13 / 3 | 3，只保留 evidence/诊断用途 | calibration | 3 |
| `conf/training.raw-reallocated.yaml` | 25 / 19 / 3 | 显式 0；前置 gap 显式 0 | inner_valid | 3 |

关闭校准不会自动挪用窗口。三个示例的 evidence 日期恰同为 2025-08-31..2025-10-31，数量可以相同；其他分配可能改变资格。raw 跨产品排名是业务策略，不能把分别训练的 raw propensity 当成已保证可比的校准购买概率。

标签窗口不重叠（前块末 + H ≤ 后块首）与标签可用（快照直接 + H+m ≤ 阶段 as_of）分别检查；H=3、m=2，按日历月一次裁剪。当前回顾性留出没有保证模型在历史预测起点已可部署，也没有统一强加 5 个月 gap。完整日期、参数和例子见 [规格 §5](docs/MVP_SPEC.md#timeline)。

## 目录与表

`notebooks/01..04` 为迁移入口；`common.py` 负责平台/输入验证，`l1_contract.py` 负责配置、日期及 L1 artifact 契约，`modeling.py` 负责模型和解释函数。`conf/` 含三个完整 training 配置、环境、产品、Layer 2 和 feature/reason 配置。`resources/mvp.job.yml` 定义 Databricks **01→04**。本地合成准备器、测试与 runner 保留在开发机，不包含在迁移工作树中。

| 表组 | 逻辑表 |
|---|---|
| 合成准备源 / 购买事件 | `raw.client_snapshot_monthly`、`raw.l2_purchase_config` |
| 上游建模契约 | `ml.modeling_table`、`ml.data_dictionary` |
| L1 | `ml.l1_raw_scores`、`ml.l1_training_runs`、`ml.l1_evaluation`、`ml.model_selection` |
| L2 | `ml.l2_evaluation`、`ml.layer2_models` |
| 服务输出 | `scores.customer_l2_scores`、`scores.customer_top_recommendations`、`scores.customer_reason_contributions`、`scores.scoring_runs` |

模型/校准器 run ID、feature spec、配置和文件哈希与输入表版本分别追踪；不兼容或无 provenance 的旧 release 会被拒绝。逐表 pins 不是跨表原子事务；当前表标识是物理表名，不是 Delta UUID。

生产效果评价器（Point 5）、monitoring/drift、滚动校准/OOF、最终全成熟数据 refit、isotonic、Optuna、UC alias、季节特征、联合配置指导仍推迟。远端 `bundle validate` 和部署未通过或执行，不能用本地 schema 检查替代。
