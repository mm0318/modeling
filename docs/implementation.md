# Implementation — Same-runtime MoE Component Modeling

> Extended: 本项目从现有 profiling 工作区中独立出来，只复用设计经验与可审计数据格式，不修改 vLLM、LLMServingSim 或原 `code/`。

## Scope

第一版实现一个离线建模与预测工具。输入模型、硬件、workload、routing 和 runtime config，输出
单个 scheduler step 的 Attention、Expert、Dense、Other、总延迟、显存与瓶颈归因；再通过
scheduler trace replay 生成 TTFT、TPOT 和吞吐。

V3 主路径是只针对 MoE 模型的细粒度白盒 timing oracle 与多资源事件引擎。旧四组件模型保持 artifact
兼容，仅作为 baseline。V3 输入为 realized `StepExecutionPlan` 或 runtime interception events，输出
Attention、Router、Dispatch、Expert GEMM、Combine、Dense、Transition、Fixed、communication、
critical path、step、memory、逐层明细和瓶颈占比。主模型不训练 routing correction、Other 或
total residual。

正式 timing artifact 当前只覆盖 H20、Qwen3-30B-A3B、TP=1、EP=1。组件训练记录必须来自同一 vLLM
runtime 的 CUDA Event。事件引擎可表达 TP/EP>1，但缺少正式 collective artifact 时只允许
`synthetic_fixture` 验证，不回退到旧 Transformers eager 数据或冒充 H20 多卡实测。

## Directory Tree

```text
moe-component-modeling/
├── docs/
│   ├── implementation.md
│   ├── user_requirements.md
│   └── dev_log.md
├── code/
│   ├── configs/qwen3_30b_a3b_h20.json
│   ├── examples/
│   │   ├── training_records.jsonl
│   │   ├── predict_config.json
│   │   └── replay_trace.jsonl
│   ├── moe_perf_model/
│   │   ├── __init__.py
│   │   ├── __main__.py
│   │   ├── schema.py
│   │   ├── response.py
│   │   ├── model.py
│   │   ├── replay.py
│   │   ├── collector.py
│   │   ├── worker_collector.py
│   │   ├── operator_db.py
│   │   ├── execution_plan.py
│   │   ├── whitebox_model.py
│   │   ├── event_engine.py
│   │   ├── collective_model.py
│   │   ├── runtime_frontend.py
│   │   ├── evaluation.py
│   │   └── cli.py
│   ├── tests/
│   │   ├── test_schema.py
│   │   ├── test_model.py
│   │   └── test_replay.py
│   ├── notebooks/
│   ├── README.md
│   └── requirements.txt
```

## File Responsibilities

| File | Responsibility | Input | Output |
|---|---|---|---|
| `schema.py` | dataclass schema、加载、跨字段校验、stable signature | JSON object | typed config/record |
| `response.py` | 分组 response table、range-normalized IDW、分段一维插值 | component samples | latency + extrapolation metadata |
| `model.py` | 训练组件模型、routing correction、Other/residual、memory、Basic RF | training JSONL | serializable artifact / performance profile |
| `replay.py` | 串行 scheduler-step replay 与逐请求指标 | step JSONL + predictor | TTFT/TPOT/throughput |
| `collector.py` | 启动原生 vLLM 离线引擎、提交规则请求并导出 records | collection manifest | raw JSON + training JSONL |
| `worker_collector.py` | 在 worker 内安装可撤销的 instance hook，采集真实 step/组件 CUDA Event 与 routing | vLLM worker | per-step raw record |
| `operator_db.py` | operator signature、去重 median、分段/局部 response 与 cache | operator samples | response database |
| `execution_plan.py` | realized step、逐请求 shape、逐层 routing 与执行次数 schema | fine-grained raw/JSON | `StepExecutionPlan` |
| `whitebox_model.py` | 构建 operator database、逐层 timing query、单卡 DAG builder、artifact I/O | fine records / plan | white-box profile |
| `event_engine.py` | execution graph schema、多资源离散事件调度、critical path 与 overlap 指标 | events/resources | execution report |
| `collective_model.py` | collective query/backend 协议与 synthetic timing fixture | collective shape/topology | duration + provenance |
| `runtime_frontend.py` | 为未来 vLLM 劫持冻结 operator/collective/stream-wait 事件注册接口 | runtime metadata | execution graph |
| `evaluation.py` | workload-group holdout、误差分位数与自适应补点排序 | raw records | evaluation report |
| `cli.py` | validate/train/predict/replay 命令 | files + CLI args | JSON artifacts/results |
| `configs/*.json` | 模型、硬件与 runtime 固定信息 | N/A | reusable profile header |
| `examples/*.json*` | 仅用于 CPU smoke 的 synthetic fixture | N/A | reproducible demo |

## Data Format

### ExecutionGraph

```json
{
  "provenance": {
    "label_source": "synthetic_fixture",
    "formal_hardware_prediction": false
  },
  "resources": [
    {"resource_id": "gpu:0:compute", "capacity": 1.0, "kind": "gpu_compute"},
    {"resource_id": "link:nvlink:0-1", "capacity": 1.0, "kind": "nvlink"}
  ],
  "events": [{
    "event_id": "collective/all_to_all/0",
    "rank": null,
    "stream": "collective",
    "category": "all_to_all",
    "duration_us": 30.0,
    "dependencies": ["rank0/dispatch", "rank1/dispatch"],
    "resource_demands": {"link:nvlink:0-1": 1.0},
    "input_bytes": 1048576,
    "output_bytes": 1048576,
    "metadata": {
      "participants": [0, 1],
      "participant_dependencies": {"0": "rank0/dispatch", "1": "rank1/dispatch"}
    }
  }]
}
```

同 stream 顺序必须出现在 `dependencies` 中。collective 使用单一 rendezvous event，避免将同一通信
按 rank 重复计入 communication work。

### InferenceConfig

```json
{
  "model": {
    "model_id": "Qwen/Qwen3-30B-A3B",
    "num_layers": 48,
    "hidden_size": 2048,
    "num_experts": 128,
    "top_k": 8,
    "expert_intermediate_size": 768,
    "num_attention_heads": 32,
    "num_kv_heads": 4
  },
  "hardware": {
    "hardware_id": "NVIDIA-H20",
    "model_weight_bytes": 61064794624,
    "kv_bytes_per_token": 98304,
    "activation_bytes_per_scheduled_token": 4096
  },
  "workload": {
    "phase": "prefill",
    "batch_size": 4,
    "input_length": 1024,
    "context_length": 1024,
    "query_length": 1024,
    "scheduled_tokens": 4096,
    "total_kv_tokens": 4096
  },
  "routing": {
    "routed_tokens": 32768,
    "active_experts": 128,
    "tokens_per_expert": [256]
  },
  "runtime": {
    "runtime": "vllm",
    "vllm_version": "0.18.1",
    "precision": "bfloat16",
    "cuda_graph": false,
    "max_num_batched_tokens": 8192,
    "max_num_seqs": 32,
    "tensor_parallel_size": 1,
    "expert_parallel_size": 1
  }
}
```

`tokens_per_expert` 必须有 128 个非负整数；其非零数等于 `active_experts`，总和等于
`routed_tokens`。`scheduled_tokens`、query/context 和 batch 的关系由 phase-specific validation
检查。暂不支持的并行度显式失败。

### TrainingRecord

每行包含 `config` 和 CUDA Event labels：

```json
{
  "config": {"...": "InferenceConfig"},
  "labels": {
    "label_source": "cuda_event",
    "runtime": "vllm",
    "attention_us": 1000.0,
    "expert_us": 2000.0,
    "dense_us": 500.0,
    "other_us": 100.0,
    "step_us": 3700.0,
    "peak_memory_bytes": 65000000000
  }
}
```

正式训练默认只接受 `label_source=cuda_event` 且 labels runtime/version/precision 与配置一致。
CPU demo 使用 `label_source=synthetic_fixture`，只有 CLI 显式传入 `--allow-synthetic-fixture` 才可
训练，artifact 的 `calibration_scope` 必须写为 `synthetic-fixture`。组件和不能超过 `step_us`；
`other_us` 是 forward 内已测 transition/framework residual，最终 residual correction单独保存，
避免重复计数。

### CollectionManifest and RawStepRecord

采集 manifest 固定模型路径、runtime 参数和请求矩阵。每个 workload 使用离线 `LLM.generate` 提交
同批请求；prompt 由确定性 token ID 构造，不依赖 tokenizer 文本长度。采集器只支持 vLLM 0.18.1、
Qwen3-30B-A3B、单进程 TP=1/EP=1，且要求 `enforce_eager=true`；不满足时直接失败。
规则表采集固定关闭 prefix caching 与 async scheduling，防止跨 workload 前缀复用或异步批次重排改变
目标 shape；这些能力应由未来单独的 scheduler replay/残差实验覆盖。
每个 requested workload 必须独立执行 manifest 指定次数的 warmup，再清空 worker records 后进入正式
repeat，避免 Triton/attention 的新 shape cold-start 污染 operator response。

worker 侧包装当前实例的 `GPUModelRunner.execute_model`，不修改 vLLM 源码。step CUDA Event 覆盖
input preparation、model forward 和 logits，但不包含随后单独执行的 sampling/bookkeeping。模块事件定义为：

- Attention：所有 `Qwen3MoeAttention` 完整 forward 之和；
- Expert：所有 `Qwen3MoeSparseMoeBlock` 完整 forward 之和，包含 router gate 与 fused experts；
- Dense：embedding、两类 layer norm、final norm 和 `compute_logits` 显式事件之和；
- Other：`max(0, step - Attention - Expert - Dense)`，包含未归类 GPU preprocessing、transition 与
  logits 周边开销。

每层 MoE hook 仅保留 router logits 引用；step 结束事件记录后才计算 top-k/bincount，因此 routing
统计开销不进入 `step_us`。RawStepRecord 同时保留 request 级 scheduled/query/context 长度、逐层
expert histogram、聚合 histogram、峰值 allocated memory 和采集器版本。转换为 TrainingRecord 时：

- `batch_size` 是该 scheduler step 的活跃请求数，不是用户并发上限；
- `query_length` 是单请求最大 scheduled tokens，`scheduled_tokens` 是全 batch 总和；
- `context_length` 是该步结束后的单请求最大 KV 长度，`total_kv_tokens` 是其总和；
- mixed prefill/decode step 标记为 `prefill`，并在 raw record 保留逐请求 phase；
- routing histogram 跨全部 MoE 层求和，所以 `routed_tokens = scheduled_tokens × top_k × num_moe_layers`。

InferenceConfig 的 routing 语义采用“一次完整 model forward 的全部 MoE 层路由量”，使 Expert label
与 routing features 覆盖相同计时范围。

### PerformanceProfile

```json
{
  "step_us": 3700.0,
  "attention_us": 1000.0,
  "expert_us": 2000.0,
  "dense_us": 500.0,
  "other_us": 200.0,
  "residual_us": 100.0,
  "memory_bytes": 65000000000,
  "component_shares": {
    "attention": 0.2703,
    "expert": 0.5405,
    "dense": 0.1351,
    "other": 0.0541
  },
  "primary_bottleneck": "expert",
  "extrapolated_components": [],
  "artifact_provenance": {}
}
```

占比分母为四个非负组件之和，等于预测 step；`residual_us` 是 `other_us` 的组成部分，不再次加入
总和。输出包含 artifact runtime、数据量和训练范围。

### FineGrainedRawStepRecord

V2 collector 在既有 raw step 上新增：

```json
{
  "phase": "mixed",
  "kernel_identity": {
    "attention_backend": "FLASH_ATTN_V3",
    "moe_backend": "TRITON",
    "moe_kernel_config": "default:E128:N768"
  },
  "layer_records": [{
    "layer_id": 0,
    "tokens_per_expert": [0],
    "timing_us": {
      "attention": 100.0,
      "router": 20.0,
      "dispatch": 10.0,
      "expert_gemm": 200.0,
      "combine": 15.0,
      "dense": 30.0,
      "transition": 5.0
    }
  }],
  "fixed_us": 50.0,
  "step_us": 18290.0
}
```

`router` 是 gate projection 与 `select_experts` 的 CUDA 时间之和。`dispatch/expert_gemm/combine` 直接
对应 modular kernel 的 `_prepare/_fused_experts/_finalize`；monolithic kernel 不生成伪拆分记录。
`transition` 是同一 layer 中相邻阶段 end→start 的非负 CUDA timeline gap 之和。`fixed_us` 是 step
中直接测得的非层级 embedding/final norm/lm-head CUDA Event 之和，不允许通过 step 减法生成。
preprocess、层间、边界和未显式命名的串行 GPU 区间通过全局 event timeline 的 gap 计入 Transition。
collector 同时保存 `serial_composed_us`、`composition_overrun_us` 和 `uncovered_us`；这些是审计量，
不作为预测输入或 residual target。

### StepExecutionPlan

plan 保存 phase（prefill/decode/mixed）、realized batch、`N_q`、total/mean/max KV、dtype/backend/kernel
identity，以及每层 routing histogram。每个 `LayerExecution` 显式给出 Attention、Router、Dispatch、
Expert GEMM、Combine、Dense、Transition 的执行次数。Qwen3 当前为 48 个同构 MoE layers，但 schema
不把 48 写死，以支持其他纯 MoE/混合 MoE 模型。

### OperatorSignature

```text
(operator_type, phase, numeric_shape, categorical_shape,
 dtype, backend, kernel_config) -> median latency_us
```

numeric shape 分组件定义：Attention 使用 `N_q,total_kv,realized_batch,kv_mean,kv_max`；Router/Dense/
Transition 使用 scheduled tokens 与必要的二维 shape；Dispatch/Combine 使用 routed tokens、active
experts；Expert GEMM 使用 routed tokens、active experts、max/p95 expert load 与 kernel config。
同一 signature 多次采样取 median并保存 count/MAD。查询严格匹配 categorical identity；一维使用
分段线性，多维使用归一化局部 kNN 插值，并返回 nearest distance/extrapolation。

### ReplayStep

JSONL 每行包含 `step_id`、完整 `config` 和 `request_events`。事件字段为 `request_id`、
`arrival_us`、`emitted_tokens`、`finished`。Replay 按文件顺序串行执行预测 step，并以 step 完成时刻
作为该步 token 的可见时间。TTFT 是 arrival 到首 token，TPOT 是首 token 后相邻完成 token 的
平均间隔；只有一个输出 token 时 TPOT 为 null。

## Modeling Logic

### White-box V2 main path

1. 从 fine-grained raw records 展开逐 layer operator samples。
2. 按完整 signature 去重构建 `OperatorResponseDatabase`。
3. 对输入 plan 的每层分别查询七类 operator；Attention/Dense/Router 的相同 query 使用 artifact 内
   memoization，只计算一次再乘执行次数，MoE 依据每层 histogram 单独查询。
4. `serial` composition 将 Fixed 和所有节点相加；`dag` 接口接受节点依赖并计算 critical path，但
   第一版 artifact 在 Nsight audit 前不得默认启用。
5. 输出细粒度占比，同时聚合兼容视图：Attention；MoE=Router+Dispatch+Expert GEMM+Combine；
   Dense；Transition+Fixed。

V2 不使用 routing Ridge、Other Ridge 或 final residual。真实 step 只用于 holdout evaluation，不作为
组合器训练目标。

### Event-driven composition V3

最终组合器不再直接对 operator duration 求和，而是将 timing oracle 输出转换为 `ExecutionEvent`：

```text
event_id, rank, stream, category, duration_us,
dependencies[], resource_demands{}, collective metadata, bytes
```

`ExecutionGraph` 负责唯一 ID、依赖存在性、DAG 和正 duration/resource demand 校验。单 rank builder
将每层 Attention→Router→Dispatch→Expert GEMM→Combine→Dense→Transition 注册到 compute stream，
最后接 Fixed；因此单 stream 图的 critical path 必须严格等于旧 serial sum。

`DiscreteEventEngine` 使用非抢占式、确定性 earliest-ready 调度。每个 resource 有 capacity，事件在其
整个 interval 内持有声明的 demand；调度器从已完成依赖的候选中选择最早可获得全部资源者，同 start
时按 graph insertion order 仲裁。stream 顺序由显式依赖表示，不由名字隐式推断。resource capacity
允许表达独立 compute/copy/network engine 和 fractional occupancy，但第一版不求解全局最优排程。

collective 使用一个跨 rank rendezvous event：它依赖所有参与 rank 的 stream predecessor，并成为
所有后继的共同依赖。这样 collective start 为最慢 rank ready time，报告每个 rank 的 synchronization
wait。TP/EP 合成测试支持 All-to-All、All-Reduce、All-Gather、Reduce-Scatter；初始 duration 使用
明确标记为 `synthetic_fixture` 的 startup + bytes/effective-bandwidth backend。

执行输出分开表示：

- `total_work_us`：所有节点 duration 总和，可大于 step latency；
- `step_us`：资源约束排程的 makespan；
- `critical_path` 与 `critical_path_shares`：用于可加和的瓶颈归因；
- `exposed_communication_us`：完整排程 latency 减去 communication duration 置零后的 latency；
- `overlapped_communication_us`：communication work 减 exposed time，非真实链路测量；
- per-resource utilization 与 per-rank collective wait。

formal/synthetic provenance 必须传播到 execution report。没有正式 collective timing artifact 时，
多 rank graph 只能作为执行引擎验证，不输出 formal H20 TP/EP latency。

### vLLM forward hijack V4

`VllmForwardHijackHook` 实现现有 executor hook 协议，不改 vLLM 源码。launcher 在 import vLLM 前设置
emulator/CUDA-mock 环境并 monkeypatch `vllm_emulator.hooks.executor_hook.get_executor_hook`。每步路径为：

```text
SchedulerOutput
  -> SchedulerStepAdapter (realized batch/query/context)
  -> RoutingProvider (profile replay / future routing proxy)
  -> StepExecutionPlan
  -> Whitebox timing oracle
  -> ParallelForwardGraphCompiler
  -> DiscreteEventEngine
  -> ModelRunnerOutput placeholder + JSONL trace
```

`SchedulerStepAdapter` 维护 req_id 到 prompt length/computed tokens 的 CPU state，并在 request finish 时清理；
phase 由该 step 各 request 的 prefill/decode 状态共同决定。占位 token 只用于推动真实 vLLM 请求生命周期，
mid-prefill 不返回 token，完成 prefill/decode 返回固定非 EOS token。

`RoutingProvider` 是强制 provenance 边界。`ProfileReplayRoutingProvider` 从已有 raw H20 steps 选择最近
scheduled-token/phase anchor，逐层缩放整数 histogram 并保证每层 routed token 总数严格等于
`scheduled_tokens * top_k`。它用于 forward timing plan，不代表当前请求内容的真实 routing。

`ParallelLayout` 保存 world size、TP/EP groups、rank-to-node 和 topology。`ParallelForwardGraphCompiler`
以每层单卡 operator prediction 为模板，产生每 rank Attention/Router/Expert/Dense 事件，在 Attention 与
MoE 输出边界插入 TP collective，在 Expert 前后插入 EP All-to-All。collective event 依赖所有 participant
predecessors并成为全部 rank 后继，从而显式表达最慢 rank rendezvous。跨节点 collective 同时占用
network fabric 与参与节点 NIC；节点内 collective 占用相应 local fabric。

Data Format：V4 trace 顶层包含 `schema_version=vllm-forward-hijack/v1`、`step_index`、
`scheduler_context`、`routing_provenance`、`timing_provenance`、`parallel_layout`、`execution_plan`、
`execution_report` 与 `fake_output`。`formal=false` 只要 routing、parallel compute scaling 或 collective
timing 任一来源为 synthetic/replay。

### Multi-GPU fixed-configuration oracle V5

首个 formal artifact 的配置键固定为：

```text
model_id, hardware_id, gpu_count=2, tp=2, ep=2,
vllm_version=0.18.1, precision=bfloat16, cuda_graph=false,
all2all_backend, expert_placement, CUDA version, NCCL version
```

`MultiRankCollector` 在每个 vLLM worker 安装同一组 forward hooks。driver 将各 rank records 按
`workload_index/repeat/step_index` 对齐，保留原始 `rank_records`，并产生 aggregate record：

```text
step_us = max_r step_us[r]
rank_skew_us = max_r step_us[r] - min_r step_us[r]
components_us[stage] = max_r components_us[r, stage]
routing = rank-consistency checked global histogram
```

aggregate component 只用于诊断；主标签始终是同步语义的 `max_r step_us[r]`。如果各 rank scheduler
metadata、step 数或 global routing histogram 不一致，采集立即失败。

`NcclResponseTable` 读取 torch.distributed NCCL microbenchmark JSON，以
`collective_type/rank_count/topology/backend` 为 categorical identity，对 message bytes 使用 log-space
分段插值；超出测量范围返回 extrapolation flag 和 anchor distance。正式 provenance 包含 GPU UUID、
NVLink topology、driver/CUDA/NCCL/PyTorch 版本及 warmup/repetition 设置。

`FixedConfigForwardOracle` 对同一配置的直接 step labels 建表。特征分为：

- workload：phase、batch、scheduled tokens、n_q、total/mean/max KV；
- routing：逐层 rank load 的 max/mean、p95/mean、CV 与 routed tokens；
- request composition：Prefill/Decode request 数和 token 比例。

第一版使用 phase-strict local interpolation；查询输出 predicted step、nearest anchor distance、是否外推和
查询耗时。组件 execution graph 作为 auxiliary report，不参与 formal total 的强制求和。评估以
leave-one-workload-out、unseen length 和 large-region holdout 报告 MAPE/p95/max，同时报告每次预测
CPU wall time。只有该固定配置的严格评估达标后，才扩展 TP4 或其他 EP/通信 backend。

Data Format：NCCL JSON 顶层为 `schema_version=nccl-response/v1`、`profile_header`、`samples`；每个 sample
包含 collective、message bytes、rank count、每 rank repetitions、median/p95/min/max μs。多 rank forward
JSONL 顶层为 `schema_version=multirank-forward/v1`、`config_identity`、`requests`、`rank_records`、
`aggregate`、`routing_consistency`。

### Trusted range and profiling budget V6

V6 将 V5 明确定义为配置级稀疏 profiling oracle。训练时按 phase 和完整 feature signature 合并重复
records，目标取最慢 rank step latency 的 median。NCCL provenance 可以附着在 artifact 上用于审计，但
不进入总时间公式。

每个 phase 在标准化 feature 空间内计算 anchor 到最近其他 anchor 的距离分布，可信半径取该分布 p95。
一次查询只有同时满足以下条件才是 `supported`：

1. phase 存在；
2. 所有连续特征均落在训练 bounding box 内；
3. 最近 anchor 距离不超过 phase 的可信半径。

否则返回 `support_status=not_predictable`、`needs_profile=true`、原因、最近距离与阈值；默认
`step_us=null`。内部仍保留 `candidate_step_us` 供严格离线评估诊断，但 runtime/hijack 不得把它当正式
预测。评估器显式使用 `allow_unsupported=true`，同时分别报告全表候选误差、可信覆盖率和可信域内误差。

预算实验以唯一 `(phase, feature signature)` 为 anchor 单位。`adaptive_farthest` 选择器先给每个 phase
一个中心 seed，再按 phase 内归一化最近距离做 farthest-point coverage；选择过程不读取 latency label。
它是已有候选池上的 retrospective replay，不等同于真实在线采样。每个预算报告：

- 全 surface MAPE/p95/max APE；
- supported fraction 与 supported-only MAPE/p95；
- 最慢 10% workload 的 Top-k recall 与 workload surface Spearman；
- 被选 anchor 的真实 forward median 时间之和（不含模型加载、warmup 和进程启动）；
- 训练 wall、median query wall、artifact size（保存时）。

只有一个 deployment identity 时，`configuration_ranking.status=not_applicable`。至少存在两个配置并有同一
服务 workload suite 的真实指标后，才增加配置排序准确率与配置 Top-k recall。

Data Format：`FixedConfigPrediction` 包含 `step_us`、`candidate_step_us`、`support_status`、
`needs_profile`、`support_reason`、`nearest_distance`、`coverage_threshold`、`extrapolated`、neighbors、
query wall 和 config identity。budget report 顶层包含 budgets、selection protocol、deployment identity、
configuration-ranking status 和逐预算 metrics/selected signatures。

受控 Mixed 采集扩展 `CollectionWorkload`：可选 `prompt_token_lengths` 明确每个 request 的 prompt 长度，
其长度必须等于 batch size；未指定时继续使用统一 `prompt_tokens`。manifest 保存原始异构请求组成，真实
vLLM scheduler 决定最终 Prefill/Decode/Mixed step，不从 requested workload 静态构造 phase。

### Observed-error adaptive acquisition V7

`residual_aware` 与 `phase_aware_farthest` 共用同一个 unique-anchor candidate pool、phase-specific feature
normalization 和 nested budget sequence。初始 seed 为每 phase 的中心点与距中心最远点；后续只使用已测
anchors 计算 acquisition：

```text
acquisition(x)
  = geometric_distance(x, S)
  + beta * local_latency_disagreement(x, S)
  + gamma * phase_observed_LOO_error(S)
```

- `geometric_distance`：候选到同 phase 最近已测 anchor 的归一化距离；
- `local_latency_disagreement`：候选最近 k 个已测 anchors 的加权 latency coefficient of variation；
- `phase_observed_LOO_error`：已测 anchors 内局部 IDW leave-one-anchor-out 相对误差的稳健分位数；
- `S` 只含已经 profile 的 anchors。候选真实 latency 仅在被选中后加入 `S`。

三个 acquisition 项在当前未采样候选集合内按 phase 做稳健归一化，再用固定系数组合；phase error 项用于
把新增预算转向 observed residual 较高的 Prefill/Mixed，而不是按候选数量持续分配给已稳定的 Decode。
所有预算从一个最大预算序列取 prefix，保证 nested。选择日志对每个 anchor 保存 step、phase、distance、
local disagreement、phase observed error 和 final score。

observed-error acquisition 在 6 个 seed 时 residual 不稳定，因此 V7.2 固定使用 16-point
`phase_aware_farthest` warm start；budget=16 必须与 baseline 是同一组 anchors。16 点之后才启用 observed
error。为防止 observed error 在有限 pool 中耗尽困难 phase、同时丢失 Decode 几何覆盖，另增加两个只依赖
候选 feature/count 的约束：每个 phase 至少获得其候选数量比例对应预算的 50% (`exploration_floor=0.5`)；
任一 phase 最多选择其候选 anchors 的 85% (`phase_cap=0.85`)。若下一步存在未达到 floor 的 phase，先在
这些 phase 中按 acquisition 选点；否则在未达到 cap 的全体候选中选择。约束不读取任何未测 latency。

Data Format：budget report 新增 `strategy`、`acquisition_config`、`selection_trace` 和
`uses_unprofiled_latency_labels=false`。同一 JSON 可包含 baseline 与 residual-aware 两条曲线；比较表按预算
报告 delta MAPE/p95、phase p95、coverage 和 Top-k recall，不选择性隐藏退化预算。

### Forward-only structural predictor V10

V10 不包含请求到达、排队或 scheduler replay，输入是已经确定的 `StepExecutionPlan`。新增/修改文件：

- `parallel_forward.py`：`ParallelLayout.vllm(...)` 接受显式 `world_size/tp_size/ep_size`，TP/EP
  groups 各自只要求覆盖全部 ranks，允许 group alias；`expert_coordinate` 返回 rank 在其 EP group 内的
  coordinate。routing/linear placement 产生 per-rank expert load 和 source→destination token matrix。
- `ParallelForwardGraphCompiler`：按 layer/rank 生成 Attention、Router、Dispatch、Expert GEMM、Combine、
  Dense、Transition；TP collective 由 activation tensor shape 决定。EP backend 为
  `allgather_reducescatter` 时插入 All-Gather/Reduce-Scatter，为 `alltoall` 时按 routing send matrix 插入
  All-to-All。collective metadata 保存 volume derivation、participant readiness 和 timing provenance。
- `structural_forward.py`：组合 `WhiteboxMoEModel`、graph compiler、`NcclResponseTable` 和
  `DiscreteEventEngine`，输出 `structural-forward/v1`。formal status 同时检查 compute/routing/collective/
  topology provenance；任一为 heuristic/synthetic 时报告 `formal=false` 和明确 reasons。
- `test_structural_forward.py`：覆盖 TP2/EP1 world=2、TP2/EP2 group alias、routing imbalance、backend
  specific volume、慢 rank collective wait、TP1/EP1 退化和 unsupported layout。

Data Format：

```text
schema_version: structural-forward/v1
input: {execution_plan, parallel_layout, runtime_config, routing_semantics}
prediction: {step_us, rank_finish_us, memory_bytes_per_rank}
attribution: {critical_path, critical_path_shares, communication_work_us,
              exposed_communication_us, synchronization_wait_us, primary_bottleneck}
support: {formal, reasons, extrapolated_signatures}
graph: {resources, events, provenance}
```

V10.0 已实现 no-overlap dependency policy 与 `predict-structural-forward` CLI。未采集 TP rank-local compute-exclusive responses 前，旧 TP1
Operator DB 加 scaling 只能作为 engine prototype，必须标记 `parallel_compute_scaling` reason；不能报告为
正式 H20 跨配置精度。完成基本图验证后再补 compute-exclusive anchors、co-run slowdown 和 423 个
shape-aligned steps 的 MAPE/ranking evaluation。

### Rank-local compute-exclusive anchors V10.1

`worker_collector.py` 在既有 module/fine CUDA Events 之外保存 `rank_local_compute_us`。Attention 的
compute-exclusive 边界定义为 Attention module start 到其内部第一个真实 TP collective start；这覆盖
QKV projection、Q/K norm、RoPE、attention kernel 与 output GEMM，但排除末尾 collective。若 TP>1 且
Attention interval 内没有捕获 collective，记录 `valid=false/missing_tp_collective`，不得退回 inclusive time。
Router 与 Expert GEMM 继续使用真实 `select_experts/gate` 和 modular `_fused_experts` CUDA Event；Dispatch/
Combine 暂不声称 compute-exclusive，因为 prepare/finalize 内部可能混合 EP communication。

每层输出：

```text
rank_local_compute_us: {
  attention: {duration_us, valid, boundary, collective_event_count},
  router: {duration_us, valid, boundary},
  expert_gemm: {duration_us, valid, boundary}
}
```

这些 anchors 只在相同 TP/local kernel identity 下查询，不按 TP size 做除法。Transition/runtime gap 继续由
少量 phase affine calibration 吸收；它不能回填到任一 compute node。正式结构评估必须分别报告 raw
structural 与 calibrated structural，避免校准掩盖通信量或 rank-load 逻辑错误。

V10.1 的最终严格 5-fold workload holdout 会在每个 fold 内重建 rank-local DB，并从 train-only surface
无标签选择 32 个 phase-aware farthest calibration anchors。评估键包含完整 realized request shape，修复了
同 workload/step/phase 下不同 scheduler shape 被覆盖的问题。固定 CUDA Event 质量门禁排除 5 条明确的
runtime tuning 污染记录后，TP2/EP1 与 TP2/EP2 的 MAPE 分别为 4.92%/5.30%，p95 APE 为
16.88%/16.46%；139 个 exact-aligned steps 的配置 Top-1 准确率为 96.40%。完整报告见
`code/results/h20_structural_compute_full_v1/strict_evaluation_qc_v1.json`。

查询路径使用 exact-signature index/cache、event ready set 和 unit-capacity resource fast path；summary-only
模式不序列化数千个 DAG events。50 次进程内查询的 warm median 为 TP2/EP1 68.43 ms、TP2/EP2
92.37 ms。由于仍为 partial component coverage + serialized no-overlap，formal status 保持 false。

评估分成两个明确协议：

- `pool_surface_recovery`：选中的 anchor 使用实测值，衡量有限候选 surface 被恢复的程度；同时报告实际
  被选 phase 数和 forward CUDA profiling cost。
- `fixed_workload_holdout`：按 requested workload ID 做确定性 fold，先完全移除 test workloads，再从其余
  pool 生成 acquisition sequence。测试 features 和 latency 都不参与 normalization、seed、distance 或
  observed-error acquisition；所有预算共享相同 test workloads。

fixed holdout report 保存 train/test workload IDs、train candidate 数、test unique signature 数和
`test_features_visible_to_acquisition=false`。若候选预算超过 train pool，显式失败，不自动缩小预算。

`aggregate-budget-folds` 对完全相同 protocol/modulus 的 fold JSON 做 paired aggregation，保存每个策略和
预算的 mean/std/min/max，以及 adaptive-minus-baseline 的 paired delta。推荐预算必须同时满足：相对
baseline 的 mean MAPE 与 mean p95 均不退化、mean Top-k recall 不退化，并在满足条件的预算中取最小值；
若没有预算满足，返回 `no_stable_improvement`。不得在聚合后继续用同一 folds 调 acquisition 超参数。

### Second configuration reference surface V8

第二配置 identity 为 `2xH20 + TP2/EP1 + BF16 eager + vLLM 0.18.1`。collector 仍启动两个 ranks，但
`enable_expert_parallel=false`，每个 rank 的 Expert 路径遵循真实 TP runtime；aggregate label 仍取较慢
rank。采集使用与 TP2/EP2 相同的 requested workload suite，包括异构 prompt Mixed workload，使两个配置
可以在 requested-workload 层面对齐。

第一阶段只训练独立 TP2/EP1 fixed-config oracle 和 reference budget curve，不训练跨配置模型。Data Format
保持 `multirank-forward/v1`；`config_identity.expert_parallel_size=1` 使 loader/merge/oracle 自动拒绝与 EP2
混合。后续 calibration report 以 requested workload 对齐两个 identity，明确区分 source anchors、target
calibration anchors 和 target-only holdout。

`compare-fixed-configs` 仅做同 requested workload/repeat/step 且 request-level realized shape 完全一致的成对
实测比较，输出 target/source latency ratio 的 median/mean/p95 和 target-faster fraction。shape 不一致的
scheduler steps 单独计数并排除；该报告是配置差异诊断，不是跨配置预测或 calibration accuracy。

### Cross-configuration sparse calibration V9

V9 不把 source 与 target records 合并建表，而把 source fixed-config oracle 作为一个独立、只读的先验：

```text
source oracle(shape-aligned source step) -> source_us
target realized features + B measured target anchors -> target/source ratio calibrator
source_us * predicted ratio -> predicted target_us
```

第一版实现两个可审计校准器。`phase_scale` 对每个 phase 取已测 target/source ratio 的 median；
`local_ratio_idw` 在 transfer-safe feature 空间内对 ratio 做 phase-strict inverse-distance interpolation，并以
phase median 作为远距离 shrinkage prior。transfer-safe features 为 batch、scheduled/query、KV/context、
Prefill request/token fraction、global expert max/mean 和 active-expert fraction；三个 EP-rank load features
因为 EP2 与 EP1 的 rank partition 口径不同而排除。

目标校准点以完整 target signature 去重，重复测量取 median。每个 fold 先移除 test requested workloads，
再在 train target candidates 上生成一个最大预算的 `phase_aware_farthest` 序列，预算 8/16/32 取其 prefix，
因此严格 nested 且 acquisition 不读取 latency。校准和 target-only baseline 使用完全相同的 target anchor
signatures。测试只保留 workload/repeat/step/phase 与逐 request realized shape 完全对齐的 source-target
steps，并按 target signature 去重。

source oracle 是已经覆盖 source 配置的既有 artifact，因此评估允许查询 holdout workload 对应的 source
step signal；这不是 target leakage，但构成部署输入依赖。报告固定保存
`source_test_signal_available=true`、`target_test_features_visible_to_acquisition=false`、
`target_test_labels_visible=false`。未来若没有 source scheduler/routing replay，必须显式拒绝，不能用 target
shape 静默伪造 source signal。

可信域由 target calibration anchors 的 phase bounding box 与 anchor nearest-neighbor p95 半径共同定义。
默认查询在域外返回 `not_predictable`；离线评估可以显式读取 candidate estimate，同时单独报告 coverage。
评估模型包括 `zero_shot_source`、`phase_scale`、`local_ratio_idw`、`target_only_same_budget`，并附已有
target-only 64-anchor 五折 aggregate 作为非同测试 surface 的 reference，禁止把它当 paired delta。

Data Format：`cross-config-calibration/v1` artifact 保存 source/target identity、transfer feature indices、
phase ratio tables、可信阈值、selected target signatures 和 source-signal contract；prediction 保存
`target_us`、`candidate_target_us`、`source_us`、`ratio`、support status/reason、nearest distance、coverage
threshold、extrapolated 和 query wall。evaluation report 保存 pairing diagnostics、fold/workload split、
budgets、各 baseline 的 overall/by-phase metrics、target anchor phase allocation 与 leakage flags。

### Service-level trajectory replay V10

服务级评估以 `(requested_workload, repeat)` 为单位，而不是 step key。source 与 target 各自从对应
`multirank-forward/v1` records 恢复真实 scheduler trajectory，按本配置的 `step_index` 排序；只校验两边
requested workload 定义一致，不要求 step 数、phase 序列或 request shape 一致。

对一个 trajectory，虚拟时间从 0 开始，所有静态批次请求 arrival=0。每个 step 串行推进其最慢-rank
forward duration。request 在完整 Prefill 结束或 Decode step 末尾产生一个 token；chunked Prefill 未完成时
不产生 token。到达 requested `output_tokens` 后完成。使用 step-end visibility 计算：

```text
TTFT = first_token_time - arrival
TPOT = (last_token_time - first_token_time) / (output_tokens - 1)
E2E  = finish_time - arrival
Throughput = total_output_tokens / trajectory_elapsed_time
```

同一 target trajectory 建立三种时间线：真实 target labels、target full oracle、fold 内 cross-config
calibrator。校准路径不查找 aligned source step，而是以 target realized requests 和 global routing histogram
重新按 source EP size 计算 source fixed-oracle feature，得到 `counterfactual_source_on_target_trajectory` signal；
ratio calibrator 仍只使用 V9 的 transfer-safe target features。这使既有 shape mismatch 不再被丢弃，但
source/correction 任一可信域外时必须计入 unsupported coverage，正式 runtime 仍会拒绝。

严格 protocol 为 requested-workload 五折。每折 target test workloads 在 calibration 前完全移除，在其余
workloads 的 aligned paired pool 中重新执行 label-free 32-anchor selection。source full oracle 被视为已有
配置资产，可以覆盖 test workload；target full oracle 仅作上界 reference。每个 held-out workload/repeat
分别输出实际与预测服务画像，并在 source/target 两配置间计算 mean TTFT、mean TPOT、p95 E2E、
output-token throughput 和 SLO attainment 的 Top-1 accuracy 与 optimality gap。

SLO 由 `--ttft-slo-ms`、`--tpot-slo-ms`、`--e2e-slo-ms` 显式传入。request 同时满足所有已配置阈值才
算 attained；未传任何阈值则 SLO 字段为 `not_configured`，且不参与排序。

Data Format：`service-config-ranking/v1` 顶层保存 protocol、两个 identity、SLO、folds、service error 与
configuration ranking aggregate。trajectory 保存 workload/repeat、独立 phase/step sequence、actual 与
predicted summary、step support fraction；summary 包含 TTFT/TPOT/E2E 分位数、output/request throughput
和 SLO attainment。provenance 必须标记不含 scheduler CPU、output processing 和动态 arrival queueing。

### Candidate table and conservative pruning V11

`build_candidate_table_from_service_report` 将 V10 folds 中每个 workload 的三次 repeat prediction 取 median，
形成 `(candidate_id, workload_id)` 行。source 使用 `source_full`，target 使用 `target_calibrated`；误差区间
来自对应服务指标五折 p95 APE：

```text
latency_lower = prediction * max(0, 1 - error_p95)
latency_upper = prediction * (1 + error_p95)
throughput_lower/upper 同理
```

每 workload 的 `coverage` 为 repeat 的 supported-step fraction median；per-GPU memory 为该配置该 workload
所有实测 steps 的 peak maximum。候选表不保存 target 真值作为剪枝输入。arrival rate、cost 和 launch status
只有调用方显式提供时才可参与判断；缺失字段保持 null/unknown。

Pruner 对每个 candidate-workload 依次执行：

1. **Feasibility**：launch failed、world size 超设备池、TP/EP 不整除 world、memory 超
   `gpu_memory_bytes * max_memory_utilization` 时 prune；unknown/missing 进入 validation。
2. **Confidence**：coverage 小于阈值或 prediction status 非 supported 时进入 profile queue，并停止后续
   性能规则，避免用不可信 candidate estimate 剪枝。
3. **SLO**：`risk_averse` 使用 latency upper bound；upper>SLO 为 risk violation，lower>SLO 另标
   certain violation。SLO 未配置时 not evaluated。
4. **Capacity**：将 request arrival rate 乘以每请求 output tokens，转换为 output-token demand，再与同单位
   throughput lower bound 比较并计算 replicas；进入 replica evaluation，不假设 replicas 对 latency/queueing
   无影响。显式 `capacity_action=prune` 或 required replicas 超过 policy limit 时才直接剪候选。
5. **Pareto**：仅在前四步 retained 的同 workload candidates 上执行 conservative interval dominance。

Pareto 比较 TTFT、TPOT、p95 E2E、throughput 和显式 cost。A 支配 B 需 A 的 latency upper≤B lower、
throughput lower≥B upper、cost≤，且至少一项严格；缺 cost 时跳过 cost 维度并记录。输出保留每条 rule
decision、reason、bound values 和 dominator，便于后续 MILP/CP-SAT 读取 retained set。

Data Format：`config-candidate-table/v1` 顶层包含 hardware pool、workloads、candidates、provenance；每个
candidate 包含 identity、parallel layout、launch status、cost、worst-case memory 和 per-workload predictions/
intervals/coverage。`candidate-pruning-report/v1` 包含 policy、workload decisions、profile/validation/replica
queues、Pareto sets、全局 surviving candidates 与 rule counts。

### Nsight overlap audit

audit run 开启独立 NVTX 标记，范围包括 `STEP/phase` 与每层 Attention、Router、Dispatch、
ExpertGEMM、Combine、Dense。Nsight 报告至少输出：使用的 CUDA stream IDs、每个 phase 的 GPU span、
跨 stream overlap、同 stream stage ordering、CUDA Event serial composition 与 timeline span 的差异。
NVTX 只用于 audit run，不混入正式 response records。

判定规则：代表性 Prefill/Decode/Mixed 均只有单一 compute stream，且 component interval overlap
不超过各 phase GPU span 的 1%，同时 serial composition overrun 不超过 1%，则 artifact 可标记
`audited_serial`；否则 composition engine 必须使用 interval DAG critical path。

### Sparse anchor grid

首轮 requested workloads 使用对数断点；由于真实 scheduler 可能拆分请求，最终建表仍以 realized
plan 为准。当前 H20 grid 覆盖 batch `{1,2,4,8}`、prompt/context `8–8192` 的 25 个稀疏配对，
Decode 由每组输出 steps 覆盖 context，Mixed 由 continuous batching 的真实交错 step 记录。prompt
必须采用固定种子的多样 token 序列，并将 pattern 写入 provenance；全相同 token 只允许作为极端
routing stress case，不进入正式 routing 泛化结论。

### Strict evaluation protocols

- unseen-batch：按 realized batch 整组留出；
- unseen-length：按 scheduled/KV 对数 bucket 留出；
- region：小/中区域训练，大区域测试；
- phase：Prefill、Decode、Mixed 分别整 phase 留出；缺少对应 family 时报告 unsupported，而非回退；
- signature-distance：按 nearest normalized distance 分桶报告 APE；
- component/bottleneck：只在具有同口径细粒度真实标签的 steps 上计算 component MAPE 和 top-1/group
  bottleneck accuracy。

当前 `phase_holdout` 对 Mixed 会报告 `not_predictable`：Attention response family 仍以 phase 为
categorical identity，Prefill/Decode anchors 不能无依据地冒充 mixed Attention kernel。后续必须采集
受控 mixed ratio anchors 或实现可验证的 mixed attention execution decomposition，不能静默 fallback。

### Historical V1 response tables (baseline only)

- Attention：按 phase 分开，以 batch/query/context/KV 为 shape key；重复 shape 取 median，查询使用
  range-normalized inverse-distance interpolation。
- Dense：scheduled tokens 到 latency 的按 phase 分组分段线性插值。
- Expert base：`routed_tokens × active_experts` 重复 shape 取 median并做二维插值。
- Routing correction：用 `max/mean`、`p95/mean`、CV、Gini、zero ratio 拟合 Ridge，只预测
  `measured_expert - base_expert`。
- Other：用 phase、batch、scheduled tokens、KV、CUDA Graph 拟合非负裁剪的 Ridge。
- Final residual：用同一小特征集拟合 `step - predicted components`，预测时允许有符号修正，但
  `other + residual` 最终裁剪为非负。
- Memory：默认采用 weights + KV bytes/token + activation bytes/scheduled-token；若配置与 artifact
  hardware ID 不一致则失败。

### Basic RF baseline

Basic RF 只使用 phase、batch、query/context、scheduled tokens 和 KV，直接预测 step latency。
它与组件模型共用 split 和输入 schema，但不用于瓶颈归因。

## Public Interfaces

- `InferenceConfig.from_dict(raw) -> InferenceConfig`
- `TrainingRecord.from_dict(raw) -> TrainingRecord`
- `ComponentModel.fit(records) -> ComponentModel`
- `ComponentModel.predict(config) -> PerformanceProfile`
- `BasicRFModel.fit(records) -> BasicRFModel`
- `replay_steps(model, steps) -> ReplaySummary`
- `save_artifact(model, path)` / `load_artifact(path)`
- `StepExecutionPlan.from_dict(raw) -> StepExecutionPlan`
- `OperatorResponseDatabase.fit(samples) -> OperatorResponseDatabase`
- `WhiteboxMoEModel.fit_raw(records) -> WhiteboxMoEModel`
- `WhiteboxMoEModel.predict(plan) -> WhiteboxProfile`
- `evaluate_holdout(raw_records, holdout_workloads) -> EvaluationReport`
- `ExecutionGraph.add_event(event) -> None`
- `DiscreteEventEngine.simulate(graph) -> ExecutionReport`
- `build_single_rank_graph(profile) -> ExecutionGraph`
- `CollectiveTimingBackend.predict(query) -> CollectiveEstimate`
- `RuntimeEventRecorder.record_operator(...) -> event_id`
- `RuntimeEventRecorder.record_collective(...) -> event_id`
- `RuntimeEventRecorder.record_stream_wait(...) -> event_id`
- `FixedConfigForwardOracle.predict_raw(..., allow_unsupported=False) -> FixedConfigPrediction`
- `evaluate_profiling_budgets(rows, budgets, method) -> budget report`
- `merge_multirank_record_files(paths, output) -> merge summary`
- `CrossConfigRatioCalibrator.fit(paired_anchors) -> CrossConfigRatioCalibrator`
- `CrossConfigRatioCalibrator.predict(source_us, target_raw) -> CrossConfigPrediction`
- `evaluate_cross_config_calibration(...) -> calibration report`
- `replay_multirank_trajectory(rows, duration_provider, slo) -> service profile`
- `evaluate_service_configuration_ranking(...) -> service ranking report`
- `build_candidate_table_from_service_report(...) -> candidate table`
- `prune_candidate_table(table, policy) -> pruning report`

## CLI

```text
python -m moe_perf_model validate-config CONFIG.json
python -m moe_perf_model train RECORDS.jsonl --output ARTIFACT.joblib
python -m moe_perf_model predict ARTIFACT.joblib CONFIG.json --output PROFILE.json
python -m moe_perf_model replay ARTIFACT.joblib TRACE.jsonl --output SUMMARY.json
python -m moe_perf_model collect MANIFEST.json --output-dir results/h20_collection
python -m moe_perf_model train-whitebox RAW_STEPS.jsonl --output WHITEBOX.joblib
python -m moe_perf_model predict-whitebox WHITEBOX.joblib PLAN.json --output PROFILE.json
python -m moe_perf_model evaluate-whitebox RAW_STEPS.jsonl --output EVAL.json
python -m moe_perf_model audit-nsight REPORT.sqlite RAW_STEPS.jsonl --output AUDIT.json
python -m moe_perf_model report-surface RAW_STEPS.jsonl --output-json SURFACE.json --output-markdown SURFACE.md
python -m moe_perf_model merge-multirank INPUT1.jsonl INPUT2.jsonl --output MERGED.jsonl
python -m moe_perf_model evaluate-budget-curve MULTIRANK.jsonl --budgets 16 32 64 96 128 187 --output CURVE.json
python -m moe_perf_model aggregate-budget-folds FOLD0.json FOLD1.json --output AGGREGATE.json
python -m moe_perf_model compare-fixed-configs SOURCE.jsonl TARGET.jsonl --output COMPARISON.json
python -m moe_perf_model train-cross-config-calibration SOURCE.jsonl TARGET.jsonl SOURCE.joblib --budget 16 --output CALIBRATION.joblib
python -m moe_perf_model evaluate-cross-config-calibration SOURCE.jsonl TARGET.jsonl SOURCE.joblib --budgets 8 16 32 --output CALIBRATION_EVAL.json
python -m moe_perf_model evaluate-service-ranking SOURCE.jsonl TARGET.jsonl SOURCE.joblib TARGET.joblib --target-calibration-budget 32 --output SERVICE_RANKING.json
python -m moe_perf_model build-candidate-table SERVICE_RANKING.json SOURCE.jsonl TARGET.jsonl --gpu-memory-bytes BYTES --output CANDIDATES.json
python -m moe_perf_model prune-candidates CANDIDATES.json --policy PRUNING_POLICY.json --output PRUNING.json
```

`train` 默认构建 component model；`--model basic-rf` 构建 baseline。仅 demo 可增加
`--allow-synthetic-fixture`。`predict/replay` 若 artifact
不是 component model，则不输出伪造的 component attribution并显式报错，Basic RF 只通过单独
baseline evaluation 接口使用。

## Implementation Order

1. 固定 requirements、配置和 schema。
2. 实现 response tables 与边界标记。
3. 实现 component model、Basic RF、artifact I/O。
4. 实现 scheduler replay。
5. 实现 CLI、examples、README 和测试。
6. 运行 CPU tests、compileall 和 train→predict→replay smoke。
7. 安装 same-runtime worker collector，执行 dry-run、最小 H20 smoke，再扩大规则 workload。
8. 细化 modular MoE hook 与 per-layer raw schema。
9. 实现 operator database、execution plan、显式 composition 和 holdout evaluation。
10. 用 synthetic fine fixture 做 CPU 单测，再用少量真实 H20 fine records 做 schema/composition smoke。

## Known Boundaries

- examples 是 `synthetic_fixture` CPU 数据，只验证代码，不是 H20 精度结果；artifact provenance
  不得出现 `cuda_event` 或 `same-runtime`。
- 第一版 replay 是离线串行 step replay，不复刻 vLLM admission、preemption、prefix cache 和异步
  output processor；这些需要后续接 LLM-Emu/vLLM scheduler adapter。
- TP/EP 通信接口预留，但本轮不返回零通信来冒充已支持。
- 没有相同 vLLM runtime CUDA profile artifact 时，工具不能给出正式 H20 性能预测。
- eager hook 采集用于建立响应表；CUDA Graph 需要 graph-compatible range/event 采集，不能把 eager
  hook 数值标成 `cuda_graph=true`。
## Dense–MoE Transfer Experiment

Dense collector 在真实 vLLM forward 中以 CUDA Event 包围每层 `Qwen3Attention` 和 `Qwen3MLP`，保留真实
scheduler step latency。归一化标签为：

```text
attention_us = sum(Qwen3Attention)
ffn_us       = sum(Qwen3MLP)                 # Dense
ffn_us       = sum(Qwen3MoeSparseMoeBlock)   # MoE
other_us     = max(step_us - attention_us - ffn_us, 0)
```

实验复用 `h20_sparse_anchor_v1.json` 的 25 个 workload，新增 Dense manifest 仅替换模型结构、权重路径及
对应显存 provenance。分析器先按 `(workload_index, phase)` 聚合 step，再对重复值取中位数；同时尝试按
realized batch、scheduled tokens、context/KV 精确对齐并报告覆盖率。转移仅在同 phase、同 batch 的负载有序
序列中定义，主瓶颈发生变化才记为 transfer；边界仅报告为相邻已测 anchors 的区间。

## Multi-GPU Communication-exclusive Attribution

新增共享 communication hook，在 worker 初始化后取得真实 TP/EP `GroupCoordinator`，包装
`all_reduce`、`all_gather(v)`、`reduce_scatter(v)`。每次调用保存 group kind、collective type、输入 bytes、
CUDA start/end event。事件只在 collector active step 内记录，并在 worker 卸载时恢复原方法。

step 同步后将每个 communication event 按时间包含关系归属到 Attention、FFN 或 Other module interval：

```text
T_attention_compute = T_attention_inclusive - T_comm_nested_in_attention
T_ffn_compute       = T_ffn_inclusive       - T_comm_nested_in_ffn
T_communication     = sum(T_collective_device_elapsed)
T_other             = max(T_step - T_attention_compute - T_ffn_compute
                          - T_communication, 0)
```

若通信事件区间重叠，`sum(T_collective)` 只作为 communication work；归一化时改用区间 union 得到 exposed
device time，并单独保存 work/union/overlap。当前 BF16 eager 路径预期为单 compute stream，但实现不依赖
这一假设。多 rank aggregate 保存所有 rank records，并以 `argmax(rank_step_us)` 为 critical rank；报告只从
该 rank 读取互斥组件。Dense TP2 与 MoE TP2/EP2 采用独立 manifests，但 requested workload 完全一致。

### Communication Attribution Audit Correction

真实复验表明上面的 high-level API-event 互斥分解不能作为最终通信归因：collective 异步返回时，CUDA
Event end 可能记录在调用 stream，而实际 NCCL/custom-all-reduce kernel 和等待发生在另一 stream 或后续
同步点，导致 communication 与外层 Attention/Expert 时间反向迁移。总 step 稳定但通信占比可在重复间从
个位数跳至 80%，这是测量伪影。

最终口径改为两路独立证据：无 profiler 的 component envelope 只比较 Attention 与 FFN/Expert；通信由
worker 初始化完成后的同步 GPU-only CUPTI trace 识别真实 NCCL/custom-all-reduce kernels。为消除 profiler
造成的 rank 启动偏差，通信报告采用逐 collective 的 rank-min duration 之和，并明确标为 service-time
lower bound。该下界不能与 component envelope 相加，也不能单独证明 exposed critical-path contribution。
