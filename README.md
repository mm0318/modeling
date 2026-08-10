# Same-runtime MoE Component Modeling

## Overview

该工具输入 Qwen3-30B-A3B、H20、workload、MoE routing 和 vLLM runtime config，使用同一
runtime 的 Attention、Router、Dispatch、Expert GEMM、Combine、Dense、Transition 与 Fixed 响应表
预测 scheduler-step latency，输出
组件占比、主要瓶颈和显存；离线 replay 可进一步输出 TTFT、TPOT 与吞吐。

当前代码包含可运行建模底座与 same-runtime collector。`results/h20_smoke/` 是真实 H20 最小功能
验证，但 shape/repeat 数不足以作为正式精度 artifact；`examples/` 仍是 deterministic synthetic
fixture，只用于 CPU smoke。

当前推荐主路径是 `whitebox-moe-v3`：Attention、Router、Dispatch、Expert GEMM、Combine、Dense、
Transition 和 Fixed 的 operator response database 加执行 DAG 离散事件组合。旧 `ComponentModel`、routing
Ridge、residual 与 Basic RF 仅保留为历史 baseline。

## Architecture

```text
真实 vLLM control flow / StepExecutionPlan
                  ↓
Operator DB + Collective Timing Backend
                  ↓
rank/stream/resource ExecutionGraph
                  ↓
DiscreteEventEngine → critical path → TTFT / TPOT / Throughput
```

V3 不使用 RF、Ridge 或总时间 residual。`total_work_us` 保存全部节点工作量，`step_us` 来自资源约束
后的 makespan；可加和瓶颈占比使用 critical-path shares。TP=1 单 stream 时二者严格相等。Memory
当前使用 weights + KV + scheduled-token activation 的解析模型。

## Environment

```bash
conda activate vllm
cd /home/marongbai/profiling/moe-component-modeling/code
pip install -r requirements.txt
```

`requirements.txt` 不安装或覆盖 PyTorch/vLLM。当前 CPU 建模依赖 NumPy、scikit-learn 和 joblib；
后续真实 profiler 复用 conda `vllm` 中已有的 PyTorch/vLLM 0.18.1。

## Quick Smoke

V3 白盒 CPU 闭环：

```bash
PYTHONPATH=. python -m moe_perf_model make-whitebox-demo --output-dir examples/whitebox
PYTHONPATH=. python -m moe_perf_model train-whitebox \
  examples/whitebox/fine_raw_steps.jsonl --output results/whitebox-demo.joblib \
  --allow-synthetic-fixture
PYTHONPATH=. python -m moe_perf_model predict-whitebox \
  results/whitebox-demo.joblib examples/whitebox/whitebox_plan.json \
  --output results/whitebox-demo-profile.json
PYTHONPATH=. python -m moe_perf_model evaluate-whitebox \
  examples/whitebox/fine_raw_steps.jsonl --output results/whitebox-demo-eval.json \
  --allow-synthetic-fixture
```

`--allow-synthetic-fixture` 是强制 provenance 隔离；正式 CUDA Event 数据不使用该参数。

生成 synthetic fixtures：

```bash
PYTHONPATH=. python -m moe_perf_model make-demo --output-dir examples
```

输出：

- `examples/training_records.jsonl`
- `examples/predict_config.json`
- `examples/replay_trace.jsonl`
- `examples/SYNTHETIC_ONLY.txt`

训练 synthetic component artifact 必须显式授权：

```bash
PYTHONPATH=. python -m moe_perf_model train \
  examples/training_records.jsonl \
  --output results/demo-component.joblib \
  --allow-synthetic-fixture
```

输出 artifact 的 `label_source=synthetic_fixture`、`calibration_scope=synthetic-fixture`。省略授权参数
会直接失败，防止把 demo 当成 H20 实测。

预测单个配置：

```bash
PYTHONPATH=. python -m moe_perf_model predict \
  results/demo-component.joblib \
  examples/predict_config.json \
  --output results/demo-profile.json
```

`demo-profile.json` 包含 step/Attention/Expert/Dense/Other、residual、memory、component shares、
primary bottleneck、外推组件和 artifact provenance。

执行离线 scheduler replay：

```bash
PYTHONPATH=. python -m moe_perf_model replay \
  results/demo-component.joblib \
  examples/replay_trace.jsonl \
  --output results/demo-replay.json
```

结果包含逐 step performance profile、逐请求 TTFT/TPOT/E2E，以及 request/output throughput。

运行测试：

```bash
PYTHONPATH=. python -m pytest -q
```

## Formal H20 Workflow

先校验采集 manifest（不初始化 GPU）：

```bash
PYTHONPATH=. python -m moe_perf_model collect configs/h20_collection_smoke.json \
  --output-dir results/h20_smoke --dry-run
```

在指定空闲 H20 上执行真实采集：

```bash
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. python -m moe_perf_model collect \
  configs/h20_collection_smoke.json --output-dir results/h20_smoke
```

输出 `raw_steps.jsonl`、严格 `training_records.jsonl` 和 `collection_summary.json`。采集使用原生
vLLM 0.18.1 EngineCore，通过 `worker_extension_cls` 安全 RPC 安装 instance hook；不修改 vLLM
源码，也不开启 pickle fallback。固定 eager、TP=1、EP=1，并关闭 prefix cache/async scheduling。
manifest 的 batch 是提交请求数；训练中的 batch 是真实 scheduler step 的活跃请求数，两者可能因
continuous batching 不同，raw/summary 会同时保留。
prompt 使用固定种子的多样 token ID 序列，raw 中记录
`prompt_pattern=deterministic_diverse_uniform_v1`；常量 token prompt 会制造极端固定 routing，不能作为
正式 routing-aware anchors。

细粒度 collector 还输出 `execution_plans.jsonl`。当前 H20 路径要求 modular FusedMoE，并直接插装：

```text
gate + select_experts       → Router
kernel.impl._prepare        → Dispatch
kernel.impl._fused_experts  → Expert GEMM
kernel.impl._finalize       → Combine
```

若 runtime 选择 monolithic MoE kernel，安装阶段直接失败，不按比例伪造子阶段。每个 requested
workload 会先独立 warmup，再清空 worker records 后正式采集。

正式 V3 建表、预测与 holdout：

```bash
PYTHONPATH=. python -m moe_perf_model train-whitebox \
  results/h20_fine_smoke/raw_steps.jsonl \
  --audit-report results/nsight_audit/audit_report_diverse.json \
  --output results/h20_fine_smoke/whitebox.joblib
PYTHONPATH=. python -m moe_perf_model predict-whitebox \
  results/h20_fine_smoke/whitebox.joblib \
  results/h20_fine_smoke/execution_plans.jsonl \
  --plan-index 0 --output results/h20_fine_smoke/profile.json
PYTHONPATH=. python -m moe_perf_model evaluate-whitebox \
  results/h20_fine_smoke/raw_steps.jsonl \
  --output results/h20_fine_smoke/evaluation.json \
  --strict-output results/h20_fine_smoke/strict-generalization.json
PYTHONPATH=. python -m moe_perf_model report-surface \
  results/h20_fine_smoke/raw_steps.jsonl \
  --output-json results/h20_fine_smoke/bottleneck-surface.json \
  --output-markdown results/h20_fine_smoke/bottleneck-surface.md
```

evaluation 采用 requested-workload group leave-one-out；完整 step CUDA Event 只作为评价标签，不训练
总时间 residual。严格报告增加 unseen batch、length interval、large-region、Mixed phase holdout、
component MAPE、bottleneck accuracy 与 signature-distance 分桶。`report-surface` 按 requested batch
分别输出 Attention/Expert/Dense/Other 占比及 active experts、max/mean、p95/mean。

Nsight SQLite 审计与门禁：

```bash
PYTHONPATH=. python -m moe_perf_model audit-nsight REPORT.sqlite RAW_STEPS.jsonl \
  --output results/nsight_audit/audit_report.json
```

只有 `serial_validated=true` 的报告才能通过 `train-whitebox --audit-report` 将 artifact 标记为
`nsight_serial_validated`。

## Event Graph Engine

生成四类明确标为 synthetic 的多 rank 执行图：

```bash
PYTHONPATH=. python -m moe_perf_model make-event-demo \
  --scenario overlap \
  --output-graph results/event_demo/overlap_graph.json \
  --output-report results/event_demo/overlap_report.json
```

`scenario` 可选 `serialized`、`overlap`、`slow_rank`、`contention`。也可模拟外部生成的图：

```bash
PYTHONPATH=. python -m moe_perf_model simulate-event-graph GRAPH.json \
  --output EXECUTION_REPORT.json
```

报告包含 event start/end、critical path、total work、exposed/overlapped communication、resource
utilization、rank synchronization wait 和 bottleneck scores。`RuntimeEventRecorder` 已冻结未来 vLLM
phase/operator/collective 劫持所需的注册接口，但尚未实现占位 tensor adapter。

## Forward-Only Structural Predictor

结构预测器不模拟请求到达、排队或 scheduler，只接收一个已经 realized 的 MoE forward shape。它按真实
world size 建立显式 TP/EP process groups；TP group 与 EP group 可以 alias，因此 2×H20 的 TP2/EP2
不会被错误展开成 4 ranks。逐层 routing histogram 经 expert placement 转成 rank load，通信按 runtime
backend 生成 All-to-All 或 All-Gather/Reduce-Scatter，最后由离散事件引擎求 multi-rank critical path。

使用已有 H20 Operator DB 与 NCCL response table 预测一个 TP2/EP2 step：

```bash
PYTHONPATH=. python -m moe_perf_model predict-structural-forward \
  results/h20_sparse_anchor_v3_diverse/whitebox_event_v3.joblib \
  results/h20_tp2_ep2/combined_v3/multirank_steps.jsonl \
  results/h20_tp2_ep2/nccl_response_tp2.json \
  --world-size 2 --tp-size 2 --ep-size 2 --gpus-per-node 8 \
  --hidden-size 2048 --ep-communication-backend allgather_reducescatter \
  --rank-local-compute \
    results/h20_structural_compute_tp2_ep2_full_v1/rank_local_compute_v1.joblib \
  --plan-index 0 --output results/structural_forward_tp2_ep2_step0_v1.json
```

输出 schema 为 `structural-forward/v1`，包含 forward latency、per-rank finish、显存、critical path、
Attention/Expert/Communication/Synchronization 占比、完整 DAG 与 provenance。当前第一版固定
`serialized_collectives_no_overlap`。传入 `--rank-local-compute` 后，Attention、Router、Expert GEMM 使用
同 TP/EP identity 的 compute-exclusive CUDA Event 响应；Dispatch、Combine、Dense、Transition 仍为部分
覆盖，因此结果仍会明确返回 `formal_hardware_prediction=false`。NCCL 表是实测不代表整个预测已正式。

TP2 compute-exclusive 校准采集使用两个相同 workload manifests：

```bash
CUDA_VISIBLE_DEVICES=1,4 PYTHONPATH=. python -m moe_perf_model collect \
  configs/h20_tp2_ep1_structural_compute_v1.json \
  --output-dir results/h20_structural_compute_tp2_ep1_v1
CUDA_VISIBLE_DEVICES=1,4 PYTHONPATH=. python -m moe_perf_model collect \
  configs/h20_tp2_ep2_structural_compute_v1.json \
  --output-dir results/h20_structural_compute_tp2_ep2_v1
```

`rank_records[].layer_records[].rank_local_compute_us` 保存 Attention、Router、Expert GEMM 的 rank-local
CUDA Event anchors。TP2 Attention 若没有捕获 module 内 collective start，会标记 invalid 并阻止正式建表，
不会回退使用 inclusive module time。

从已采集 multi-rank records 建立 rank-local 响应库，并进行严格 workload holdout 评估：

```bash
PYTHONPATH=. python -m moe_perf_model train-rank-local-compute \
  results/h20_structural_compute_tp2_ep2_full_v1/multirank_steps.jsonl \
  --output results/h20_structural_compute_tp2_ep2_full_v1/rank_local_compute_v1.joblib

PYTHONPATH=. python scripts/evaluate_structural_pilot.py \
  --whitebox results/h20_sparse_anchor_v3_diverse/whitebox_event_v3.joblib \
  --nccl-response results/h20_tp2_ep2/nccl_response_tp2.json \
  --ep1-steps results/h20_structural_compute_tp2_ep1_full_v1/multirank_steps.jsonl \
  --ep2-steps results/h20_structural_compute_tp2_ep2_full_v1/multirank_steps.jsonl \
  --ep1-rank-compute results/h20_structural_compute_tp2_ep1_full_v1/rank_local_compute_v1.joblib \
  --ep2-rank-compute results/h20_structural_compute_tp2_ep2_full_v1/rank_local_compute_v1.joblib \
  --folds 5 --representative-repeats --strict-component-holdout \
  --calibration-budget 32 --max-rank-local-component-us 10000 \
  --output results/h20_structural_compute_full_v1/strict_evaluation_qc_v1.json
```

严格协议会在每个 fold 中先移除测试 workload，再重建 rank-local component DB；32 个 calibration
anchors 也只从训练 workload 按 feature-space farthest 规则选择，不读取候选 latency。重复运行先按 realized
request shape 取 forward median，评估键包含完整 request shape，防止同 step index 的不同 scheduler shape
被错误合并。`--max-rank-local-component-us` 是采集质量门禁：它只检查 rank-local CUDA Event，不读取
forward label 或预测误差；被拒记录及其最大 Event 会完整写入评估报告。

Summary-only 查询延迟可独立复测：

```bash
PYTHONPATH=. python scripts/benchmark_structural_query.py \
  --whitebox results/h20_sparse_anchor_v3_diverse/whitebox_event_v3.joblib \
  --nccl-response results/h20_tp2_ep2/nccl_response_tp2.json \
  --steps results/h20_structural_compute_tp2_ep2_full_v1/multirank_steps.jsonl \
  --rank-compute results/h20_structural_compute_tp2_ep2_full_v1/rank_local_compute_v1.joblib \
  --ep-size 2 --iterations 50 \
  --output results/h20_structural_compute_full_v1/query_benchmark_ep2.json
```

当前严格 QC 结果与边界汇总在
`results/h20_structural_compute_full_v1/RESULTS.md`。

响应库可审计导出：

```bash
PYTHONPATH=. python -m moe_perf_model inspect-whitebox \
  results/h20_fine_smoke/whitebox.joblib \
  --output results/h20_fine_smoke/operator-database.json
```

已有 raw records 可独立重做 schema 转换与汇总，不占用 GPU：

```bash
PYTHONPATH=. python -m moe_perf_model convert-collection \
  configs/h20_collection_smoke.json results/h20_smoke/raw_steps.jsonl \
  --output-dir results/h20_smoke
```

正式数据准备为一行一个 `TrainingRecord`：

```json
{
  "config": {
    "model": {},
    "hardware": {},
    "workload": {},
    "routing": {},
    "runtime": {}
  },
  "labels": {
    "label_source": "cuda_event",
    "runtime": "vllm",
    "vllm_version": "0.18.1",
    "precision": "bfloat16",
    "attention_us": 0.0,
    "expert_us": 0.0,
    "dense_us": 0.0,
    "other_us": 0.0,
    "step_us": 0.0,
    "peak_memory_bytes": 0
  }
}
```

正式训练不使用 synthetic flag：

```bash
PYTHONPATH=. python -m moe_perf_model train \
  data/h20_vllm_component_records.jsonl \
  --output results/h20-qwen3-component.joblib
```

训练器会拒绝：

- 非 CUDA Event label；
- 混合 label source；
- 不一致的 model/hardware/runtime/version/precision；
- routing token 不守恒；
- `components > step`；
- TP/EP>1。

完整字段定义见 [implementation.md](../docs/implementation.md)。

## Commands

### Validate Config

```bash
PYTHONPATH=. python -m moe_perf_model validate-config CONFIG.json
```

验证所有跨字段约束，成功时打印 stable `config_id`。

### Train Component Model

```bash
PYTHONPATH=. python -m moe_perf_model train RECORDS.jsonl \
  --output ARTIFACT.joblib \
  --model component
```

- `RECORDS.jsonl`：同 runtime component records。
- `--output`：joblib artifact 路径。
- `--model`：默认 `component`；也可用 `basic-rf` 训练黑盒 baseline。
- `--allow-synthetic-fixture`：只允许 CPU demo 使用，正式训练禁止。
- `--seed`：Basic RF random seed，默认 0。

### Predict Profile

```bash
PYTHONPATH=. python -m moe_perf_model predict ARTIFACT.joblib CONFIG.json \
  --output PROFILE.json
```

只接受 component artifact。Basic RF 没有可信组件归因，因此此命令显式拒绝 Basic artifact。

### Replay Trace

```bash
PYTHONPATH=. python -m moe_perf_model replay ARTIFACT.joblib TRACE.jsonl \
  --output SUMMARY.json
```

Trace 每行包含严格递增的 `step_id`、完整 config 和 request events。第一版是串行离线 replay；
同一步产生的 token 统一在 step end 可见。

## Basic RF Baseline

```bash
PYTHONPATH=. python -m moe_perf_model train RECORDS.jsonl \
  --output results/basic-rf.joblib \
  --model basic-rf \
  --seed 7
```

Basic RF API 提供 `predict_step_us(config)`，只预测总 step latency。它不输出 Attention/Expert
占比，也不能进入 component replay；这是有意的语义隔离，避免用黑盒特征重要性伪装瓶颈归因。

## White-box Replay

`replay-whitebox` 接受 JSONL，每行包含 `step_id`、完整 `plan` 和 `request_events`：

```bash
PYTHONPATH=. python -m moe_perf_model replay-whitebox \
  results/h20_fine_smoke/whitebox.joblib TRACE.jsonl \
  --output results/whitebox-replay.json
```

它沿用 step-end token visibility，输出 TTFT、TPOT、吞吐、逐 step 七阶段 profile 与显存。

## vLLM Forward Hijack

V5 launcher 保留真实 vLLM 0.18.1 scheduler、continuous batching、KV block 管理与 request lifecycle，
只在 executor forward 边界查询实测 TP2/EP2 fixed-configuration oracle：

```bash
PYTHONPATH=.:/home/marongbai/profiling/llm-emu \
  /home/marongbai/miniconda3/envs/vllm/bin/python \
  scripts/run_vllm_forward_hijack.py \
  --prompt-lengths 8,8 --max-tokens 2
```

控制流 frontend 固定为 uniproc vLLM，以便原位注入 hook；forward latency artifact 来自真实 2×H20、
TP2/EP2、BF16 eager、allgather-reducescatter 采集。launcher 使用 CUDA mock，不加载权重；默认不生成
昂贵的 attribution graph。TP 和 EP 使用同一组 2 ranks，world size=2，不采用 `TP×EP` ranks。

正式采集与训练：

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m moe_perf_model collect \
  configs/h20_tp2_ep2_adaptive_v2.json \
  --output-dir results/h20_tp2_ep2/adaptive_v2

PYTHONPATH=. python -m moe_perf_model train-fixed-config \
  results/h20_tp2_ep2/adaptive_v2/multirank_steps.jsonl \
  --method routing_rf \
  --nccl-response results/h20_tp2_ep2/nccl_response_tp2.json \
  --output results/h20_tp2_ep2/adaptive_v2/fixed_config_routing_rf.joblib
```

受控 Mixed/大 Prefill 扩展和预算评估：

```bash
CUDA_VISIBLE_DEVICES=2,3 PYTHONPATH=. python -m moe_perf_model collect \
  configs/h20_tp2_ep2_controlled_mixed_v3.json \
  --output-dir results/h20_tp2_ep2/controlled_mixed_v3

PYTHONPATH=. python -m moe_perf_model merge-multirank \
  results/h20_tp2_ep2/adaptive_v2/multirank_steps.jsonl \
  results/h20_tp2_ep2/controlled_mixed_v3/multirank_steps.jsonl \
  --output results/h20_tp2_ep2/combined_v3/multirank_steps.jsonl

PYTHONPATH=. python -m moe_perf_model evaluate-budget-curve \
  results/h20_tp2_ep2/combined_v3/multirank_steps.jsonl \
  --budgets 16 32 64 96 128 187 295 --method routing_rf \
  --output results/h20_tp2_ep2/combined_v3/budget_curve_routing_rf.json
```

`adaptive_v2` 的 187-anchor 局部 oracle 为 1.73% MAPE / 5.64% p95。`combined_v3` 扩大到 601 个
实测 steps、295 个唯一 anchors（Prefill 62 / Decode 185 / Mixed 48）；strict workload LOO 为
2.30% MAPE / 10.03% p95。可信域接受 85.0% steps，域内为 2.16% / 7.44%；Prefill p95 由
15.85% 降至 10.30%，Mixed p95 由 20.16% 降至 17.56%。扩大 workload surface 后总体 p95 上升，
不能把它解释为同一测试集上的直接退化或提升。

187-anchor budget 在 295-anchor surface 的 108 个未采样点上为 2.89% MAPE / 11.18% p95；其中
92.6% 位于可信域，域内为 2.54% / 9.53%，最慢 10% workload recall 为 93.3%。当前只有一个
deployment，configuration ranking 明确为 `not_applicable`。

Observed-error acquisition 使用固定 16-point farthest warm start，之后只读取已测 anchors 的局部 LOO
residual、latency disagreement 和 feature distance。五折 requested-workload holdout 命令：

```bash
for fold in 0 1 2 3 4; do
  PYTHONPATH=. python -m moe_perf_model evaluate-budget-curve \
    results/h20_tp2_ep2/combined_v3/multirank_steps.jsonl \
    --budgets 16 32 64 96 128 187 --method routing_rf --strategy both \
    --protocol fixed_workload_holdout --holdout-modulus 5 --holdout-fold "$fold" \
    --output "results/h20_tp2_ep2/combined_v3/holdout_strategy_v3_fold${fold}.json"
done

PYTHONPATH=. python -m moe_perf_model aggregate-budget-folds \
  results/h20_tp2_ep2/combined_v3/holdout_strategy_v3_fold{0,1,2,3,4}.json \
  --output results/h20_tp2_ep2/combined_v3/holdout_strategy_v3_aggregate.json
```

固定 holdout 中测试 workload 的 features 与 labels 均不进入 acquisition。五折均值如下：

| Anchors | Farthest MAPE/p95 | Observed-error MAPE/p95 | Observed-error Top-k | Supported |
|---:|---:|---:|---:|---:|
| 16 | 9.70% / 50.50% | 9.70% / 50.50% | 43.3% | 49.3% |
| 32 | 9.81% / 59.58% | 4.91% / 34.96% | 69.5% | 78.8% |
| 64 | 7.19% / 43.13% | 3.47% / 18.48% | 96.7% | 87.7% |
| 96 | 4.53% / 27.11% | **2.19% / 10.29%** | **97.1%** | **88.8%** |
| 128 | 3.87% / 21.83% | 2.19% / 10.17% | 97.1% | 88.5% |
| 187 | 2.37% / 10.17% | 2.25% / 10.19% | 97.1% | 85.4% |

按 MAPE≤3%、p95≤15%、Top-k≥95%、supported≥80% 且不劣于 baseline 的冻结规则，推荐最小预算为
**96 anchors**。187 点时 observed-error 不再稳定优于 baseline，因此不继续在同一五折上调参。

## Second Multi-GPU Configuration

第二个正式 identity 为同机 `2×H20 + TP2/EP1 + BF16 eager`。使用与 EP2 相同的两个 requested
workload suites，在 GPU 2/3 上实测 612 steps、243 unique anchors，并独立训练 artifact：

```bash
PYTHONPATH=. python -m moe_perf_model train-fixed-config \
  results/h20_tp2_ep1/combined_v3/multirank_steps.jsonl \
  --method routing_rf --nccl-response results/h20_tp2_ep2/nccl_response_tp2.json \
  --output results/h20_tp2_ep1/combined_v3/fixed_config_routing_rf.joblib
```

独立 workload LOO 为 2.04% MAPE / 8.71% p95，可信域覆盖 84.6%，域内为 2.03% / 7.80%。冻结的
observed-error policy 在五折 fixed-workload holdout 上推荐 **64 anchors**：2.47% MAPE、9.26% p95、
97.1% Top-k recall、87.8% coverage。这里不混入任何 EP2 anchor。

成对实测诊断：

```bash
PYTHONPATH=. python -m moe_perf_model compare-fixed-configs \
  results/h20_tp2_ep2/combined_v3/multirank_steps.jsonl \
  results/h20_tp2_ep1/combined_v3/multirank_steps.jsonl \
  --output results/tp2_ep2_vs_tp2_ep1_paired.json
```

423 个 request-level realized shape 完全一致的 paired steps 中，TP2/EP1 相对 TP2/EP2 的 latency ratio
中位为 0.961：Prefill 0.916、Decode 0.965、Mixed 0.952，96.9% paired steps 上 EP1 更快。该结果只说明
当前 kernel/runtime 下的 matched-step 差异，不等于 TTFT/throughput 配置推荐，也不是跨配置预测精度。

## Cross-configuration Sparse Calibration

TP2/EP2 完整 oracle 可作为 source timing signal，再用少量 TP2/EP1 anchors 学习 phase-strict
`target/source` ratio。两个配置不会混入同一个 fixed-config oracle；跨 EP distance 也不使用口径变化的
EP-rank load 三个特征。

```bash
PYTHONPATH=. python -m moe_perf_model evaluate-cross-config-calibration \
  results/h20_tp2_ep2/combined_v3/multirank_steps.jsonl \
  results/h20_tp2_ep1/combined_v3/multirank_steps.jsonl \
  results/h20_tp2_ep2/combined_v3/fixed_config_routing_rf.joblib \
  --budgets 8 16 32 --folds 5 \
  --target-reference results/h20_tp2_ep1/combined_v3/holdout_strategy_aggregate.json \
  --output results/h20_tp2_ep1/combined_v3/cross_config_calibration_5fold.json

PYTHONPATH=. python -m moe_perf_model train-cross-config-calibration \
  results/h20_tp2_ep2/combined_v3/multirank_steps.jsonl \
  results/h20_tp2_ep1/combined_v3/multirank_steps.jsonl \
  results/h20_tp2_ep2/combined_v3/fixed_config_routing_rf.joblib \
  --budget 32 --method local_ratio_idw \
  --output results/h20_tp2_ep1/combined_v3/cross_config_local_ratio_b32.joblib
```

严格 requested-workload 五折 holdout 的 shape-aligned test 结果如下；数值为五折 mean，target-only 使用与
transfer 完全相同的 unique target anchors 和同 anchor 的重复测量：

| Target anchors | Zero-shot source | Phase scale | Local ratio IDW | Target-only | Local coverage |
|---:|---:|---:|---:|---:|---:|
| 8 | 6.45% / 21.05% | 3.79% / 16.19% | 3.85% / 16.12% | 7.06% / 50.71% | 19.2% |
| 16 | 6.45% / 21.05% | 3.62% / 15.86% | **2.78% / 12.26%** | 8.78% / 43.08% | 59.8% |
| 32 | 6.45% / 21.05% | 3.64% / 16.16% | **2.22% / 8.43%** | 9.12% / 53.03% | 87.4% |

表中每格为 MAPE / p95 APE。按 MAPE≤3%、p95≤15%、coverage≥80% 的规则推荐 **32 target anchors**；
它比独立 target oracle 的冻结预算 64 anchors 减半。独立 64-anchor reference 为 2.47% / 9.26%，但其
测试 surface 更广，不能当作同一 paired test 的直接 delta。校准查询需要 shape-aligned source scheduler/
routing signal；若调用方只有 target shape 而没有 source replay，工具会按接口契约拒绝静默替代。

## Service-level Replay and Configuration Ranking

服务评估不再按 step 对齐。TP2/EP2 与 TP2/EP1 分别 replay 自己真实采集的 scheduler trajectory，只在
requested workload/repeat 层对齐；不同 step 数、chunked Prefill 或 Mixed 顺序全部保留。五折内重新使用
32 个 train-target anchors 校准，held-out target workloads 不参与选点或训练：

```bash
PYTHONPATH=. python -m moe_perf_model evaluate-service-ranking \
  results/h20_tp2_ep2/combined_v3/multirank_steps.jsonl \
  results/h20_tp2_ep1/combined_v3/multirank_steps.jsonl \
  results/h20_tp2_ep2/combined_v3/fixed_config_routing_rf.joblib \
  results/h20_tp2_ep1/combined_v3/fixed_config_routing_rf.joblib \
  --target-calibration-budget 32 --folds 5 \
  --ttft-slo-ms 200 --tpot-slo-ms 80 --e2e-slo-ms 500 \
  --output results/service_ranking_tp2_ep2_vs_tp2_ep1_b32.json
```

129 个 workload/repeat trajectories 全部请求级兼容，其中 **37 个 step sequence 不同**。32-anchor
target calibration 的服务指标误差与配置排序结果：

| 服务指标 | Target MAPE / p95 APE | Top-1 配置准确率 | Mean optimality gap |
|---|---:|---:|---:|
| Mean TTFT | 3.23% / 12.16% | **98.45%** | 0.028% |
| Mean TPOT | 1.63% / 4.48% | **95.35%** | 0.162% |
| Output-token throughput | 1.80% / 5.16% | **97.67%** | 0.061% |
| p95 E2E | 1.84% / 5.31% | **97.67%** | 0.083% |

实测 forward-time replay 中，TP2/EP1 的 throughput 在 91.5% trajectories 更高，throughput ratio
中位为 1.064；mean TTFT、mean TPOT 和 p95 E2E ratio 中位分别为 0.919、0.960、0.941。因此单 step
更快在当前静态 workload suite 上确实转化成服务指标优势，但不是所有 workload 都成立。不同 step sequence
的 37 个 trajectories 上 throughput MAPE 为 1.62%，没有因为 shape mismatch 恶化。

校准路径平均 supported-step coverage 为 83.8%；可信域外 candidate 仅用于本离线评估，正式调用仍拒绝。
这里的 SLO 是为管线验证显式给定的示例阈值，不是业务 SLO。trajectory 与 step labels 来自真实 scheduler
和多 rank CUDA Event，但虚拟时间只累计 GPU forward，不含 scheduler CPU、output processor、动态 arrival
和排队，因此结果不是完整线上 wall-clock E2E。

## Candidate Table and Conservative Pruning

V11 将服务预测转换为统一 candidate table，并按 workload 执行 Feasibility → Confidence → SLO → Capacity
→ conservative Pareto。未知状态不会被当作低性能剪掉：

```bash
PYTHONPATH=. python -m moe_perf_model build-candidate-table \
  results/service_ranking_tp2_ep2_vs_tp2_ep1_b32.json \
  results/h20_tp2_ep2/combined_v3/multirank_steps.jsonl \
  results/h20_tp2_ep1/combined_v3/multirank_steps.jsonl \
  --gpu-memory-bytes 150754820096 --available-gpus 2 \
  --source-cost-units 2 --target-cost-units 2 \
  --output results/candidate_table_tp2_ep2_tp2_ep1_v1.json

PYTHONPATH=. python -m moe_perf_model prune-candidates \
  results/candidate_table_tp2_ep2_tp2_ep1_v1.json \
  --policy configs/pruning_policy_conservative_v1.json \
  --output results/pruning_tp2_ep2_tp2_ep1_conservative_v1.json
```

当前 2 candidates × 43 workloads 的 86 个决策为：

| 状态 | 数量 | 含义 |
|---|---:|---|
| retained | 68 | 可进入后续搜索 |
| needs_profile | 13 | TP2/EP1 coverage < 80%，进入补采队列 |
| pruned | 5 | 4 个示例 SLO violation，1 个区间 Pareto dominance |
| needs_replica_evaluation | 0 | arrival rate 未提供，Capacity 未评估 |
| needs_validation | 0 | 两配置均已真机成功启动且显存可行 |

没有配置被全局删除。只有 batch4×prompt896 的 `w27` 上，TP2/EP1 在 TTFT、TPOT、p95 E2E、throughput
及相同 normalized cost=2 下以不重叠区间保守支配 TP2/EP2。另 4 个 TP2/EP2 workload 被示例 SLO 风险
策略删除，其中 3 个 lower bound 也超 SLO，1 个仅 upper bound 超限。成本 2 是显式传入的 GPU-count
归一化单位，不是美元价格；真实 cost 与 arrival rate 未提供时保持 `not_evaluated`。

### Candidate Data Format

`config-candidate-table/v1`：

```text
hardware_pool
workloads[workload_id, request_shape, arrival_rate_rps, slo, weight]
candidates[candidate_id, config_identity, parallel, launch_status, cost_units,
           memory_bytes_per_gpu_worst_case,
           predictions[workload_id][metrics(point/lower/upper), coverage, memory]]
```

`candidate-pruning-report/v1` 输出逐规则 reason、workload-conditioned retained/pruned sets，以及
`profile_queue`、`replica_evaluation_queue` 和 `validation_queue`，可直接作为后续搜索器的输入边界。

NCCL response 表在空闲 GPU 2、3（同机 H20、NV18）采集；256 MiB median 为 All-Reduce 885.1 μs、
All-Gather 1023.1 μs、Reduce-Scatter 1030.5 μs、All-to-All 476.3 μs。forward anchors 在此前空闲的
GPU 0、1 上采集，artifact 分别保存两次采集的实际 UUID。

### Data Format

`trace.jsonl` 每行是一个真实 scheduler step，顶层字段为：

```text
schema_version, step_index, scheduler_context,
routing_provenance, timing_provenance, parallel_layout,
execution_plan, predicted_step_us, latency_prediction,
execution_report, fake_output, formal
```

`scheduler_context` 来自 realized `SchedulerOutput`；`routing_provenance=profile_replay` 表示逐层 histogram
来自最近 H20 anchor 并按 routed-token 守恒缩放，不是当前 token 的真实 router 输出。因此默认 hijack
trace 仍为 `formal=false`；若调用方提供同口径真实/预测 routing，fixed-config timing artifact 本身是
正式多卡实测。`execution_report` 默认是 `null`，只有显式请求辅助归因图时才生成。

fixed-config prediction 额外返回：

```text
step_us, candidate_step_us, support_status, needs_profile, support_reason,
nearest_distance, coverage_threshold, extrapolated, neighbors
```

超出 phase feature bounds 或最近 anchor 超过 coverage threshold 时，正式 `step_us=null` 且
`support_status=not_predictable`；只有离线评估显式允许读取 `candidate_step_us`。NCCL response 只作为
provenance/归因信息附着在 artifact，不会再次加入 forward 总时间。

cross-config prediction 额外返回：

```text
target_us, candidate_target_us, source_us, ratio, support_status, needs_profile,
support_reason, nearest_distance, coverage_threshold, extrapolated, prediction_wall_us
```

## Known Limits

- 已有 25 requested anchors、67 realized steps、长度 8–8192 的 H20 artifact；但 large-region MAPE
  仍为 31.32%，尚未达到冻结性能预测器的标准。
- fixed-config oracle 不做正式外推；可信域外返回 `not_predictable`。当前阈值是 anchor 最近邻距离 p95
  的几何规则，不是统计置信区间，仍需用 error-aware calibration 改进。
- TP2/EP2 已有正式 NCCL 与 forward artifacts；其他 TP/EP、CUDA Graph、all-to-all backend 仍没有正式
  表，配置不匹配时必须拒绝。
- 旧离线 replay 尚不模拟 admission、preemption、prefix cache、continuous batching；V4 forward hijack 已
  使用真实 vLLM scheduler/control flow，但目前仅支持 in-process uniproc runtime，分布式 ranks 是虚拟图。
- 当前 vLLM 安装缺少 EP2 对应的 `E=64,N=768,device_name=NVIDIA_H20-3e.json` fused-MoE tuning config；
  artifact 固定记录当前 runtime-selected default kernel，更换 tuning config 后必须重新采集。
- combined-v3 的 Mixed p95 仍为 17.56%，Prefill p95 为 10.30%；仅按 shape/routing farthest-point 选择
  在 16→32 anchors 时 RF 误差非单调，下一轮需要稳定的小样本 backend 或 uncertainty/error-aware 选点。
- budget report 的 profiling time 是被选 anchor median forward CUDA time 之和，不含模型加载、warmup、
  scheduler 和进程启动；不能当作端到端采集 wall time。
- 当前 service ranking 是静态批次的真实 scheduler trajectory + forward-time replay；尚未采集动态 arrival、
  queueing、scheduler CPU 和 output processor wall time，SLO 结论不能直接外推到线上请求流。
- 当前剪枝结果中的 SLO 是示例阈值，cost 是 GPU-count normalized units，arrival rate 未配置；它验证了
  规则和数据流，但在接入真实业务 policy 前不能作为生产部署决策。
- 单卡 V3 已通过多样 prompt 的 Nsight audit，正式 artifact 标记为 `nsight_serial_validated`；更换 runtime、
  CUDA Graph 或 TP/EP 配置后必须重新审计，不能复用本次结论。
## Dense–MoE Bottleneck Transfer

Qwen3-8B Dense 与 Qwen3-30B-A3B MoE 可以在同一 H20 workload grid 下统一为
`Attention / FFN / Other` 口径比较：

```bash
CUDA_VISIBLE_DEVICES=5 PYTHONPATH=. python -m moe_perf_model collect \
  configs/h20_qwen3_8b_dense_transfer_v1.json \
  --output-dir results/h20_qwen3_8b_dense_transfer_v1

PYTHONPATH=. python -m moe_perf_model compare-dense-moe \
  results/h20_qwen3_8b_dense_transfer_v1/raw_steps.jsonl \
  results/h20_sparse_anchor_v3_diverse/raw_steps.jsonl \
  --output-json results/h20_qwen3_8b_dense_transfer_v1/dense_moe_comparison.json \
  --output-markdown results/h20_qwen3_8b_dense_transfer_v1/RESULTS.md
```

Dense 侧劫持每层完整 `Qwen3Attention` 与 `Qwen3MLP`；MoE 侧 FFN 对应完整
`Qwen3MoeSparseMoeBlock`。报告同时给出 exact realized-step alignment coverage 和 requested
workload/phase 聚合，scheduler trajectory 不一致的 step 不会被强制配对。

## Multi-GPU Dense–MoE Transfer

2×H20 的正式对照使用 Dense TP2 与 MoE TP2/EP2，每个基础 anchor 10 次，16K/32K 长上下文 anchor 5 次。
最终报告位于 `results/h20_multigpu_dense_moe_transfer_v1/FINAL_RESULTS.md`。通信 kernel 审计可复现为：

```bash
PYTHONPATH=. python -m moe_perf_model.kernel_trace_audit \
  --dense-dir results/h20_multigpu_dense_moe_transfer_v1/torch_trace/dense \
  --moe-dir results/h20_multigpu_dense_moe_transfer_v1/torch_trace/moe \
  --dense-config configs/h20_qwen3_8b_dense_tp2_nsight_v1.json \
  --moe-config configs/h20_qwen3_30b_a3b_tp2_ep2_nsight_v1.json \
  --dense-steps results/h20_qwen3_8b_dense_tp2_transfer_v1/multirank_steps.jsonl \
  --moe-steps results/h20_qwen3_30b_a3b_tp2_ep2_transfer_v1/multirank_steps.jsonl \
  --output-dir results/h20_multigpu_dense_moe_transfer_v1
```

不要使用 high-level collective API CUDA Event 生成通信瓶颈标签；异步返回会将等待错误归给调用组件。
kernel audit 的 `rank-min comm` 是 service-time 下界，不是可与组件时间相加的 exposed latency。
