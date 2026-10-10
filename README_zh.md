<p align="center">
  <img src="assets/openesm-github-title.png" alt="OpenESM" width=800>
</p>

<p align="center">
  <a href="https://github.com/datamllab/openesm"><img src="https://img.shields.io/badge/GitHub-181717?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"/></a>
  <a href="https://huggingface.co/collections/guan-wang/openesm"><img src="https://img.shields.io/badge/HuggingFace-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face"/></a>
</p>

<p align="center">
  <strong>用于训练和扩展 Energy-Steered Models 的开源代码</strong>
</p>

<p align="center">
  <a href="README.md">English</a> · 简体中文
</p>

---

## 🎉 更新

- **2026-10-07** — ✨✨ 完整代码库发布。

## 🚀 快速开始

一次性安装 GPU、Hugging Face 和网页聊天所需依赖：

```bash
uv sync --extra gpu --extra hf --extra web
source .venv/bin/activate
```

下载并分词训练数据（首次训练前运行一次，需联网）。默认准备 `climbmix`、`dclm`、
`fineweb` 全部三个，用 `--datasets` 只准备其中一部分：

```bash
bash runs/prepare_data.sh                  # 默认：climbmix + dclm + fineweb
bash runs/prepare_data.sh --datasets dclm  # 只准备指定数据集
```

设置训练 token 预算后启动完整训练：

```bash
TARGET_TOTAL_TOKENS=100000000 bash runs/train.sh
```

> 数据准备需要联网，请在联网的 cpu-worker 上先执行一次；训练本身不联网，
> 但必须在已经准备好数据的机器上运行。

## 📈 Scaling Law

在 DCLM 上，增加训练 token 数和模型规模可降低验证 BPB。在 ClimbMix、
DCLM 和 FineWeb 上，零样本准确率随模型规模提升，也会随训练 token 数增加。

<table>
  <tr>
    <td align="center" width="50%"><img src="figs/scaling_pretrain_dclm_validation_bpb.png" width="100%" alt="DCLM 验证 BPB 与训练 token 数的关系"><br>DCLM 预训练</td>
    <td align="center" width="50%"><img src="figs/dclm_best_val_bpb_vs_depth.png" width="100%" alt="DCLM 验证 BPB 与模型规模的关系"><br>参数规模扩展</td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="figs/qa_acc_vs_model_size.png" width="100%" alt="不同训练数据集上的平均零样本准确率与模型规模"><br>零样本参数规模扩展</td>
    <td align="center" width="50%"><img src="figs/zero_shot_acc_vs_training_tokens.png" width="100%" alt="零样本准确率与训练 token 数的关系"><br>零样本 token 扩展</td>
  </tr>
</table>

可使用 `configs/train.yaml` 作为默认配置，并通过 `TARGET_TOTAL_TOKENS`
设置训练预算。

## ⚙️ 预训练 Checkpoint

预训练模型发布在
[OpenESM Hugging Face collection](https://huggingface.co/collections/guan-wang/openesm)
中。Collection 包含基于 DCLM、FineWeb 和 ClimbMix 训练的 160M、520M 和 1B
参数规模模型；一个 160M OWT 模型；以及一个名为 `ESM-FineWeb-1B-CHAT` 的
FineWeb SFT 模型。

可以直接通过 Hugging Face 模型 ID 加载已发布模型。首次加载会自动下载
并缓存权重和 tokenizer：

```python
import torch
from esm.modeling_esm import load_checkpoint

model_id = "guan-wang/ESM-FineWeb-1B-CHAT"
model, tokenizer, hparams, device = load_checkpoint(model_id, device="cuda")
tokens = tokenizer.encode("Hello, ESM.", append=tokenizer.get_bos_token_id())
input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
with torch.no_grad():
    logits = model(input_ids)
print(logits.shape)
```

也可以将模型 ID 替换为 collection 中的其他模型仓库。

## 💬 Chat Demo

<p align="center">
  <img src="figs/chat.jpg" alt="ESM Chat 网页界面" width="850">
</p>

使用 Hugging Face 上发布的 SFT 模型启动网页聊天：

```bash
bash runs/chat.sh --web \
  --checkpoint guan-wang/ESM-FineWeb-1B-CHAT \
  --host 0.0.0.0 \
  --port 8000
```

在运行服务器的机器上用浏览器打开
[http://localhost:8000](http://localhost:8000)。如果服务运行在远程 GPU 节点，
请使用集群端口转发将 8000 端口映射到本机。

CPU 和 CUDA 的 PyTorch 软件源配置在 `pyproject.toml` 中。每个环境选择
一个 extra 即可。

## 运行

`runs/` 中的 shell 启动脚本保持轻量，并且始终从项目根目录运行，因此在
本地 shell 和分布式任务中都能稳定解析相对路径。

```bash
# 预训练：训练长度由 TARGET_TOTAL_TOKENS 决定。
TARGET_TOTAL_TOKENS=100000000 bash runs/train.sh

# 微调：从 Hugging Face 上发布的模型初始化权重。
TARGET_TOTAL_TOKENS=100000000 bash runs/sft.sh \
  --dataset_name esm_sft \
  --finetuning_model_ckpt guan-wang/ESM-DCLM-1B

# Position-wise BPB evaluation。
MODEL_DIR="$(python -c 'from huggingface_hub import snapshot_download; print(snapshot_download(repo_id="guan-wang/ESM-DCLM-1B"))')"
bash runs/eval.sh \
  --checkpoint guan-wang/ESM-DCLM-1B \
  --dataset dclm \
  --data-dir /path/to/data \
  --tokenizer-path "${MODEL_DIR}" \
  --output_root . \
  --run_name eval-dclm

# QA evaluation。
bash runs/qa.sh \
  --checkpoint guan-wang/ESM-DCLM-1B \
  --tokenizer-path "${MODEL_DIR}" \
  --eval-bundle /path/to/eval_bundle \
  --output_root . \
  --run_name qa-dclm

# 交互式生成。
bash runs/chat.sh --checkpoint guan-wang/ESM-FineWeb-1B-CHAT
```

Chat、验证集评估、QA 评估和 SFT 初始化均可直接使用 Hugging Face 模型 ID。BPB
评估还需传入本地下载目录以读取 `token_bytes.pt`。Zero-shot 评估目前仍要求本地
Lightning checkpoint。

训练读取 `configs/train.yaml`，评估读取 `configs/eval.yaml`。显式命令行参数
优先于 YAML 配置。训练长度由 `TARGET_TOTAL_TOKENS` 控制。有效的梯度累积
会根据 `global_batch_size`、`device_batch_size` 和分布式 world size 自动计算，
因此 `max_steps` 和 `accumulate_grad_batches` 不是独立的训练控制参数。

每次运行使用相同的输出目录结构：

```text
outputs/<run-name>/
├── checkpoints/
├── logs/
│   └── wandb/
└── results/
```

数据和 tokenizer 文件不包含在代码仓库中，需要通过配置或命令行传入路径。
支持的预训练和 BPB 数据集包括 DCLM、FineWeb、ClimbMix 和 OWT。

## 从 Hugging Face 加载

`load_checkpoint()` 接受 Hugging Face 模型 ID，并自动下载 Transformers 权重
和 tokenizer。例如：

```python
import torch
from esm.modeling_esm import load_checkpoint

model, tokenizer, hparams, device = load_checkpoint(
    "guan-wang/ESM-OWT-160M", device="cuda"
)
tokens = tokenizer.encode("hello", append=tokenizer.get_bos_token_id())
input_ids = torch.tensor(
    [tokens], dtype=torch.long, device=device
)
with torch.no_grad():
    logits = model(input_ids)
print(logits.shape)
```

进行 BPB 评估时，需将同一模型仓库下载到本地，并把该目录传给
`--tokenizer-path`；仓库中包含指标计算所需的 tokenizer byte table。加载器也
支持本地 Transformers 目录和本项目训练产生的 Lightning checkpoint。

## 仓库结构

```text
openesm/
├── assets/                                  # 仓库品牌素材
│   └── openesm-github-title.png             # README 标题横幅
├── configs/                                 # 默认运行配置
│   ├── eval.yaml                 # 评估默认配置
│   └── train.yaml                # 预训练默认配置
├── esm/                                     # 模型、数据和训练库
│   ├── __init__.py                           # 包初始化
│   ├── common.py                             # 共享常量和工具函数
│   ├── config.py                             # 训练与评估配置解析
│   ├── configuration_esm.py                  # Transformers 模型配置
│   ├── core_eval.py                          # Core 基准评估
│   ├── dataloader.py                         # 流式数据加载器
│   ├── dataset.py                             # 通用数据集工具
│   ├── dataset_sft.py                        # 监督微调数据集
│   ├── disk_aware_checkpoint.py              # checkpoint 保存与加载工具
│   ├── logger.py                              # 训练指标和产物日志
│   ├── metrics.py                             # 损失与 BPB 指标
│   ├── modeling_esm.py                        # ESM 架构、加载和推理
│   ├── optim.py                               # Muon、AdamW 优化器和学习率调度
│   ├── pretrain_dataset.py                    # 预训练数据管线
│   ├── tokenizer.py                           # ESM 分词器及加载逻辑
│   └── trainer.py                             # Lightning 训练与评估模块
├── figs/                                    # README 图片和演示截图
│   ├── chat.jpg                               # 网页聊天界面截图
│   ├── dclm_best_val_bpb_vs_depth.png          # DCLM 参数规模扩展图
│   ├── qa_acc_vs_model_size.png                # 不同模型规模的零样本准确率
│   ├── scaling_pretrain_dclm_validation_bpb.png # DCLM 验证 BPB 与 token 数关系图
│   └── zero_shot_acc_vs_training_tokens.png    # 零样本准确率与 token 数关系图
├── runs/                                    # 常用工作流的 Shell 入口
│   ├── chat.sh                   # 启动交互式聊天
│   ├── eval.sh                   # 执行验证集评估
│   ├── prepare_data.sh           # 准备并分词数据集
│   ├── qa.sh                     # 执行问答任务评估
│   ├── sft.sh                    # 从 checkpoint 进行微调
│   ├── train.sh                  # 预训练 ESM 模型
│   └── zeroshot.sh               # 执行零样本基准测试
├── scripts/                                 # Python 命令行入口
│   ├── chat.py                   # 提供终端或网页聊天服务
│   ├── eval.py                   # 评估验证损失和 BPB
│   ├── export_hf.py              # 将 checkpoint 导出为 Transformers 格式
│   ├── prepare_data.py           # 构建训练和评估数据
│   ├── qa.py                     # 评估问答任务集合
│   ├── sft.py                    # 执行监督微调
│   ├── train.py                  # 执行预训练
│   ├── zeroshot.py               # 执行零样本基准评估
│   └── zeroshot_datasets.py      # 定义零样本基准数据集
├── tasks/                                   # 评估用数据集和任务适配器
│   ├── common.py                             # 共享任务接口和工具
│   ├── customjson.py                         # 加载自定义 JSON 评估任务
│   ├── gsm8k.py                              # GSM8K 数学推理任务
│   ├── mmlu.py                               # MMLU 选择题任务
│   ├── smoltalk.py                           # SmolTalk 对话任务
│   └── spellingbee.py                        # SpellingBee 拼写生成任务
├── tests/                                   # 轻量单元与兼容性测试
│   ├── test_config_and_boundaries.py        # 检查配置和数据边界
│   ├── test_dataloader_lightweight_state.py # 检查轻量级数据加载状态
│   ├── test_dataset_sft_mask.py             # 检查 SFT 损失掩码
│   ├── test_hf_configuration.py             # 检查 Transformers 配置
│   └── test_modeling_esm_rope.py            # 检查旋转位置嵌入
├── .gitignore                                # 忽略本地数据、输出和缓存
├── pyproject.toml                            # 包元数据和依赖分组
├── README.md                                 # 英文项目说明
├── README_zh.md                              # 简体中文项目说明
└── uv.lock                                   # 锁定的依赖版本
```

## 开发

```bash
uv sync --extra cpu --group dev
pytest -q
python -m scripts.train --help
python -m scripts.eval --help
python -m scripts.qa --help
python -m scripts.chat --help
```
