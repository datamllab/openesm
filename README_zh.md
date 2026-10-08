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

在仓库根目录安装环境，并选择一个 PyTorch extra：

```bash
# NVIDIA GPU
uv sync --extra gpu

# 仅使用 CPU
uv sync --extra cpu
```

设置训练 token 预算后启动完整训练：

```bash
TARGET_TOTAL_TOKENS=100000000 bash runs/train.sh
```

使用 checkpoint 执行评估或启动交互式生成：

```bash
bash runs/eval.sh --checkpoint /path/to/model.ckpt \
  --dataset dclm --data-dir /path/to/data

bash runs/chat.sh --checkpoint /path/to/model.ckpt \
  --tokenizer-path /path/to/tokenizer

# 浏览器界面和流式 API
uv sync --extra gpu --extra web
bash runs/chat.sh --web --checkpoint /path/to/model.ckpt --port 8000
```

发布到 Hugging Face 的模型可以直接使用 Transformers 加载。先安装额外
依赖 `uv sync --extra gpu --extra hf`，然后参考下方的 checkpoint 章节。

## 📈 Scaling Law

在 ClimbMix、DCLM 和 FineWeb 上，验证 BPB 随训练 token 数增加而下降。
DCLM 结果还显示，增加 token 预算和模型深度可降低验证误差；IsoFLOP 曲线
则表明，训练预算提高时，计算最优模型规模也随之增大。

<table>
  <tr>
    <td align="center" width="50%"><img src="figs/scaling_pretrain_climbmix_validation_bpb.png" width="100%" alt="ClimbMix 预训练验证 BPB"><br>ClimbMix</td>
    <td align="center" width="50%"><img src="figs/scaling_pretrain_dclm_validation_bpb.png" width="100%" alt="DCLM 预训练验证 BPB"><br>DCLM</td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="figs/scaling_pretrain_fineweb_validation_bpb.png" width="100%" alt="FineWeb 预训练验证 BPB"><br>FineWeb</td>
    <td align="center" width="50%"><img src="figs/dclm_best_val_bpb_vs_tokens.png" width="100%" alt="DCLM 验证 BPB 与训练 token 数的关系"><br>Token scaling</td>
  </tr>
  <tr>
    <td align="center" width="50%"><img src="figs/dclm_best_val_bpb_vs_depth.png" width="100%" alt="DCLM 验证 BPB 与模型深度的关系"><br>Depth scaling</td>
    <td align="center" width="50%"><img src="figs/scaling_isoflop_dclm.png" width="100%" alt="DCLM IsoFLOP scaling 曲线"><br>IsoFLOP scaling</td>
  </tr>
</table>

可使用 `configs/train.yaml` 作为默认配置，并通过 `TARGET_TOTAL_TOKENS`
设置训练预算。

## ⚙️ 预训练 Checkpoint

预训练模型发布在
[OpenESM Hugging Face collection](https://huggingface.co/collections/guan-wang/openesm)
中。Collection 包含基于 OWT、DCLM、FineWeb 和 ClimbMix 训练的 160M、520M
和 1B 参数规模模型，以及 FineWeb SFT 模型。

推荐使用标准 Transformers 格式。可以使用下面的命令将旧版 Lightning
checkpoint 转换为标准格式：

```bash
uv sync --extra cpu --extra hf
python -m scripts.export_hf \
  /path/to/model.ckpt \
  /path/to/hf-model \
  --tokenizer-dir /path/to/tokenizer
```

转换后的目录包含 `config.json`、`model.safetensors`、自定义 modeling 和
configuration 文件，以及 tokenizer 文件。之后可以这样加载：

```python
from transformers import AutoModelForMaskedLM, AutoTokenizer

model_id = "guan-wang/ESM-OWT-160M"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForMaskedLM.from_pretrained(model_id, trust_remote_code=True)
```

## 💬 Chat Demo

<p align="center">
  <img src="figs/chat.jpg" alt="ESM Chat 网页界面" width="850">
</p>

使用本地 checkpoint 启动交互式对话：

```bash
bash runs/chat.sh \
  --checkpoint /path/to/model.ckpt \
  --tokenizer-path /path/to/tokenizer
```

该 Demo 使用 checkpoint 中保存的 hyperparameters，并支持独立 checkpoint
加载章节中介绍的 tokenizer 文件。

在仓库根目录执行：

```bash
uv sync --extra gpu
```

仅使用 CPU 时执行 `uv sync --extra cpu`。运行测试时添加 `--group dev`：

```bash
uv sync --extra cpu --group dev
```

CPU 和 CUDA 的 PyTorch 软件源配置在 `pyproject.toml` 中。每个环境选择
一个 extra 即可。

## 运行

`runs/` 中的 shell 启动脚本保持轻量，并且始终从项目根目录运行，因此在
本地 shell 和分布式任务中都能稳定解析相对路径。

```bash
# 预训练：训练长度由 TARGET_TOTAL_TOKENS 决定。
TARGET_TOTAL_TOKENS=100000000 bash runs/train.sh

# 使用指定 checkpoint 进行监督微调。
bash runs/sft.sh \
  --execution_mode finetune \
  --dataset_name esm_sft \
  --finetuning_model_ckpt /path/to/pretrain.ckpt

# Position-wise BPB evaluation。
bash runs/eval.sh \
  --checkpoint /path/to/model.ckpt \
  --dataset dclm \
  --data-dir /path/to/data \
  --output_root . \
  --run_name eval-dclm

# QA evaluation。
bash runs/qa.sh \
  --checkpoint /path/to/model.ckpt \
  --eval-bundle /path/to/eval_bundle \
  --output_root . \
  --run_name qa-dclm

# 交互式生成。
bash runs/chat.sh --checkpoint /path/to/model.ckpt

# Zero-shot evaluation 使用 OWT 预训练 checkpoint 和准备好的数据。
CKPT=/path/to/owt-model.ckpt DATA_ROOT=/path/to/zeroshot-data \
  bash runs/zeroshot.sh
```

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

## 独立加载 Checkpoint

`esm/modeling_esm.py` 包含模型类和 Lightning checkpoint 加载器。若要直接
使用 ESM 加载 Lightning checkpoint，请将 checkpoint、tokenizer 文件和此文件
一同放入模型仓库：

```text
model-repository/
├── modeling_esm.py
├── model.ckpt
└── tokenizer/
    ├── tokenizer.pkl
    └── token_bytes.pt
```

安装 PyTorch 和 `tiktoken`，将 `modeling_esm.py` 放到 checkpoint 旁边，然后
执行：

```python
import torch
from modeling_esm import load_checkpoint

model, tokenizer, hparams, device = load_checkpoint(
    "model.ckpt",
    tokenizer_path="tokenizer",
)

input_ids = torch.tensor(
    [tokenizer.encode("hello", append=tokenizer.get_bos_token_id())],
    device=device,
)
logits = model(input_ids)
print(logits.shape)
```

如果 checkpoint 和 `tokenizer/` 是同级目录，可以省略 `tokenizer_path`。
`token_bytes.pt` 用于 BPB evaluation；只进行 logits 推理时只需要
`tokenizer.pkl`。加载器接受 Lightning checkpoint、本地 Transformers 目录或
Hugging Face 模型 ID，并返回统一的推理接口。标准 Transformers 模型仓库还
需要 `configuration_esm.py`、模型权重、`config.json` 和 tokenizer 文件；
`scripts/export_hf.py` 可以生成这种目录结构。

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
│   ├── dclm_best_val_bpb_vs_depth.png          # DCLM 深度扩展图
│   ├── dclm_best_val_bpb_vs_tokens.png         # DCLM token 扩展图
│   ├── scaling_isoflop_dclm.png                # DCLM IsoFLOP 扩展图
│   ├── scaling_pretrain_climbmix_validation_bpb.png # ClimbMix 验证 BPB 图
│   ├── scaling_pretrain_dclm_validation_bpb.png     # DCLM 验证 BPB 图
│   └── scaling_pretrain_fineweb_validation_bpb.png  # FineWeb 验证 BPB 图
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

集群专用的 rjob 提交脚本、私有数据、checkpoint、缓存、日志和生成结果应
放在公共源码树之外，或由 `.gitignore` 忽略。

## 开发

```bash
uv sync --extra cpu --group dev
pytest -q
python -m scripts.train --help
python -m scripts.eval --help
python -m scripts.qa --help
python -m scripts.chat --help
```

公共代码树保持精简：模型、数据接口、可运行入口和轻量测试是项目的主要
来源。
