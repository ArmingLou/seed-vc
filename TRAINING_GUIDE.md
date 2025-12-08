# Seed-VC 训练指南

本文档详细介绍了如何使用 Seed-VC 的训练功能，特别是新增的验证集支持和早停机制。

## 目录
1. [基础训练](#基础训练)
2. [增量训练](#增量训练)
3. [验证集支持](#验证集支持)
4. [早停机制](#早停机制)
5. [FP16 精度支持](#fp16-精度支持)
6. [渐进式训练](#渐进式训练)
7. [常见问题](#常见问题)

## 基础训练

### 使用 train.sh 脚本（推荐）

```bash
# 基础训练
./train.sh --v1 --dataset-dir /path/to/dataset

# 指定配置文件
./train.sh --v1 --config ./configs/presets/config_cantonese-dit_mel_seed_uvit_whisper_small_wavenet.yml --dataset-dir /path/to/dataset

# 使用 GPU 训练
./train.sh --v1 --gpu --dataset-dir /path/to/dataset

# 设置训练参数
./train.sh --v1 --gpu --dataset-dir /path/to/dataset --max-steps 2000 --max-epochs 500 --save-every 100
```

### 使用 train.py 直接训练

```bash
# 基础训练
python train.py --config ./configs/presets/config_cantonese-dit_mel_seed_uvit_whisper_small_wavenet.yml --dataset-dir /path/to/dataset

# 使用 GPU 训练
python train.py --config ./configs/presets/config_cantonese-dit_mel_seed_uvit_whisper_small_wavenet.yml --dataset-dir /path/to/dataset --gpu 0

# 设置训练参数
python train.py --config ./configs/presets/config_cantonese-dit_mel_seed_uvit_whisper_small_wavenet.yml --dataset-dir /path/to/dataset --max-steps 2000 --max-epochs 500 --save-every 100
```

## 增量训练

增量训练允许您在已有模型的基础上，使用新数据进行微调，而无需重新训练整个模型。

### 数据准备

1. 准备新数据集，确保格式与原始数据集一致
2. 建议保持数据文件命名的一致性以便于管理

### 训练步骤

```bash
# 增量训练 - 继续使用之前的模型和日志目录
./train.sh --v1 --gpu --dataset-dir /path/to/new_dataset --run-name previous_run_name

# 或者指定具体的检查点
python train.py --config ./configs/presets/config_cantonese-dit_mel_seed_uvit_whisper_small_wavenet.yml --dataset-dir /path/to/new_dataset --run-name previous_run_name --pretrained-ckpt /path/to/checkpoint
```

## 验证集支持

新增的验证集功能可以帮助您监控模型在未见数据上的表现，并实现早停机制。

### 准备验证集

1. 准备一个独立的验证数据集，格式与训练集相同
2. 确保验证集数据与训练集数据不重叠

### 使用验证集

```bash
# 使用 train.sh 脚本
./train.sh --v1 --gpu --dataset-dir /path/to/dataset --val-dataset-dir /path/to/val_dataset

# 使用 train.py 直接训练
python train.py --config ./configs/presets/config_cantonese-dit_mel_seed_uvit_whisper_small_wavenet.yml --dataset-dir /path/to/dataset --val-dataset-dir /path/to/val_dataset
```

### 命令行参数

- `--val-dataset-dir` 或 `--val-dir`: 验证集目录路径
- `--patience` 或 `-p`: 早停耐心值（默认: 20）
- `--validation-interval` 或 `-v`: 验证间隔（steps，默认: 50）

### 示例

```bash
# 完整的带验证集训练命令
./train.sh --v1 --gpu --dataset-dir /path/to/dataset --val-dataset-dir /path/to/val_dataset --patience 30 --validation-interval 100 --max-steps 2000

# 使用短参数
./train.sh --v1 --gpu -d /path/to/dataset --val-dir /path/to/val_dataset -p 30 -v 100 -s 2000
```

## 早停机制

早停机制会在验证集损失连续若干次迭代没有改善时自动停止训练，防止过拟合。

### 工作原理

1. 每隔 `validation-interval` 步在验证集上评估模型
2. 记录最佳验证损失
3. 如果验证损失连续 `patience` 次没有改善，则停止训练

### 使用示例

```bash
# 设置早停参数
./train.sh --v1 --gpu --dataset-dir /path/to/dataset --val-dataset-dir /path/to/val_dataset --patience 20 --validation-interval 50

# 更敏感的早停设置（更快停止）
./train.sh --v1 --gpu --dataset-dir /path/to/dataset --val-dataset-dir /path/to/val_dataset --patience 10 --validation-interval 25

# 更宽松的早停设置（更长时间训练）
./train.sh --v1 --gpu --dataset-dir /path/to/dataset --val-dataset-dir /path/to/val_dataset --patience 50 --validation-interval 100
```

## FP16 精度支持

FP16 精度可以减少内存使用并加速训练，但可能在某些设备上存在兼容性问题。

### 启用 FP16

```bash
# 使用 train.sh 脚本
./train.sh --v1 --gpu --dataset-dir /path/to/dataset --fp16

# 使用 train.py 直接训练
python train.py --config ./configs/presets/config_cantonese-dit_mel_seed_uvit_whisper_small_wavenet.yml --dataset-dir /path/to/dataset --fp16
```

### 注意事项

1. 在 CPU 设备上使用 FP16 可能不会带来性能提升，且可能导致兼容性问题
2. 如果遇到问题，建议关闭 FP16 精度 (`--fp16 False`)

## 交互式模式

train.sh 脚本支持交互式参数选择，方便快速配置训练参数。

```bash
# 启动交互式模式
./train.sh --interactive

# 或者
./train.sh -I
```

在交互式模式下，您可以：
1. 选择版本（V1 或 V2）
2. 选择是否使用 GPU
3. 设置运行名称
4. 选择配置文件
5. 选择数据集目录
6. 设置训练参数
7. 指定验证集目录
8. 设置早停参数
9. 选择是否使用 FP16 精度

## 渐进式训练

渐进式训练是一种避免灾难性遗忘并逐步扩展模型泛化能力的训练方法。它通过使用教师模型进行知识蒸馏，保留之前学到的知识，从而在新数据集上进行训练。

### 渐进式训练

#### 数据准备

灵活渐进式训练不要求特定的目录结构，您可以直接指定任意数据集目录：

```
dataset1/                      <- 任意命名的数据集目录
├── audio1.wav
├── audio2.wav
└── ...

dataset2/                      <- 另一个数据集目录
├── audio3.wav
├── audio4.wav
└── ...

special_dataset/               <- 特殊命名的数据集目录
├── audio5.wav
├── audio6.wav
└── ...
```

#### 使用渐进式训练

```bash
# 使用教师模型进行知识蒸馏（训练单个数据集）
python train.py --config ./configs/presets/config_cantonese-dit_mel_seed_uvit_whisper_small_wavenet.yml \
  --teacher-model /path/to/previous_model/ft_model.pth \
  --dataset-dir /path/to/dataset1

# 结合传统训练参数
python train.py --config ./configs/presets/config_cantonese-dit_mel_seed_uvit_whisper_small_wavenet.yml \
  --teacher-model /path/to/previous_model/ft_model.pth \
  --dataset-dir /path/to/dataset1 \
  --max-steps 2000 --max-epochs 500 --save-every 100
```

#### 渐进式训练的优势

1. **无目录结构限制**：不需要遵循特定的version_*命名规范
2. **灵活的数据集组合**：可以自由选择任意数据集进行训练
3. **手动教师模型指定**：可以手动指定任何模型作为教师模型
4. **独立的输出目录**：每个数据集的输出目录以其名称命名，便于管理
5. **更好的可控性**：可以精确控制训练顺序和教师模型

### 渐进式训练输出目录结构

训练完成后，输出文件会按以下结构组织：

```
./runs/                        <- 训练输出根目录
├── myrun/                     <- 主模型归档目录
│   ├── ft_model.pth           <- 最终模型（从特定数据集目录拷贝）
│   └── config_cantonese-dit_mel_seed_uvit_whisper_small_wavenet.yml  <- 配置文件
├── myrun_dataset1/            <- 数据集特定目录（用于训练和续点）
│   ├── DiT_epoch_00000_step_00100.pth
│   ├── DiT_epoch_00000_step_00200.pth
│   ├── ft_model.pth           <- 数据集训练的最终模型
│   └── config_cantonese-dit_mel_seed_uvit_whisper_small_wavenet.yml
```

### 命令行参数

- `--teacher-model`: 知识蒸馏的教师模型路径


### 渐进式训练工作原理

1. 当指定了 `--teacher-model` 参数时，启用渐进式训练模式
2. 从指定的教师模型加载权重作为初始化
3. 使用知识蒸馏技术保留之前学到的知识
4. 训练输出目录格式为 `./runs/{run_name}_{dataset_name}/`，其中 `dataset_name` 是数据集目录的basename
5. 训练完成后，将最终模型和配置文件拷贝到 `./runs/{run_name}/` 目录作为主模型归档
6. 续点训练时，系统从数据集特定目录 `./runs/{run_name}_{dataset_name}/` 恢复完整训练状态

### 渐进式训练关键特性

- **数据集内继续训练**：同一数据集训练中断后，系统会自动加载该数据集的最新检查点，从上次中断的step继续训练
- **知识蒸馏**：通过指定的教师模型进行知识蒸馏，保留之前学到的知识
- **输出目录隔离**：不同数据集使用不同的输出目录 `./runs/{run_name}_{dataset_name}/`，确保训练过程隔离
- **主模型归档**：训练完成后自动将最终模型拷贝到 `./runs/{run_name}/` 目录
- **手动控制**：完全手动控制训练过程，不进行任何自动化行为

### 渐进式训练优势

1. **完全手动控制**：没有任何自动化行为，完全由用户控制
2. **输出目录管理**：自动为不同数据集创建独立的输出目录，便于管理
3. **主模型归档**：自动将最终模型拷贝到主目录，便于统一管理
4. **知识保留**：通过知识蒸馏保留教师模型的知识
5. **易于调试**：可以精确控制训练过程的每一步

## 常见问题

### 1. 如何确定验证集大小？

建议验证集大小占总数据的 10-20%。太小可能无法有效评估模型性能，太大则会浪费训练数据。

### 2. 早停参数如何设置？

- `patience`: 通常设置为 10-30，具体取决于数据集大小和训练稳定性
- `validation-interval`: 通常设置为 25-100，较小的值能更及时地检测到过拟合，但会增加训练时间

### 3. 训练中断后如何继续？

训练脚本会自动检测检查点并从中断处继续训练。确保使用相同的 `--run-name` 参数和相同的数据集目录。系统会从数据集特定目录 `./runs/{run_name}_{dataset_name}/` 恢复完整训练状态，包括优化器、调度器、epoch计数器和step计数器。

### 4. 如何查看训练日志？

训练日志保存在数据集特定目录 `./runs/{run_name}_{dataset_name}/` 下，包括：
- 配置文件副本
- 模型检查点
- 训练日志

最终模型和配置文件也会拷贝到主目录 `./runs/{run_name}/` 下。

### 5. FP16 精度导致的问题

如果在 CPU 上使用 FP16 精度时遇到问题：
```bash
# 关闭 FP16 精度
./train.sh --v1 --dataset-dir /path/to/dataset --fp16 False
```

## 验证数据最佳实践

### 1. 验证集大小建议

验证集的大小对模型评估和早停机制的有效性至关重要：

- **比例建议**：验证集应占总数据的 **10-20%**
- **最小数量**：建议至少包含 **500个样本**，以确保评估统计的可靠性
- **特殊情况**：对于低资源语言或小规模数据集，可适当提高比例至 **20-30%** 以确保评估稳定性
- **平衡考虑**：
  - 太小的验证集可能无法有效评估模型性能，导致评估结果不稳定
  - 太大的验证集会浪费宝贵的训练数据，可能影响模型的最终性能

### 2. 验证集构建原则

为了确保验证集能够有效评估模型性能，请遵循以下原则：

#### 数据代表性
- 验证集应与训练集具有相似的数据分布特征
- 包含与训练集相同类型的样本和场景
- 确保验证集中各类别样本的比例与训练集保持一致

#### 数据独立性
- 验证集必须与训练集完全独立，无重复样本
- 避免同一说话人的音频同时出现在训练集和验证集中
- 确保验证集中的音频内容与训练集不重叠

#### 数据质量
- 验证集应包含高质量的样本，避免过多噪声或损坏的数据
- 样本应涵盖各种实际应用场景，以全面评估模型性能

### 3. 验证集使用方法

#### 准备验证集
1. 从完整数据集中划分出独立的验证集
2. 确保验证集格式与训练集一致
3. 验证集应放置在单独的目录中

#### 使用验证集训练
```bash
# 使用 train.sh 脚本
./train.sh --v1 --gpu --dataset-dir /path/to/dataset --val-dataset-dir /path/to/val_dataset

# 使用 train.py 直接训练
python train.py --config ./configs/presets/config_cantonese-dit_mel_seed_uvit_whisper_small_wavenet.yml \
  --dataset-dir /path/to/dataset --val-dataset-dir /path/to/val_dataset
```

### 4. 验证频率和早停参数设置

#### 验证频率建议
- **validation-interval**：通常设置为 **25-100 steps**
  - 较小的值能更及时地检测到过拟合，但会增加训练时间
  - 较大的值会减少验证开销，但可能错过最佳停止时机

#### 早停参数建议
- **patience**：通常设置为 **10-30**
  - 数据集较小时建议使用较小的patience值（10-15）
  - 数据集较大或训练较稳定时可以使用较大的patience值（20-30）

#### 示例配置
```bash
# 敏感的早停设置（较快停止）
./train.sh --v1 --gpu --dataset-dir /path/to/dataset --val-dataset-dir /path/to/val_dataset \
  --patience 10 --validation-interval 25

# 平衡的早停设置
./train.sh --v1 --gpu --dataset-dir /path/to/dataset --val-dataset-dir /path/to/val_dataset \
  --patience 20 --validation-interval 50

# 宽松的早停设置（更长时间训练）
./train.sh --v1 --gpu --dataset-dir /path/to/dataset --val-dataset-dir /path/to/val_dataset \
  --patience 30 --validation-interval 100
```

### 5. 验证结果解读

#### 指标监控
- 密切关注训练损失和验证损失的变化趋势
- 当验证损失连续若干次迭代没有改善时，早停机制会自动触发
- 如果训练损失持续下降但验证损失开始上升，表明可能出现过拟合

#### 最佳模型选择
- 系统会自动保存验证损失最低的模型作为最佳模型
- 最终模型文件 `ft_model.pth` 将包含最佳验证性能的模型权重

通过合理设置验证集和早停参数，您可以有效地监控训练过程，防止过拟合，并自动确定最佳训练终止时机。</parameter_content>


## 如果训练集没有目标人 X，验证集用 X 的话，val loss 不会下降。必须让目标人 X 在训练集里（占 60-100%），验证集也用 X 的不同句子，val loss 才能正常下降。  

├── 训练集（train）：同说话人 A、B、C（80%）
│   └── 用于：参数更新
│
├── 验证集（val）：同说话人 A、B、C 的不同句子（10%）
│   └── 用于：早停 + 超参数调优
│
└── 测试集（test）：完全不同说话人 X、Y、Z（10%）
    └── 用于：训练完成后评估泛化能力
    
    
    
正确的音色模仿训练方案
### 方案1：目标人必须在训练集（推荐） 
  plaintext
  训练集（80%）：目标人 X 的 1520 条音频
  验证集（20%）：目标人 X 的 380 条音频（不同句子）

  目标：让模型学会 X 的音色和发音特征
  命令：
  bash
  ./train.sh \
    --dataset-dir /mnt/data/target_speaker_X/train/ \
    --val-dataset-dir /mnt/data/target_speaker_X/val/ \
    --initial-lr 5e-6 \
    --patience 10
  效果：
  train loss 降：模型学习 X 的特征
  val loss 降：模型在 X 的其他句子上泛化良好
  早停机制正常工作   

### 方案2：多人训练 + 目标人在其中
  如果你想同时学习多人 + 重点优化目标人：
  plaintext
  训练集：
    - 目标人 X：1200 条（60%）
    - 其他人 A、B、C：800 条（40%）
    
  验证集：
    - 目标人 X：300 条
    - 其他人 A、B、C：200 条
  目的：
  主要学习目标人 X
  辅助学习其他人（提升泛化能力）
  val loss 仍然能下降（因为 X 在训练集里）
  
### 你当前方案的致命问题
  plaintext
  训练集：A、B、C（非目标人）
  验证集：X（目标人）

  问题链：
  1. 模型学的是 A、B、C
  2. 验证时用 X 评估
  3. val loss 不降（正常现象）
  4. 早停触发（模型认为"学不动了"）
  5. 最终模型：只会 A、B、C，不会 X
  
### 结论：训练完成后，模型对目标人 X 的转换效果很差。

### 如果目标人数据较少，非目标人数据较多，想模仿目标人的音色又想训练出目标人说不同内容的泛化能力。
### 唯一可行的改法：
  #### 步骤1：用 A、B、C 训练通用模型
    bash
    ./train.sh \
      --dataset-dir /mnt/data/speakers_ABC/ \
      --val-dataset-dir /mnt/data/speakers_ABC_val/ \
      --max-steps 2500 \
      --patience 10
  #### 步骤2：用目标人 X 微调（Fine-tune）
    bash
    ./train.sh \
      --dataset-dir /mnt/data/target_speaker_X/train/ \
      --val-dataset-dir /mnt/data/target_speaker_X/val/ \
      --pretrained-ckpt ./runs/abc_base/best_model.pth \  # 加载步骤1的模型
      --initial-lr 1e-6 \  # 微调用更小学习率
      --max-steps 500 \
      --patience 5
  #### 效果：
  步骤1：学会通用语音转换能力（A、B、C）
  步骤2：在 X 上专门优化
  最终模型：既有通用能力，又擅长 X


## 两阶段的验证集配置  

| 阶段             | 训练集                   | 验证集                          | 原因               |
|------------------|------------------------|--------------------------------|--------------------|
| **阶段1：预训练**  | 多人A、B、C...（16000条） | 多人A、B、C...（4000条，不同句子） | 监控通用模型学习效果 |
| **阶段2：微调**    | 目标人X（30条增强到150条） | 目标人X（10条原始）              | 监控目标人特征学习   |  


阶段1用大数据正常训练参数，阶段2必须大幅降低学习率、减少步数、增加容忍度，并强制使用数据增强，防止40条数据过拟合。

### 阶段1（20000条多人）vs 阶段2（40条目标人）的训练参数必须大幅调整，否则会严重过拟合或训练不足。

#### 关键参数对比表

| 参数                 | 阶段1（20000条） | 阶段2（40条） | 原因 |
|---------------------|-----------------|-------------|------|
| initial-lr          | 5e-6            | 1e-6        | 微调用极小学习率，防止破坏通用知识 |
| warmup_steps        | 100             | 5           | 数据少，预热步数大幅减少 |
| max_steps           | 5000            | 200         | 40条数据只需少量迭代 |
| patience            | 15              | 20          | 数据少波动大，需更高容忍度 |
| validation_interval | 50              | 10          | 更频繁监控防止过拟合 |
| 数据增强             | 无               | 必须（5倍）  | 40条数据极易过拟合 |

#### 详细说明

##### 1. 学习率（initial-lr）

```bash
# 阶段1：5e-6（正常训练）
--initial-lr 5e-6

# 阶段2：1e-6（微调，降低5倍）
--initial-lr 1e-6
```

**原因**：防止微调时"覆盖"预训练学到的通用特征

##### 2. 预热步数（warmup_steps）

```bash
# 阶段1：(20000×0.04)/8 ≈ 100步
--warmup-steps 100

# 阶段2：(40×0.04)/8 ≈ 0.2步 → 实际设为5步
--warmup-steps 5
```

**原因**：40条数据预热太快结束，设5步保证平稳启动

##### 3. 最大步数（max_steps）

```bash
# 阶段1：约2.5轮epoch（20000/8=2500步/轮）
--max-steps 5000

# 阶段2：约6-7轮epoch（40/8=5步/轮）
--max-steps 200  # 40轮epoch，防止过拟合
```

**原因**：40条数据迭代太多会记住所有样本

##### 4. 早停容忍度（patience）

```bash
# 阶段1：patience 15（正常容忍波动）
--patience 15

# 阶段2：patience 20（极高容忍度）
--patience 20
```

**原因**：10条验证集波动极大，需避免误触发

##### 5. 验证频率（validation_interval）

```bash
# 阶段1：每50步验证一次
--validation-interval 50

# 阶段2：每10步验证一次（更密集监控）
--validation-interval 10
```

**原因**：快速捕捉过拟合迹象

##### 6. 数据增强（必须）

```bash
# 阶段1：不需要（数据充足）
# 阶段2：必须增强5倍（40 → 200条）
```

**方法**：
- 音频变换：变速、变调、加噪
- SpecAugment：频谱mask

#### 完整命令对比

##### 阶段1（预训练）

```bash
./train.sh \
  --dataset-dir /data/multi/train/ \
  --val-dataset-dir /data/multi/val/ \
  --initial-lr 5e-6 \
  --warmup-steps 100 \
  --max-steps 5000 \
  --patience 15 \
  --validation-interval 50
```

##### 阶段2（微调）

```bash
./train.sh \
  --dataset-dir /data/target/train_aug/ \  # 增强后200条
  --val-dataset-dir /data/target/val/ \    # 原始10条
  --pretrained-ckpt ./runs/base/model.pth \
  --initial-lr 1e-6 \
  --warmup-steps 5 \
  --max-steps 200 \
  --patience 20 \
  --validation-interval 10
```



# 什么时候应该调低 或 调高 学习率
不是验证loss上升就需要调低学习率，而是 验证loss不好（上升或停滞），而 训练loss却一直在上升，才需要调低学习率。学习率直接影响的是训练 loss。
如果验证loss在上升或停滞，但是 训练loss基本停滞（很小幅度波动），那就是学习率不足，不能让 验证率突破下降，学习力不足，这时就应该反而调高学习率。
学习率直接 影响 训练loss，直接表现就是：学习率过高，训练loss会上升（不好，在发散）；学习率过低，训练loss会停滞（极小幅度波动），基本没有什么学习效果，但是此时也会 导致 验证loss 上升的，验证loss，是逆水行舟，学习不前，验证loss就上升。