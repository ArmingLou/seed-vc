# Seed-VC V2 训练指南

本文档详细介绍了如何使用 Seed-VC V2 版本的训练功能，包括新增的验证集支持、早停机制和教师模型支持。

## 目录
1. [基础训练](#基础训练)
2. [增量训练](#增量训练)
3. [验证集支持](#验证集支持)
4. [早停机制](#早停机制)
5. [FP16 精度支持](#fp16-精度支持)
6. [渐进式训练（教师模型）](#渐进式训练教师模型)
7. [常见问题](#常见问题)

## 基础训练

### 使用 train.sh 脚本（推荐）

```bash
# 基础训练
./train.sh --v2 --dataset-dir /path/to/dataset

# 指定配置文件
./train.sh --v2 --config ./configs/v2/vc_wrapper.yaml --dataset-dir /path/to/dataset

# 使用 GPU 训练
./train.sh --v2 --gpu --dataset-dir /path/to/dataset

# 设置训练参数
./train.sh --v2 --gpu --dataset-dir /path/to/dataset --max-steps 2000 --max-epochs 500 --save-every 100

# 单独训练 CFM 模型
./train.sh --v2 --gpu --dataset-dir /path/to/dataset --train-cfm

# 单独训练 AR 模型
./train.sh --v2 --gpu --dataset-dir /path/to/dataset --train-ar

# 同时训练 CFM 和 AR 模型
./train.sh --v2 --gpu --dataset-dir /path/to/dataset --train-cfm --train-ar
```

### 使用 train_v2.py 直接训练

```bash
# 基础训练
python train_v2.py --config ./configs/v2/vc_wrapper.yaml --dataset-dir /path/to/dataset

# 使用 GPU 训练
python train_v2.py --config ./configs/v2/vc_wrapper.yaml --dataset-dir /path/to/dataset --gpu

# 设置训练参数
python train_v2.py --config ./configs/v2/vc_wrapper.yaml --dataset-dir /path/to/dataset --max-steps 2000 --max-epochs 500 --save-every 100

# 单独训练 CFM 模型
python train_v2.py --config ./configs/v2/vc_wrapper.yaml --dataset-dir /path/to/dataset --train-cfm

# 单独训练 AR 模型
python train_v2.py --config ./configs/v2/vc_wrapper.yaml --dataset-dir /path/to/dataset --train-ar

# 同时训练 CFM 和 AR 模型
python train_v2.py --config ./configs/v2/vc_wrapper.yaml --dataset-dir /path/to/dataset --train-cfm --train-ar
```

## 增量训练

增量训练允许您在已有模型的基础上，使用新数据进行微调，而无需重新训练整个模型。

### 数据准备

1. 准备新数据集，确保格式与原始数据集一致
2. 建议保持数据文件命名的一致性以便于管理

### 使用增量训练

```bash
# 使用预训练的 CFM 模型进行增量训练
python train_v2.py --config ./configs/v2/vc_wrapper.yaml \
  --pretrained-cfm-ckpt /path/to/pretrained/cfm_model.pth \
  --dataset-dir /path/to/new_dataset

# 使用预训练的 AR 模型进行增量训练
python train_v2.py --config ./configs/v2/vc_wrapper.yaml \
  --pretrained-ar-ckpt /path/to/pretrained/ar_model.pth \
  --dataset-dir /path/to/new_dataset

# 同时使用预训练的 CFM 和 AR 模型进行增量训练
python train_v2.py --config ./configs/v2/vc_wrapper.yaml \
  --pretrained-cfm-ckpt /path/to/pretrained/cfm_model.pth \
  --pretrained-ar-ckpt /path/to/pretrained/ar_model.pth \
  --dataset-dir /path/to/new_dataset
```

## 验证集支持

验证集支持允许您在训练过程中监控模型性能，并使用早停机制防止过拟合。

### 数据准备

1. 准备独立的验证数据集，确保与训练集数据分布相似但无重复
2. 将验证集放在单独的目录中

### 使用验证集训练

```bash
# 使用 train.sh 脚本
./train.sh --v2 --gpu --dataset-dir /path/to/dataset --val-dataset-dir /path/to/val_dataset

# 使用 train_v2.py 直接训练
python train_v2.py --config ./configs/v2/vc_wrapper.yaml \
  --dataset-dir /path/to/dataset --val-dataset-dir /path/to/val_dataset
```

## 早停机制

早停机制通过监控验证集损失来自动停止训练，防止过拟合。

### 参数说明

- `--patience`: 早停耐心值，默认为 20
- `--validation-interval`: 验证间隔（steps），默认为 50

### 使用早停机制

```bash
# 使用默认早停参数
./train.sh --v2 --gpu --dataset-dir /path/to/dataset --val-dataset-dir /path/to/val_dataset

# 自定义早停参数
./train.sh --v2 --gpu --dataset-dir /path/to/dataset --val-dataset-dir /path/to/val_dataset \
  --patience 15 --validation-interval 25

# 使用 train_v2.py 直接训练
python train_v2.py --config ./configs/v2/vc_wrapper.yaml \
  --dataset-dir /path/to/dataset --val-dataset-dir /path/to/val_dataset \
  --patience 15 --validation-interval 25
```

## FP16 精度支持

FP16 精度可以显著减少内存使用并加速训练过程。

### 使用 FP16 精度

```bash
# 使用 train.sh 脚本
./train.sh --v2 --gpu --dataset-dir /path/to/dataset --fp16

# 使用 train_v2.py 直接训练
python train_v2.py --config ./configs/v2/vc_wrapper.yaml \
  --dataset-dir /path/to/dataset --fp16
```

## 渐进式训练（教师模型）

渐进式训练是一种避免灾难性遗忘并逐步扩展模型泛化能力的训练方法。它通过使用教师模型进行知识蒸馏，保留之前学到的知识，从而在新数据集上进行训练。

### 数据准备

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

### 使用渐进式训练

```bash
# 使用教师模型进行知识蒸馏（训练单个数据集）
python train_v2_with_teacher.py --config ./configs/v2/vc_wrapper.yaml \
  --teacher-model /path/to/previous_model/cfm_model.pth \
  --dataset-dir /path/to/dataset

# 结合传统训练参数
python train_v2_with_teacher.py --config ./configs/v2/vc_wrapper.yaml \
  --teacher-model /path/to/previous_model/cfm_model.pth \
  --dataset-dir /path/to/dataset \
  --max-steps 2000 --max-epochs 500 --save-every 100

# 同时训练 CFM 和 AR 模型
python train_v2_with_teacher.py --config ./configs/v2/vc_wrapper.yaml \
  --teacher-model /path/to/previous_model/cfm_model.pth \
  --dataset-dir /path/to/dataset \
  --train-cfm --train-ar
```

### 渐进式训练的优势

1. **无目录结构限制**：不需要遵循特定的version_*命名规范
2. **灵活的数据集组合**：可以自由选择任意数据集进行训练
3. **手动教师模型指定**：可以手动指定任何模型作为教师模型
4. **独立的输出目录**：每个数据集的输出目录以其名称命名，便于管理
5. **更好的可控性**：可以精确控制训练顺序和教师模型

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
10. 指定教师模型路径

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
./train.sh --v2 --dataset-dir /path/to/dataset --fp16 False
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
./train.sh --v2 --gpu --dataset-dir /path/to/dataset --val-dataset-dir /path/to/val_dataset

# 使用 train_v2.py 直接训练
python train_v2.py --config ./configs/v2/vc_wrapper.yaml \
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
./train.sh --v2 --gpu --dataset-dir /path/to/dataset --val-dataset-dir /path/to/val_dataset \
  --patience 10 --validation-interval 25

# 平衡的早停设置
./train.sh --v2 --gpu --dataset-dir /path/to/dataset --val-dataset-dir /path/to/val_dataset \
  --patience 20 --validation-interval 50

# 宽松的早停设置（更长时间训练）
./train.sh --v2 --gpu --dataset-dir /path/to/dataset --val-dataset-dir /path/to/val_dataset \
  --patience 30 --validation-interval 100
```

### 5. 验证结果解读

#### 指标监控
- 密切关注训练损失和验证损失的变化趋势
- 当验证损失连续若干次迭代没有改善时，早停机制会自动触发
- 如果训练损失持续下降但验证损失开始上升，表明可能出现过拟合

#### 最佳模型选择
- 系统会自动保存验证损失最低的模型作为最佳模型
- 最终模型文件将包含最佳验证性能的模型权重

通过合理设置验证集和早停参数，您可以有效地监控训练过程，防止过拟合，并自动确定最佳训练终止时机。




├── 训练集（train）：同说话人 A、B、C（80%）
│   └── 用于：参数更新
│
├── 验证集（val）：同说话人 A、B、C 的不同句子（10%）
│   └── 用于：早停 + 超参数调优
│
└── 测试集（test）：完全不同说话人 X、Y、Z（10%）
    └── 用于：训练完成后评估泛化能力