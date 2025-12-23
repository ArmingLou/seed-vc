# 粤语发音修正功能使用指南

本功能用于修正语音转换后粤语发音/声调不准确的问题，通过后处理方式调整 F0（基频）曲线来改善声调表现。

## 粤语声调系统

粤语采用**六声九调**系统：

### 舒声（6种调型）

| 声调 | 名称 | 调值 | 例字 | F0曲线 |
|------|------|------|------|--------|
| 1 | 阴平 | 55 | 诗 si1 | 高平 ─ |
| 2 | 阴上 | 35 | 史 si2 | 高升 ╱ |
| 3 | 阴去 | 33 | 试 si3 | 中平 ─ |
| 4 | 阳平 | 21 | 时 si4 | 低降 ╲ |
| 5 | 阳上 | 23 | 市 si5 | 低升 ╱ |
| 6 | 阳去 | 22 | 事 si6 | 低平 ─ |

### 入声（3种，音节短促）

| 类型 | 粤拼标记 | 例字 | 说明 |
|------|----------|------|------|
| 阴入 | 1 | 识 sik1 | 调型同阴平，音节短 |
| 中入 | 3 | 锡 sek3 | 调型同阴去，音节短 |
| 阳入 | 6 | 食 sik6 | 调型同阳去，音节短 |

> 入声以 -p、-t、-k 结尾，在粤拼中复用 1/3/6 标记。

---

## 使用流程

### 1. 安装依赖

```bash
pip install pycantonese
```

### 2. 生成粤拼标注模板

**方式A：手动转换（需自己填时间）**

```bash
# 直接转换（行内格式）
python tools/yue_pinyin_converter.py "你好世界"
# 输出: nei5 hou2 sai3gaai3

# 生成 yue-fix 模板文件
python tools/yue_pinyin_converter.py "你好世界，今日天氣好好" -F yuefix -o my_fix.txt
```

**方式B：自动提取时间戳（推荐）**

使用 Whisper 自动识别并提取每个字的发音时间：

```bash
# 安装依赖
pip install openai-whisper

# 自动识别并提取时间
python tools/yue_time_extractor.py --audio source.wav -o fix_template.txt

# 指定文字（更准确）
python tools/yue_time_extractor.py --audio source.wav --text "你好世界" -o fix.txt

# 直接生成带时间戳的模板
python tools/yue_time_extractor.py --audio source.wav --text "你好" --with-times -o fix.txt
```

生成的模板示例：
```
# yue-fix 格式模板（自动提取时间戳）

nei5  # 你 (参考: 0.00 0.18)
hou2  # 好 (参考: 0.18 0.35)
sai3  # 世 (参考: 0.35 0.52)
gaai3  # 界 (参考: 0.52 0.70)
```

只需为需要修正的字添加时间戳（可直接复制括号内的时间）：
```
nei5  # 你 (参考: 0.00 0.18)
hou2  # 好 (参考: 0.18 0.35)
sai3 0.35 0.52  # 世  <- 复制时间到粤拼后
gaai3 0.52 0.70  # 界  <- 复制时间到粤拼后
```

### 3. 编辑修正文件

生成的模板文件格式如下：

```
# yue-fix 格式模板
# 格式: 粤拼 开始时间(秒) [结束时间(秒)]  # 原字
# 只有添加了时间戳的行才会被处理，未添加时间戳的行会被跳过

nei5  # 你
hou2  # 好
sai3  # 世
gaai3  # 界
# [，] 未识别
gam1  # 今
jat6  # 日
tin1  # 天
hei3  # 氣
hou2  # 好
hou2  # 好
```

**使用方法**：只需为需要修正的字添加时间戳，例如：

```
nei5  # 你
hou2  # 好
sai3 0.40 0.60  # 世  <- 添加了时间戳，会被修正
gaai3 0.60 0.80  # 界  <- 添加了时间戳，会被修正
gam1  # 今
jat6  # 日
```

**重要**：根据源音频的实际发音时间点，修改每行的开始时间（和可选的结束时间）。

时间戳获取方式：
- 使用 Audacity 等音频编辑软件查看波形
- 使用 Praat 等语音分析工具精确标注
- 根据经验估算（每字约 0.15-0.3 秒）

### 4. 执行修正

**方式A：在语音转换时应用修正（推荐）**

```bash
python inference.py \
    --source source.wav \
    --target reference.wav \
    --output ./output \
    --f0-condition True \
    --yue-fix my_fix.txt \
    --yue-fix-strength 0.8
```

**方式B：对已有音频进行修正（独立工具）**

```bash
# 基本用法
python tools/yue_audio_fix.py \
    --source input.wav \
    --yue-fix fix.txt \
    --output corrected.wav

# 指定修正强度
python tools/yue_audio_fix.py \
    --source input.wav \
    --yue-fix fix.txt \
    --yue-fix-strength 0.8 \
    --output corrected.wav

# 输出到目录
python tools/yue_audio_fix.py \
    --source input.wav \
    --yue-fix fix.txt \
    --output ./output/
```

---

## 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--yue-fix` | str | None | 粤语修正文件路径 |
| `--yue-fix-strength` | float | 0.8 | 修正强度 (0.0-1.0) |

### 修正强度说明

- `0.0`：不修正，保持原始 F0
- `0.5`：中等修正，原始和目标 F0 各占一半
- `0.8`：较强修正（推荐），目标 F0 占主导
- `1.0`：完全替换为目标 F0 曲线

---

## 修正文件格式

```
# 注释行（以 # 开头）
# 格式: 粤拼 开始时间 [结束时间]  # 原字符（可选注释）

nei5 0.00 0.15  # 你
hou2 0.15 0.30  # 好
sai3 0.30       # 世（省略结束时间，自动使用下一个开始时间）
```

### 格式说明

| 字段 | 必填 | 说明 |
|------|------|------|
| 粤拼 | ✓ | 粤拼（Jyutping），末尾数字表示声调 |
| 开始时间 | ✓ | 该音节在源音频中的开始时间（秒） |
| 结束时间 | ✗ | 结束时间（秒），省略时自动推断 |
| 注释 | ✗ | # 后的内容为注释，不影响解析 |

---

## 注意事项

1. **必须启用 F0 条件控制**
   - `--yue-fix` 需要配合 `--f0-condition True` 使用
   - 如果未启用，程序会自动启用并给出警告

2. **时间戳精度**
   - 时间戳越精确，修正效果越好
   - 建议精确到 0.01 秒

3. **入声处理**
   - 入声音节较短，时间窗口应相应缩短
   - 入声使用对应舒声的调型（1/3/6）

4. **多音字**
   - 同一个字可能有多个读音，需确认正确的粤拼

5. **连读变调**
   - 粤语连读时可能发生轻微变调，可适当降低修正强度

---

## 工具命令速查

```bash
# 粤拼转换工具
python tools/yue_pinyin_converter.py "文字内容"           # 行内格式
python tools/yue_pinyin_converter.py "文字" -F table      # 表格格式
python tools/yue_pinyin_converter.py "文字" -F yuefix     # yue-fix 格式
python tools/yue_pinyin_converter.py -i                   # 交互模式

# 时间戳自动提取工具
python tools/yue_time_extractor.py -i                                     # 交互模式（弹窗选择文件）
python tools/yue_time_extractor.py --audio audio.wav -o fix.txt           # 自动识别
python tools/yue_time_extractor.py --audio audio.wav --text "你好" -o fix.txt  # 指定文字
python tools/yue_time_extractor.py --audio audio.wav --with-times -o fix.txt    # 直接带时间

# 独立音频修正工具
python tools/yue_audio_fix.py -i                                         # 交互模式（弹窗选择文件）
python tools/yue_audio_fix.py --source audio.wav --yue-fix fix.txt --output out.wav

# 测试修正文件解析
python modules/yue_fix.py --fix-file my_fix.txt

# 完整推理命令
python inference.py \
    --source source.wav \
    --target reference.wav \
    --output ./output \
    --f0-condition True \
    --yue-fix my_fix.txt \
    --yue-fix-strength 0.8 \
    --diffusion-steps 30
```

---

## 常见问题

### Q: 修正后声调仍不准确？

A: 尝试以下方法：
- 提高 `--yue-fix-strength` 到 0.9 或 1.0
- 检查时间戳是否准确
- 确认粤拼声调标注正确

### Q: 修正后出现不自然的音调跳跃？

A: 可能原因：
- 时间戳不连续，存在间隙
- 修正强度过高，尝试降低到 0.6-0.7
- 边界处理问题，确保相邻音节时间衔接

### Q: 如何只修正部分字？

A: 两种方式：

**方式1**：修正文件中只列出需要修正的字
```
# 只修正“世”和“界”
sai3 0.40 0.60  # 世
gaai3 0.60 0.80  # 界
```

**方式2**：保留完整模板，只给需要修正的字添加时间戳
```
# 完整模板，只有“世”和“界”有时间戳
nei5  # 你 (无时间戳，不处理)
hou2  # 好 (无时间戳，不处理)
sai3 0.40 0.60  # 世 (有时间戳，会修正)
gaai3 0.60 0.80  # 界 (有时间戳，会修正)
gam1  # 今 (无时间戳，不处理)
jat6  # 日 (无时间戳，不处理)
```

程序会自动识别并跳过无时间戳的条目，并输出提示：
```
信息: 跳过了 4 个无时间戳的条目（不处理）
```
