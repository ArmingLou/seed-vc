#!/usr/bin/env python3
"""
粤语发音时间提取工具
使用Whisper自动提取音频中每个字的发音时间，生成yue-fix模板

用法:
    python tools/yue_time_extractor.py --audio input.wav --output fix_template.txt
    python tools/yue_time_extractor.py --audio input.wav --text "你好世界" --output fix.txt
"""

import os
import sys
import argparse

# 解决 macOS 上 OpenMP 库冲突
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_dependencies():
    """检查必要的依赖，返回可用的 whisper 类型"""
    missing = []
    whisper_type = None
    
    # 优先使用 openai-whisper（faster-whisper 对粤语识别效果差）
    try:
        import whisper
        whisper_type = "openai_whisper"
    except ImportError:
        # 回退到 faster-whisper
        try:
            from faster_whisper import WhisperModel
            print("警告: faster-whisper 对粤语识别效果较差，建议使用 --text 手动指定文字")
            whisper_type = "faster_whisper"
        except ImportError:
            missing.append("openai-whisper")
    
    # 检查 pycantonese
    try:
        import pycantonese
    except ImportError:
        missing.append("pycantonese")
    
    if missing:
        print("错误: 缺少必要依赖")
        print("运行以下命令安装:")
        print(f"  pip install {' '.join(missing)}")
        sys.exit(1)
    
    return whisper_type


def load_whisper_model(model_size: str = "small", whisper_type: str = "faster_whisper"):
    """加载Whisper模型"""
    
    if whisper_type == "faster_whisper":
        from faster_whisper import WhisperModel
        import torch
        
        # 确定设备和计算类型
        if torch.cuda.is_available():
            device = "cuda"
            compute_type = "float16"
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "cpu"  # faster-whisper 不支持 MPS
            compute_type = "int8"
        else:
            device = "cpu"
            compute_type = "int8"
        
        print(f"加载 faster-whisper 模型 ({model_size}) on {device}...")
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
        return model, "faster_whisper"
    else:
        import whisper
        print(f"加载 openai-whisper 模型 ({model_size})...")
        model = whisper.load_model(model_size)
        return model, "openai_whisper"


def extract_word_timestamps(audio_path: str, model, model_type: str = "faster_whisper", language: str = "yue") -> list:
    """
    使用Whisper提取word-level时间戳
    
    Returns:
        list of dict: [{"word": "你", "start": 0.0, "end": 0.2}, ...]
    """
    print(f"提取时间戳 (language={language})...")
    
    if model_type == "faster_whisper":
        # faster-whisper 支持 yue
        segments, info = model.transcribe(
            audio_path,
            language=language,
            word_timestamps=True,
            beam_size=5
        )
        
        words = []
        full_text = ""
        segments_list = list(segments)  # 消耗生成器
        
        for segment in segments_list:
            full_text += segment.text
            if segment.words:
                for word in segment.words:
                    word_text = word.word.strip()
                    # 过滤空白
                    if word_text:
                        words.append({
                            "word": word_text,
                            "start": word.start,
                            "end": word.end
                        })
        
        print(f"检测到语言: {info.language} (置信度: {info.language_probability:.2f})")
        print(f"原始识别结果: {full_text}")
        
        # 检查是否包含中文
        has_chinese = any('\u4e00' <= c <= '\u9fff' for c in full_text)
        if not has_chinese:
            print("警告: 识别结果不包含中文，可能识别失败")
            print("建议: 使用 --text 参数手动指定文字内容")
        
        return words, full_text
    else:
        # openai-whisper 不支持 yue，降级为 zh
        actual_lang = "zh" if language == "yue" else language
        if language == "yue":
            print("提示: openai-whisper 不支持 'yue'，使用 'zh' 识别")
        
        result = model.transcribe(
            audio_path,
            language=actual_lang,
            word_timestamps=True,
            verbose=False
        )
        
        words = []
        for segment in result.get("segments", []):
            for word_info in segment.get("words", []):
                words.append({
                    "word": word_info["word"].strip(),
                    "start": word_info["start"],
                    "end": word_info["end"]
                })
        
        return words, result.get("text", "")


def extract_using_vad(audio_path: str, sr: int = 44100) -> list:
    """
    使用VAD（语音活动检测）提取音节边界
    适用于无法使用Whisper的情况
    """
    import librosa
    import numpy as np
    
    print("使用能量检测提取音节边界...")
    
    # 加载音频
    audio, _ = librosa.load(audio_path, sr=sr)
    
    # 计算短时能量
    hop_length = int(sr * 0.01)  # 10ms
    frame_length = int(sr * 0.025)  # 25ms
    
    # 使用librosa的onset检测
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr, hop_length=hop_length)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, 
        sr=sr, 
        hop_length=hop_length,
        backtrack=True
    )
    
    # 转换为时间
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=hop_length)
    
    # 生成音节区间
    segments = []
    for i, start in enumerate(onset_times):
        if i + 1 < len(onset_times):
            end = onset_times[i + 1]
        else:
            end = len(audio) / sr
        
        segments.append({
            "word": f"[音节{i+1}]",
            "start": float(start),
            "end": float(end)
        })
    
    return segments


def text_to_jyutping_with_alignment(text: str, timestamps: list) -> list:
    """
    将文字转换为粤拼并与时间戳对齐
    
    逻辑: 按 Whisper 识别的时间戳顺序，依次分配给用户指定的每个字（包括标点）
    多出来的字放最后，不分配时间
    """
    import pycantonese
    
    # 尝试加载简繁转换库
    try:
        from opencc import OpenCC
        s2t = OpenCC('s2t')  # 简体转繁体
        has_opencc = True
    except ImportError:
        has_opencc = False
        print("提示: 安装 opencc-python-reimplemented 可改善简体字粤拼识别")
        print("  pip install opencc-python-reimplemented")
    
    def get_jyutping(char):
        """获取单字粤拼，支持简繁体"""
        # 先尝试直接查询
        jp_list = pycantonese.characters_to_jyutping(char)
        if jp_list and jp_list[0][1]:
            return jp_list[0][1]
        
        # 如果失败，尝试转繁体后查询
        if has_opencc:
            trad_char = s2t.convert(char)
            if trad_char != char:
                jp_list = pycantonese.characters_to_jyutping(trad_char)
                if jp_list and jp_list[0][1]:
                    return jp_list[0][1]
        
        return None
    
    # 粤拼转换
    chars_jyutping = []
    for char in text:
        if '\u4e00' <= char <= '\u9fff':  # 中文字符
            jp = get_jyutping(char)
            if jp:
                chars_jyutping.append((char, jp))
            else:
                # 罕见字没有粤拼，用原字占位
                chars_jyutping.append((char, f"[{char}]"))
        else:
            # 非中文字符（标点等）
            chars_jyutping.append((char, None))
    
    # 展开 Whisper 词级时间戳为字级（多字词均分）
    char_timestamps = []
    for ts in timestamps:
        word = ts["word"].strip()
        if not word:
            continue
        
        # 提取词中的汉字
        chinese_chars = [c for c in word if '\u4e00' <= c <= '\u9fff']
        
        if len(chinese_chars) == 0:
            # 没有汉字，跳过
            continue
        elif len(chinese_chars) == 1:
            char_timestamps.append({"start": ts["start"], "end": ts["end"]})
        else:
            # 多字词，均分时间
            duration = ts["end"] - ts["start"]
            char_duration = duration / len(chinese_chars)
            for i in range(len(chinese_chars)):
                char_timestamps.append({
                    "start": ts["start"] + i * char_duration,
                    "end": ts["start"] + (i + 1) * char_duration
                })
    
    print(f"Whisper 时间戳数量: {len(char_timestamps)} 个")
    
    # 统计用户文字中的汉字数
    user_chinese_count = sum(1 for c, jp in chars_jyutping if jp)
    print(f"用户指定文字汉字数: {user_chinese_count} 个")
    
    # 按顺序分配时间戳给用户文字中的汉字
    result = []
    ts_idx = 0
    
    for char, jp in chars_jyutping:
        if jp:  # 有粤拼的字符（汉字）
            if ts_idx < len(char_timestamps):
                ts = char_timestamps[ts_idx]
                result.append({
                    "char": char,
                    "jyutping": jp,
                    "start": ts["start"],
                    "end": ts["end"]
                })
                ts_idx += 1
            else:
                # 时间戳用完，多出的字不分配时间
                result.append({
                    "char": char,
                    "jyutping": jp,
                    "start": None,
                    "end": None
                })
        else:
            # 标点等不分配时间戳
            result.append({
                "char": char,
                "jyutping": None,
                "start": None,
                "end": None
            })
    
    if ts_idx < len(char_timestamps):
        print(f"提示: 还有 {len(char_timestamps) - ts_idx} 个时间戳未使用")
    if user_chinese_count > len(char_timestamps):
        print(f"提示: {user_chinese_count - len(char_timestamps)} 个字未分配时间戳")
    
    return result


def format_yuefix_output(aligned_data: list, include_all_times: bool = False) -> str:
    """
    格式化为yue-fix文件格式
    
    Args:
        aligned_data: 对齐后的数据
        include_all_times: 是否为所有字都添加时间戳
    """
    lines = []
    lines.append("# yue-fix 格式模板（自动提取时间戳）")
    lines.append("# 格式: 粤拼 开始时间(秒) [结束时间(秒)]  # 原字")
    lines.append("# 注意: 自动提取的时间可能不准确，请根据实际情况调整")
    lines.append("# 只有添加了时间戳的行才会被处理，删除时间戳可跳过该字")
    lines.append("")
    
    for item in aligned_data:
        if item["jyutping"]:
            jp = item["jyutping"]
            char = item["char"]
            
            if item["start"] is not None and include_all_times:
                # 带时间戳
                start = item["start"]
                end = item["end"]
                duration = end - start
                lines.append(f"{jp} {start:.2f} {end:.2f}  # {char} [{duration:.2f}s]")
            else:
                # 不带时间戳（默认）
                if item["start"] is not None:
                    # 时间作为注释参考
                    start = item["start"]
                    end = item["end"]
                    duration = end - start
                    lines.append(f"{jp}  # {char} (参考: {start:.2f} {end:.2f}) [{duration:.2f}s]")
                else:
                    lines.append(f"{jp}  # {char}")
        elif item["char"].strip():
            # 标点等
            lines.append(f"# [{item['char']}]")
    
    return "\n".join(lines)


def interactive_mode():
    """
    交互模式：通过弹窗选择文件和配置参数
    """
    try:
        import tkinter as tk
        from tkinter import filedialog, messagebox, simpledialog
    except ImportError:
        print("错误: 交互模式需要 tkinter 支持")
        print("请使用命令行模式，或安装带 tkinter 的 Python")
        sys.exit(1)
    
    # 初始化 tkinter
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    print("=" * 50)
    print("粤语发音时间提取工具 - 交互模式")
    print("=" * 50)
    
    # 1. 选择音频文件
    print("\n请选择音频文件...")
    audio_path = filedialog.askopenfilename(
        title="选择音频文件",
        filetypes=[
            ("Audio Files", "*.wav *.mp3 *.flac *.m4a *.ogg"),
            ("WAV Files", "*.wav"),
            ("MP3 Files", "*.mp3"),
            ("All Files", "*.*")
        ]
    )
    
    if not audio_path:
        print("已取消")
        root.destroy()
        return
    
    print(f"已选择: {audio_path}")
    
    # 2. 输入文字内容（可选）
    text_content = simpledialog.askstring(
        "输入文字",
        "请输入音频对应的文字内容（留空则自动识别）:",
        parent=root
    )
    text_content = text_content.strip() if text_content else None
    
    # 3. 选择输出目录
    print("\n请选择输出目录...")
    
    output_dir = filedialog.askdirectory(
        title="选择输出目录"
    )
    
    if not output_dir:
        print("已取消")
        root.destroy()
        return
    
    print(f"已选择目录: {output_dir}")
    
    # 根据音频文件名生成输出文件名
    audio_basename = os.path.basename(audio_path)  # 例如: "0018.00.wav"
    output_filename = f"{audio_basename}-fix.txt"  # 例如: "0018.00.wav-fix.txt"
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"输出文件: {output_path}")
    
    # 4. 选择模型大小
    model_choices = ["tiny", "base", "small", "medium", "large"]
    model_size = simpledialog.askstring(
        "Whisper模型",
        "选择Whisper模型大小 (tiny/base/small/medium/large)\n默认: small",
        initialvalue="small",
        parent=root
    )
    if not model_size or model_size not in model_choices:
        model_size = "small"
    
    # 5. 是否直接带时间戳
    with_times = messagebox.askyesno(
        "时间戳选项",
        "是否直接在模板中包含时间戳？\n\n"
        "是: 粤拼后直接带时间 (nei5 0.00 0.18  # 你)\n"
        "否: 时间作为注释参考 (nei5  # 你 (参考: 0.00 0.18))",
        parent=root
    )
    
    root.destroy()
    
    # 6. 参数确认
    print("\n" + "=" * 50)
    print("参数确认")
    print("=" * 50)
    print(f"音频文件: {audio_path}")
    print(f"文字内容: {text_content if text_content else '(自动识别)'}")
    print(f"输出文件: {output_path}")
    print(f"Whisper模型: {model_size}")
    print(f"直接带时间戳: {'是' if with_times else '否'}")
    print("=" * 50)
    
    # 7. 生成等效命令行
    cmd_parts = ["python", "tools/yue_time_extractor.py"]
    cmd_parts.extend(["--audio", f'"{audio_path}"'])
    if text_content:
        cmd_parts.extend(["--text", f'"{text_content}"'])
    cmd_parts.extend(["--output", f'"{output_path}"'])
    cmd_parts.extend(["--model", model_size])
    if with_times:
        cmd_parts.append("--with-times")
    
    equivalent_cmd = " ".join(cmd_parts)
    print("\n等效命令行:")
    print(equivalent_cmd)
    print()
    
    # 8. 确认执行
    confirm = input("是否执行？ [确定/取消] (Y/n): ").strip().lower()
    if confirm in ['n', 'no', '取消', '否']:
        print("已取消")
        return
    
    print()
    
    # 返回参数用于执行
    return {
        "audio": audio_path,
        "text": text_content,
        "output": output_path,
        "model": model_size,
        "with_times": with_times,
        "mode": "whisper",
        "language": "yue"  # 默认粤语，会自动降级为zh
    }


def run_extraction(audio: str, output: str, text: str = None, 
                   model: str = "small", mode: str = "whisper",
                   with_times: bool = False, language: str = "yue"):
    """执行时间提取"""
    # 检查依赖并确定 whisper 类型
    whisper_type = check_dependencies()
    
    print(f"音频文件: {audio}")
    print("-" * 50)
    
    if mode == "whisper":
        # Whisper模式
        whisper_model, model_type = load_whisper_model(model, whisper_type)
        timestamps, recognized_text = extract_word_timestamps(
            audio, whisper_model, model_type, language
        )
        
        print(f"识别文本: {recognized_text}")
        print(f"提取到 {len(timestamps)} 个时间点")
        
        # 使用指定文字或识别结果
        final_text = text if text else recognized_text
        
    else:
        # VAD模式
        timestamps = extract_using_vad(audio)
        print(f"检测到 {len(timestamps)} 个音节边界")
        
        if not text:
            print("警告: VAD模式需要指定 --text 参数")
            final_text = ""
        else:
            final_text = text
    
    if final_text:
        # 转换为粤拼并对齐
        aligned = text_to_jyutping_with_alignment(final_text, timestamps)
        
        # 格式化输出
        output_content = format_yuefix_output(aligned, with_times)
    else:
        # 仅输出时间戳
        lines = ["# 提取的时间戳（无文字）", ""]
        for i, ts in enumerate(timestamps):
            lines.append(f"# 音节{i+1}: {ts['start']:.2f}s - {ts['end']:.2f}s  {ts['word']}")
        output_content = "\n".join(lines)
    
    print("-" * 50)
    
    if output:
        # 确保输出目录存在
        output_dir = os.path.dirname(output)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(output, 'w', encoding='utf-8') as f:
            f.write(output_content)
        print(f"已保存到: {output}")
    else:
        print(output_content)
    
    print("\n提示: 自动提取的时间可能不准确，建议使用Audacity等工具校对")


def main():
    parser = argparse.ArgumentParser(
        description="粤语发音时间提取工具 - 自动提取音节时间戳",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --interactive                             # 交互模式（弹窗选择文件）
  %(prog)s --audio input.wav                         # 自动识别并提取时间
  %(prog)s --audio input.wav --text "你好世界"        # 使用指定文字
  %(prog)s --audio input.wav --output fix.txt        # 输出到文件
  %(prog)s --audio input.wav --with-times            # 模板直接带时间戳
  %(prog)s --audio input.wav --mode vad              # 使用VAD模式
        """
    )
    
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="交互模式（弹窗选择文件）")
    parser.add_argument("--audio", "-a", type=str, default=None,
                        help="输入音频文件路径")
    parser.add_argument("--text", "-t", type=str, default=None,
                        help="指定文字内容（不指定则自动识别）")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出文件路径")
    parser.add_argument("--model", "-m", type=str, default="small",
                        choices=["tiny", "base", "small", "medium", "large"],
                        help="Whisper模型大小 (默认: small)")
    parser.add_argument("--mode", type=str, default="whisper",
                        choices=["whisper", "vad"],
                        help="时间提取模式 (默认: whisper)")
    parser.add_argument("--with-times", action="store_true",
                        help="模板直接包含时间戳（默认时间作为注释参考）")
    parser.add_argument("--language", type=str, default="yue",
                        help="语言代码 (默认: yue 粤语，不支持时自动降级为zh)")
    
    args = parser.parse_args()
    
    # 交互模式
    if args.interactive:
        params = interactive_mode()
        if params is None:
            return
        
        run_extraction(
            audio=params["audio"],
            output=params["output"],
            text=params["text"],
            model=params["model"],
            mode=params["mode"],
            with_times=params["with_times"],
            language=params["language"]
        )
        return
    
    # 命令行模式
    if not args.audio:
        parser.print_help()
        print("\n错误: 请指定 --audio 或使用 --interactive 交互模式")
        sys.exit(1)
    
    # 检查输入文件
    if not os.path.exists(args.audio):
        print(f"错误: 音频文件不存在 - {args.audio}")
        sys.exit(1)
    
    run_extraction(
        audio=args.audio,
        output=args.output,
        text=args.text,
        model=args.model,
        mode=args.mode,
        with_times=args.with_times,
        language=args.language
    )


if __name__ == "__main__":
    main()
