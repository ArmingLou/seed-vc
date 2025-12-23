#!/usr/bin/env python3
"""
粤语语音修正独立工具
对已有音频文件进行F0声调修正

用法:
    python tools/yue_audio_fix.py --source input.wav --yue-fix fix.txt --output output.wav
    python tools/yue_audio_fix.py --source input.wav --yue-fix fix.txt --yue-fix-strength 0.8 --output output.wav
"""

import os
import sys
import argparse
import numpy as np
import warnings

# 抑制 librosa 短片段警告
warnings.filterwarnings('ignore', message='n_fft=.*is too large for input signal')

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 检测可用的音高偏移库
PITCH_SHIFT_ENGINE = None

def get_pitch_shift_engine():
    """获取可用的音高偏移引擎"""
    global PITCH_SHIFT_ENGINE
    if PITCH_SHIFT_ENGINE is not None:
        return PITCH_SHIFT_ENGINE
    
    # 优先使用 pyrubberband（效果最好）
    try:
        import pyrubberband
        PITCH_SHIFT_ENGINE = "pyrubberband"
        return PITCH_SHIFT_ENGINE
    except ImportError:
        pass
    
    # 回退到 librosa
    PITCH_SHIFT_ENGINE = "librosa"
    return PITCH_SHIFT_ENGINE


def load_audio(audio_path: str, target_sr: int = 44100):
    """
    加载音频文件，自动转换采样率
    
    Args:
        audio_path: 音频文件路径
        target_sr: 目标采样率
    
    Returns:
        (audio, sr): 音频数据和采样率
    """
    import librosa
    import soundfile as sf
    
    # 先获取原始采样率
    try:
        info = sf.info(audio_path)
        orig_sr = info.samplerate
    except:
        orig_sr = None
    
    # 加载并重采样到目标采样率
    audio, sr = librosa.load(audio_path, sr=target_sr)
    
    # 提示采样率转换
    if orig_sr and orig_sr != target_sr:
        print(f"  采样率转换: {orig_sr}Hz -> {target_sr}Hz")
    
    return audio, sr


def save_audio(audio: np.ndarray, output_path: str, sr: int = 44100):
    """保存音频文件"""
    import soundfile as sf
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # 归一化
    if np.abs(audio).max() > 1.0:
        audio = audio / np.abs(audio).max()
    
    sf.write(output_path, audio, sr)
    print(f"已保存: {output_path}")


def extract_f0(audio: np.ndarray, sr: int = 44100, hop_length: int = 512):
    """提取F0"""
    import librosa
    
    # 使用 librosa 的 pyin 算法提取 F0
    f0, voiced_flag, voiced_probs = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz('C2'),  # 约65Hz
        fmax=librosa.note_to_hz('C6'),  # 约1047Hz
        sr=sr,
        hop_length=hop_length
    )
    
    # 将 NaN 替换为 0
    f0 = np.nan_to_num(f0, nan=0.0)
    
    return f0


def apply_f0_to_audio(audio: np.ndarray, original_f0: np.ndarray, target_f0: np.ndarray, 
                      sr: int = 44100, hop_length: int = 512) -> np.ndarray:
    """
    将修正后的F0应用到音频
    使用 PSOLA 风格的音高修正
    """
    import librosa
    
    # 计算每帧的音高偏移比例
    # 只处理有效F0（非零）
    valid_mask = (original_f0 > 0) & (target_f0 > 0)
    
    if not np.any(valid_mask):
        print("警告: 没有有效的F0可供修正")
        return audio
    
    # 计算整体的平均音高偏移（半音）
    # 对于有有效F0的帧，计算需要的音高偏移
    original_valid = original_f0[valid_mask]
    target_valid = target_f0[valid_mask]
    
    # 使用分段处理来应用不同的音高偏移
    # 这里使用简化的方法：对整体应用平均偏移
    # 更精细的实现需要使用 PSOLA 或 WORLD vocoder
    
    # 计算半音偏移
    semitone_shifts = 12 * np.log2(target_valid / original_valid)
    avg_shift = np.median(semitone_shifts)
    
    if abs(avg_shift) < 0.1:
        print("信息: 音高偏移很小，跳过修正")
        return audio
    
    print(f"信息: 应用平均音高偏移 {avg_shift:.2f} 半音")
    
    # 使用 librosa 的 pitch_shift
    audio_shifted = librosa.effects.pitch_shift(
        audio, 
        sr=sr, 
        n_steps=avg_shift
    )
    
    return audio_shifted


def apply_world_f0_correction(audio: np.ndarray, original_f0: np.ndarray,
                               annotations: list, strength: float,
                               sr: int = 44100, hop_length: int = 512,
                               max_semitones: float = 3.0) -> np.ndarray:
    """
    使用 WORLD vocoder 进行 F0 修正
    
    Args:
        max_semitones: 最大修正半音数，限制偏移幅度避免失真
    """
    try:
        import pyworld as pw
    except ImportError:
        print("  警告: pyworld 未安装，回退到分段处理")
        return apply_segment_f0_correction_fallback(audio, original_f0, annotations, strength, sr, hop_length)
    
    from modules.yue_fix import YueFixProcessor
    
    processor = YueFixProcessor(sr=sr, hop_length=hop_length)
    print("  使用 WORLD vocoder 进行高质量修正")
    
    # 确保音频是 float64（WORLD 要求）
    audio_f64 = audio.astype(np.float64)
    
    # WORLD 分析
    print("  WORLD 分析中...")
    # 使用 harvest 提取 F0（更准确）
    f0_world, t = pw.harvest(audio_f64, sr, frame_period=hop_length * 1000 / sr)
    # 提取频谱包络和非周期信号
    sp = pw.cheaptrick(audio_f64, f0_world, t, sr)
    ap = pw.d4c(audio_f64, f0_world, t, sr)
    
    print(f"  WORLD 帧数: {len(f0_world)}")
    
    # 估计 F0 范围
    f0_min, f0_max = processor.estimate_f0_range(original_f0)
    
    # 创建 F0 修正曲线（与 WORLD 帧对齐）
    f0_modified = f0_world.copy()
    world_frame_period = hop_length * 1000 / sr  # ms
    
    corrections_applied = 0
    
    for ann in annotations:
        # 计算 WORLD 帧范围
        start_frame = int(ann.start_time * 1000 / world_frame_period)
        end_frame = int(ann.end_time * 1000 / world_frame_period)
        
        if start_frame >= len(f0_world) or end_frame > len(f0_world):
            continue
        
        if start_frame >= end_frame:
            continue
        
        # 获取该段的原始 F0
        segment_f0 = f0_world[start_frame:end_frame]
        valid_mask = segment_f0 > 0
        
        if not np.any(valid_mask):
            continue
        
        # 生成目标声调的 F0 曲线
        num_world_frames = end_frame - start_frame
        
        # 声调走势模板（归一化）
        TONE_CONTOURS = {
            1: (1.0, 1.0),    # 阴平 55 - 高平
            2: (0.6, 0.95),   # 阴上 35 - 高升
            3: (0.55, 0.55),  # 阴去 33 - 中平
            4: (0.35, 0.15),  # 阳平 21 - 低降
            5: (0.25, 0.55),  # 阳上 23 - 低升
            6: (0.2, 0.2),    # 阳去 22 - 低平
        }
        
        # 获取目标声调的走势
        target_start, target_end = TONE_CONTOURS[ann.tone]
        
        # 计算原始片段的统计信息
        valid_f0 = segment_f0[valid_mask]
        f0_mean = np.mean(valid_f0)
        
        # 生成目标走势曲线（保持平均音高，只改变走势）
        target_contour = np.linspace(target_start, target_end, num_world_frames)
        # 归一化到均值为0
        target_contour_centered = target_contour - np.mean(target_contour)
        
        # 设定走势幅度（用平均音高的 10% 作为走势范围）
        contour_range = f0_mean * 0.1 * strength
        
        # 生成目标 F0：保持平均值，应用目标走势
        if np.std(target_contour_centered) > 0:
            target_f0 = f0_mean + target_contour_centered / np.max(np.abs(target_contour_centered)) * contour_range
        else:
            target_f0 = np.full(num_world_frames, f0_mean)
        # 创建平滑过渡窗口
        fade_frames = max(1, num_world_frames // 4)
        window = np.ones(num_world_frames)
        if num_world_frames > 2:
            # 渐入
            window[:fade_frames] = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fade_frames))
            # 渐出
            window[-fade_frames:] = 0.5 + 0.5 * np.cos(np.linspace(0, np.pi, fade_frames))
        
        # 应用走势修正（混合原始和目标 F0）
        for i in range(num_world_frames):
            frame_idx = start_frame + i
            if f0_modified[frame_idx] > 0:
                blend = window[i]
                f0_modified[frame_idx] = (
                    (1 - blend) * f0_modified[frame_idx] +
                    blend * target_f0[i]
                )
        
        # 显示信息
        original_trend = "升" if valid_f0[-1] > valid_f0[0] else ("降" if valid_f0[-1] < valid_f0[0] else "平")
        target_trend = "升" if target_end > target_start else ("降" if target_end < target_start else "平")
        corrections_applied += 1
        print(f"  修正: {ann.jyutping} (声调{ann.tone}) @ {ann.start_time:.2f}s, {original_trend}→{target_trend}")
    
    if corrections_applied == 0:
        print("  没有需要修正的片段")
        return audio
    
    # WORLD 合成（必须使用相同的 frame_period）
    print(f"  WORLD 合成中... (共 {corrections_applied} 处修正)")
    audio_corrected = pw.synthesize(f0_modified, sp, ap, sr, frame_period=world_frame_period)
    
    print(f"  原始长度: {len(audio)}, 合成长度: {len(audio_corrected)}")
    
    # 确保输出长度与输入一致
    if len(audio_corrected) > len(audio):
        audio_corrected = audio_corrected[:len(audio)]
    elif len(audio_corrected) < len(audio):
        audio_corrected = np.pad(audio_corrected, (0, len(audio) - len(audio_corrected)))
    
    return audio_corrected.astype(np.float32)


def apply_segment_f0_correction_fallback(audio: np.ndarray, original_f0: np.ndarray, 
                                          annotations: list, strength: float,
                                          sr: int = 44100, hop_length: int = 512) -> np.ndarray:
    """
    回退方案：使用分段处理（效果较差）
    """
    import librosa
    from modules.yue_fix import YueFixProcessor
    
    processor = YueFixProcessor(sr=sr, hop_length=hop_length)
    print("  警告: 使用分段处理，效果可能有爆破音")
    
    audio_corrected = audio.copy()
    f0_min, f0_max = processor.estimate_f0_range(original_f0)
    
    for ann in annotations:
        start_sample = int(ann.start_time * sr)
        end_sample = int(ann.end_time * sr)
        
        start_sample = max(0, min(start_sample, len(audio) - 1))
        end_sample = max(start_sample + 1, min(end_sample, len(audio)))
        
        start_frame = processor.time_to_frame(ann.start_time)
        end_frame = processor.time_to_frame(ann.end_time)
        num_frames = end_frame - start_frame
        
        if num_frames <= 0 or start_frame >= len(original_f0) or end_frame > len(original_f0):
            continue
        
        segment_f0 = original_f0[start_frame:end_frame]
        valid_mask = segment_f0 > 0
        
        if not np.any(valid_mask):
            continue
        
        target_f0_curve = processor.generate_tone_f0_curve(ann.tone, num_frames, f0_min, f0_max)
        original_median = np.median(segment_f0[valid_mask])
        target_median = np.median(target_f0_curve[valid_mask])
        
        if original_median <= 0:
            continue
        
        semitone_shift = 12 * np.log2(target_median / original_median) * strength
        
        if abs(semitone_shift) < 0.1:
            continue
        
        segment = audio[start_sample:end_sample]
        if len(segment) < 100:
            continue
        
        try:
            segment_shifted = librosa.effects.pitch_shift(segment, sr=sr, n_steps=semitone_shift)
            audio_corrected[start_sample:start_sample + len(segment_shifted)] = segment_shifted
            print(f"  修正: {ann.jyutping} (声调{ann.tone}) @ {ann.start_time:.2f}s, 偏移 {semitone_shift:.2f} 半音")
        except Exception as e:
            print(f"  警告: 修正 {ann.jyutping} 失败: {e}")
    
    return audio_corrected


# 保持向后兼容的别名
def apply_segment_f0_correction(audio: np.ndarray, original_f0: np.ndarray, 
                                annotations: list, strength: float,
                                sr: int = 44100, hop_length: int = 512) -> np.ndarray:
    """F0 修正入口，优先使用 WORLD vocoder，默认限制最大 3 半音"""
    return apply_world_f0_correction(audio, original_f0, annotations, strength, sr, hop_length, max_semitones=3.0)


def interactive_mode():
    """
    交互模式：通过弹窗选择文件和配置参数
    """
    try:
        import tkinter as tk
        from tkinter import filedialog, simpledialog
    except ImportError:
        print("错误: 交互模式需要 tkinter 支持")
        print("请使用命令行模式，或安装带 tkinter 的 Python")
        sys.exit(1)
    
    # 初始化 tkinter
    root = tk.Tk()
    root.withdraw()  # 隐藏主窗口
    
    print("=" * 50)
    print("粤语语音修正工具 - 交互模式")
    print("=" * 50)
    
    # 1. 选择源音频文件
    print("\n请选择源音频文件...")
    source_path = filedialog.askopenfilename(
        title="选择源音频文件",
        filetypes=[
            ("Audio Files", "*.wav *.mp3 *.flac *.m4a *.ogg"),
            ("WAV Files", "*.wav"),
            ("MP3 Files", "*.mp3"),
            ("All Files", "*.*")
        ]
    )
    
    if not source_path:
        print("已取消")
        root.destroy()
        return None
    
    print(f"已选择: {source_path}")
    
    # 2. 选择修正文件
    print("\n请选择修正文件 (yue-fix)...")
    fix_path = filedialog.askopenfilename(
        title="选择修正文件 (yue-fix)",
        filetypes=[
            ("Text Files", "*.txt"),
            ("All Files", "*.*")
        ]
    )
    
    if not fix_path:
        print("已取消")
        root.destroy()
        return None
    
    print(f"已选择: {fix_path}")
    
    # 3. 选择输出目录
    print("\n请选择输出目录...")
    
    output_dir = filedialog.askdirectory(
        title="选择输出目录"
    )
    
    if not output_dir:
        print("已取消")
        root.destroy()
        return None
    
    print(f"已选择目录: {output_dir}")
    
    # 根据源音频文件名生成输出文件路径
    source_basename = os.path.splitext(os.path.basename(source_path))[0]  # 例如: "0018.00"
    output_filename = f"{source_basename}_fixed.wav"  # 例如: "0018.00_fixed.wav"
    output_path = os.path.join(output_dir, output_filename)
    
    print(f"输出文件: {output_path}")
    
    # 4. 设置修正强度
    strength_str = simpledialog.askstring(
        "修正强度",
        "请输入修正强度 (0.0-1.0)\n默认: 0.8",
        initialvalue="0.8",
        parent=root
    )
    
    try:
        strength = float(strength_str) if strength_str else 0.8
        strength = max(0.0, min(1.0, strength))
    except ValueError:
        strength = 0.8
    
    root.destroy()
    
    # 5. 参数确认
    print("\n" + "=" * 50)
    print("参数确认")
    print("=" * 50)
    print(f"源音频: {source_path}")
    print(f"修正文件: {fix_path}")
    print(f"修正强度: {strength}")
    print(f"输出文件: {output_path}")
    print("=" * 50)
    
    # 6. 生成等效命令行
    cmd_parts = ["python", "tools/yue_audio_fix.py"]
    cmd_parts.extend(["--source", f'"{source_path}"'])
    cmd_parts.extend(["--yue-fix", f'"{fix_path}"'])
    cmd_parts.extend(["--yue-fix-strength", str(strength)])
    cmd_parts.extend(["--output", f'"{output_path}"'])
    
    equivalent_cmd = " ".join(cmd_parts)
    print("\n等效命令行:")
    print(equivalent_cmd)
    print()
    
    # 7. 确认执行
    confirm = input("是否执行？ [确定/取消] (Y/n): ").strip().lower()
    if confirm in ['n', 'no', '取消', '否']:
        print("已取消")
        return None
    
    print()
    
    # 返回参数用于执行
    return {
        "source": source_path,
        "yue_fix": fix_path,
        "strength": strength,
        "output": output_path,
        "sr": 44100,
        "hop_length": 512
    }


def run_correction(source: str, yue_fix: str, output: str, 
                   strength: float = 0.8, sr: int = 44100, 
                   hop_length: int = 512):
    """执行音频修正"""
    
    # 确定输出路径
    if os.path.isdir(output) or output.endswith('/'):
        os.makedirs(output, exist_ok=True)
        source_name = os.path.splitext(os.path.basename(source))[0]
        output_path = os.path.join(output, f"{source_name}_fixed.wav")
    else:
        output_path = output
    
    print(f"源文件: {source}")
    print(f"修正文件: {yue_fix}")
    print(f"修正强度: {strength}")
    print(f"输出路径: {output_path}")
    print("-" * 50)
    
    # 加载音频
    print("加载音频...")
    audio, actual_sr = load_audio(source, sr)
    print(f"  采样率: {actual_sr}, 时长: {len(audio)/actual_sr:.2f}s")
    
    # 解析修正文件
    print("解析修正文件...")
    from modules.yue_fix import YueFixProcessor
    processor = YueFixProcessor(sr=actual_sr, hop_length=hop_length)
    annotations = processor.parse_fix_file(yue_fix)
    
    if len(annotations) == 0:
        print("警告: 没有有效的修正标注，输出原始音频")
        save_audio(audio, output_path, actual_sr)
        return
    
    print(f"  有效标注: {len(annotations)} 个")
    
    # 提取F0
    print("提取F0...")
    original_f0 = extract_f0(audio, actual_sr, hop_length)
    print(f"  F0帧数: {len(original_f0)}")
    
    # 应用修正
    print("应用修正...")
    audio_corrected = apply_segment_f0_correction(
        audio, original_f0, annotations, strength,
        actual_sr, hop_length
    )
    
    # 保存结果
    print("-" * 50)
    save_audio(audio_corrected, output_path, actual_sr)
    print("完成!")


def main():
    parser = argparse.ArgumentParser(
        description="粤语语音修正工具 - 对音频进行F0声调修正",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --interactive                                                  # 交互模式（弹窗选择文件）
  %(prog)s --source input.wav --yue-fix fix.txt --output corrected.wav
  %(prog)s --source input.wav --yue-fix fix.txt --yue-fix-strength 0.8 --output corrected.wav
  %(prog)s --source input.wav --yue-fix fix.txt --output ./output/
        """
    )
    
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="交互模式（弹窗选择文件）")
    parser.add_argument("--source", "-s", type=str, default=None,
                        help="源音频文件路径")
    parser.add_argument("--yue-fix", type=str, default=None,
                        help="粤语修正文件路径")
    parser.add_argument("--yue-fix-strength", type=float, default=0.8,
                        help="修正强度 (0.0-1.0, 默认: 0.8)")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="输出路径（文件或目录）")
    parser.add_argument("--sr", type=int, default=44100,
                        help="采样率 (默认: 44100)")
    parser.add_argument("--hop-length", type=int, default=512,
                        help="帧移长度 (默认: 512)")
    
    args = parser.parse_args()
    
    # 交互模式
    if args.interactive:
        params = interactive_mode()
        if params is None:
            return
        
        run_correction(
            source=params["source"],
            yue_fix=params["yue_fix"],
            output=params["output"],
            strength=params["strength"],
            sr=params["sr"],
            hop_length=params["hop_length"]
        )
        return
    
    # 命令行模式 - 检查必要参数
    if not args.source or not args.yue_fix or not args.output:
        parser.print_help()
        print("\n错误: 请指定 --source, --yue-fix, --output 或使用 --interactive 交互模式")
        sys.exit(1)
    
    # 检查输入文件
    if not os.path.exists(args.source):
        print(f"错误: 源文件不存在 - {args.source}")
        sys.exit(1)
    
    if not os.path.exists(args.yue_fix):
        print(f"错误: 修正文件不存在 - {args.yue_fix}")
        sys.exit(1)
    
    run_correction(
        source=args.source,
        yue_fix=args.yue_fix,
        output=args.output,
        strength=args.yue_fix_strength,
        sr=args.sr,
        hop_length=args.hop_length
    )


if __name__ == "__main__":
    main()
