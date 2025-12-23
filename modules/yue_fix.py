"""
粤语发音/声调后处理修正模块
用于修正语音转换后的粤语发音和声调问题

粤语六声九调系统：
    舒声（6种调型）：
        声调1 阴平 (55 高平)  - 诗 si1
        声调2 阴上 (35 高升)  - 史 si2
        声调3 阴去 (33 中平)  - 试 si3
        声调4 阳平 (21 低降)  - 时 si4
        声调5 阳上 (23 低升)  - 市 si5
        声调6 阳去 (22 低平)  - 事 si6
    
    入声（3种，音节短促，以 -p/-t/-k 结尾）：
        阴入 (高入) - 粤拼用 1 标记，如 识 sik1
        中入       - 粤拼用 3 标记，如 锡 sek3
        阳入 (低入) - 粤拼用 6 标记，如 食 sik6
    
    入声与对应舒声的调型相同，主要区别在于音节时长更短。
"""

import numpy as np
import librosa
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import re


@dataclass
class YueFixAnnotation:
    """粤语修正标注"""
    jyutping: str       # 粤拼（如 nei5）
    tone: int           # 声调 1-6
    start_time: float   # 开始时间（秒）
    end_time: float     # 结束时间（秒），可选
    character: str      # 原字符（可选，用于调试）


class YueFixProcessor:
    """粤语发音/声调后处理修正器"""
    
    # 粤语六声九调的F0曲线模板（归一化到 0-1 范围）
    # 每个模板是 [起始相对音高, 结束相对音高]
    # 入声与对应舒声调型相同，但时长更短
    TONE_TEMPLATES = {
        1: (1.0, 1.0),    # 阴平 55 - 高平调 (也用于阴入)
        2: (0.6, 0.95),   # 阴上 35 - 高升调
        3: (0.55, 0.55),  # 阴去 33 - 中平调 (也用于中入)
        4: (0.35, 0.15),  # 阳平 21 - 低降调
        5: (0.25, 0.55),  # 阳上 23 - 低升调
        6: (0.2, 0.2),    # 阳去 22 - 低平调 (也用于阳入)
    }
    
    # F0范围参考值（Hz）- 可根据说话人调整
    DEFAULT_F0_MIN = 80.0
    DEFAULT_F0_MAX = 400.0
    
    def __init__(self, sr: int = 44100, hop_length: int = 512):
        """
        初始化修正器
        
        Args:
            sr: 采样率
            hop_length: 帧移长度
        """
        self.sr = sr
        self.hop_length = hop_length
        self.frame_duration = hop_length / sr  # 每帧时长（秒）
    
    def parse_fix_file(self, fix_file_path: str) -> List[YueFixAnnotation]:
        """
        解析修正文件
        
        文件格式示例:
            nei5 0.00 0.15  # 你
            hou2 0.15 0.30  # 好
            
        或简化格式（只有开始时间）:
            nei5 0.00  # 你
            hou2 0.15  # 好
        
        注意: 只有拼音没有时间的行会被自动跳过（不处理）
        
        Returns:
            List of YueFixAnnotation
        """
        annotations = []
        skipped_count = 0
        
        with open(fix_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        prev_annotation = None
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            
            # 跳过空行和注释行
            if not line or line.startswith('#'):
                continue
            
            # 提取注释中的原字符
            character = ""
            if '#' in line:
                parts = line.split('#', 1)
                line = parts[0].strip()
                character = parts[1].strip()
            
            # 解析主要内容
            parts = line.split()
            if len(parts) < 1:
                continue
            
            jyutping = parts[0]
            
            # 检查是否有时间戳
            if len(parts) < 2:
                # 只有拼音没有时间，跳过不处理
                skipped_count += 1
                continue
            
            # 尝试解析时间
            try:
                start_time = float(parts[1])
            except ValueError:
                # 时间格式无效，跳过
                skipped_count += 1
                continue
            
            # 可选的结束时间
            end_time = None
            if len(parts) > 2:
                try:
                    end_time = float(parts[2])
                except ValueError:
                    pass  # 结束时间解析失败，使用自动推断
            
            # 解析声调（粤拼最后一个数字）
            tone = self._extract_tone(jyutping)
            if tone is None:
                print(f"警告: 第{line_num}行无法识别声调，跳过: {jyutping}")
                continue
            
            # 如果前一个标注没有结束时间，用当前开始时间作为其结束时间
            if prev_annotation and prev_annotation.end_time is None:
                prev_annotation.end_time = start_time
            
            annotation = YueFixAnnotation(
                jyutping=jyutping,
                tone=tone,
                start_time=start_time,
                end_time=end_time,
                character=character
            )
            annotations.append(annotation)
            prev_annotation = annotation
        
        # 处理最后一个标注的结束时间
        if prev_annotation and prev_annotation.end_time is None:
            # 假设最后一个字持续0.3秒
            prev_annotation.end_time = prev_annotation.start_time + 0.3
        
        if skipped_count > 0:
            print(f"信息: 跳过了 {skipped_count} 个无时间戳的条目（不处理）")
        
        return annotations
    
    def _extract_tone(self, jyutping: str) -> Optional[int]:
        """从粤拼中提取声调数字"""
        match = re.search(r'(\d)$', jyutping)
        if match:
            tone = int(match.group(1))
            if 1 <= tone <= 6:
                return tone
        return None
    
    def time_to_frame(self, time_sec: float) -> int:
        """将时间（秒）转换为帧索引"""
        return int(time_sec / self.frame_duration)
    
    def frame_to_time(self, frame_idx: int) -> float:
        """将帧索引转换为时间（秒）"""
        return frame_idx * self.frame_duration
    
    def generate_tone_f0_curve(
        self,
        tone: int,
        num_frames: int,
        f0_min: float,
        f0_max: float
    ) -> np.ndarray:
        """
        根据声调生成目标F0曲线
        
        Args:
            tone: 声调 1-6
            num_frames: 帧数
            f0_min: F0最小值（Hz）
            f0_max: F0最大值（Hz）
            
        Returns:
            F0曲线数组
        """
        if tone not in self.TONE_TEMPLATES:
            raise ValueError(f"无效声调: {tone}")
        
        start_ratio, end_ratio = self.TONE_TEMPLATES[tone]
        
        # 生成线性过渡曲线
        ratios = np.linspace(start_ratio, end_ratio, num_frames)
        
        # 转换为实际F0值
        f0_range = f0_max - f0_min
        f0_curve = f0_min + ratios * f0_range
        
        return f0_curve
    
    def estimate_f0_range(self, f0: np.ndarray) -> Tuple[float, float]:
        """
        估计音频的F0范围
        
        Args:
            f0: F0数组
            
        Returns:
            (f0_min, f0_max)
        """
        # 只考虑有效的F0值（非零）
        valid_f0 = f0[f0 > 0]
        if len(valid_f0) == 0:
            return self.DEFAULT_F0_MIN, self.DEFAULT_F0_MAX
        
        # 使用百分位数避免异常值影响
        f0_min = np.percentile(valid_f0, 10)
        f0_max = np.percentile(valid_f0, 90)
        
        return float(f0_min), float(f0_max)
    
    def apply_f0_correction(
        self,
        f0: np.ndarray,
        annotations: List[YueFixAnnotation],
        strength: float = 0.8
    ) -> np.ndarray:
        """
        应用F0声调修正
        
        Args:
            f0: 原始F0数组
            annotations: 修正标注列表
            strength: 修正强度 (0-1)，1表示完全替换，0表示不修正
            
        Returns:
            修正后的F0数组
        """
        if len(annotations) == 0:
            return f0
        
        f0_corrected = f0.copy()
        
        # 估计整体F0范围
        f0_min, f0_max = self.estimate_f0_range(f0)
        
        for ann in annotations:
            start_frame = self.time_to_frame(ann.start_time)
            end_frame = self.time_to_frame(ann.end_time)
            
            # 确保帧范围有效
            start_frame = max(0, min(start_frame, len(f0) - 1))
            end_frame = max(start_frame + 1, min(end_frame, len(f0)))
            
            num_frames = end_frame - start_frame
            if num_frames <= 0:
                continue
            
            # 生成目标F0曲线
            target_f0 = self.generate_tone_f0_curve(
                ann.tone, num_frames, f0_min, f0_max
            )
            
            # 获取原始片段
            original_segment = f0_corrected[start_frame:end_frame]
            
            # 混合原始和目标F0
            # 对于有声部分进行修正，无声部分保持不变
            voiced_mask = original_segment > 0
            
            corrected_segment = original_segment.copy()
            corrected_segment[voiced_mask] = (
                (1 - strength) * original_segment[voiced_mask] +
                strength * target_f0[voiced_mask]
            )
            
            f0_corrected[start_frame:end_frame] = corrected_segment
        
        return f0_corrected
    
    def apply_pitch_smoothing(
        self,
        f0: np.ndarray,
        window_size: int = 5
    ) -> np.ndarray:
        """
        对F0曲线进行平滑处理，减少突变
        
        Args:
            f0: F0数组
            window_size: 平滑窗口大小
            
        Returns:
            平滑后的F0数组
        """
        if window_size <= 1:
            return f0
        
        f0_smoothed = f0.copy()
        valid_mask = f0 > 0
        
        # 只对有效F0值进行平滑
        if np.sum(valid_mask) > window_size:
            # 使用移动平均平滑
            kernel = np.ones(window_size) / window_size
            f0_valid = np.convolve(f0, kernel, mode='same')
            f0_smoothed[valid_mask] = f0_valid[valid_mask]
        
        return f0_smoothed


def apply_yue_fix(
    audio: np.ndarray,
    f0: np.ndarray,
    fix_file_path: str,
    sr: int = 44100,
    hop_length: int = 512,
    strength: float = 0.8,
    smooth: bool = True
) -> np.ndarray:
    """
    应用粤语修正的便捷函数
    
    Args:
        audio: 音频数组（用于参考，暂不修改）
        f0: F0数组
        fix_file_path: 修正文件路径
        sr: 采样率
        hop_length: 帧移长度
        strength: 修正强度 (0-1)
        smooth: 是否平滑F0曲线
        
    Returns:
        修正后的F0数组
    """
    processor = YueFixProcessor(sr=sr, hop_length=hop_length)
    
    # 解析修正文件
    annotations = processor.parse_fix_file(fix_file_path)
    
    if len(annotations) == 0:
        print("警告: 修正文件中没有有效标注")
        return f0
    
    print(f"已加载 {len(annotations)} 个修正标注")
    
    # 应用F0修正
    f0_corrected = processor.apply_f0_correction(f0, annotations, strength)
    
    # 可选的平滑处理
    if smooth:
        f0_corrected = processor.apply_pitch_smoothing(f0_corrected)
    
    return f0_corrected


# 用于测试的示例代码
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试粤语修正模块")
    parser.add_argument("--fix-file", type=str, required=True, help="修正文件路径")
    parser.add_argument("--test", action="store_true", help="运行测试")
    args = parser.parse_args()
    
    processor = YueFixProcessor()
    annotations = processor.parse_fix_file(args.fix_file)
    
    print("解析结果:")
    for ann in annotations:
        print(f"  {ann.jyutping} (声调{ann.tone}): {ann.start_time:.2f}s - {ann.end_time:.2f}s  # {ann.character}")
    
    if args.test:
        # 生成测试F0曲线
        num_frames = 100
        test_f0 = np.random.uniform(150, 250, num_frames)
        test_f0[::10] = 0  # 模拟无声部分
        
        # 应用修正
        corrected_f0 = processor.apply_f0_correction(annotations, test_f0, strength=0.8)
        
        print(f"\n原始F0范围: {test_f0[test_f0 > 0].min():.1f} - {test_f0[test_f0 > 0].max():.1f}")
        print(f"修正后F0范围: {corrected_f0[corrected_f0 > 0].min():.1f} - {corrected_f0[corrected_f0 > 0].max():.1f}")
