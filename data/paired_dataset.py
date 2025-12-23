# 配对训练数据集 - 用于固定源说话人→固定目标说话人的场景
# 
# 重要：需要平行语料（同一句话的不同说话人录音）
#
# 目录结构:
# dataset/
# ├── source/     # 源说话人音频
# │   ├── 001.wav    # A说"今日天气好好"
# │   ├── 002.wav    # A说"我想食饭"
# │   └── ...
# └── target/     # 目标说话人音频（文件名对应，表示同一句话）
#     ├── 001.wav    # B说"今日天气好好"
#     ├── 002.wav    # B说"我想食饭"
#     └── ...
#
# 训练逻辑:
# - source 和 target 文件名对应，表示同一句话
# - 从 source 提取内容特征（语义）
# - 从 target 提取风格特征 + 目标 mel
# - 模型学习：source内容 + target风格 → target mel

import torch
import librosa
import numpy as np
import random
import os
import soundfile as sf
from torch.utils.data import DataLoader
from modules.audio import mel_spectrogram

duration_setting = {
    "min": 1.0,
    "max": 30.0,
}

def to_mel_fn(wave, mel_fn_args, file_path=None):
    return mel_spectrogram(wave, **mel_fn_args, file_path=file_path)


class PairedFTDataset(torch.utils.data.Dataset):
    """
    配对训练数据集 - 需要平行语料
    
    关键：source 和 target 文件名必须对应，表示同一句话。
    例如：source/001.wav 和 target/001.wav 是同一句话的不同说话人录音。
    
    推理时的三要素对应：
    - source = 推理中的"源音频"（提供内容）
    - reference = 推理中的"参考音频"（提供风格）
    - target = 推理中的"输出音频"（目标mel）
    
    支持两种参考模式：
    1. same_reference=True: 参考音频 = target 本身（风格和目标mel完全一致）
    2. same_reference=False: 参考音频 = target集合中的其他音频（style encoder更鲁棒）
    """
    def __init__(
        self,
        target_path,
        spect_params,
        sr=22050,
        batch_size=1,
        source_path=None,  # 可选的源说话人目录
        same_reference=True,  # True: 参考=target本身, False: 参考=target集合中的其他音频
    ):
        self.target_path = target_path
        self.source_path = source_path
        self.use_paired_mode = source_path is not None
        self.same_reference = same_reference
        
        print(f"Loading target data from: {target_path}")
        if self.use_paired_mode:
            print(f"Loading source data from: {source_path}")
            print("Mode: Parallel corpus training (source content + target style -> target mel)")
            if not same_reference:
                print("Reference mode: Random reference from target set (more robust style encoder)")
            else:
                print("Reference mode: Same as target (consistent style and mel)")
        else:
            print("Mode: Self-reconstruction (target only)")
        
        # 加载 target 音频（按文件名建立索引）
        self.target_files = self._load_audio_files_dict(target_path)
        if len(self.target_files) == 0:
            raise AssertionError(f"No valid audio files found in target path: {target_path}")
        print(f"Found {len(self.target_files)} target audio files")
        
        # 保存 target 文件列表（用于随机选择参考音频）
        self.target_list = list(self.target_files.values())
        
        # 加载 source 音频并匹配
        if self.use_paired_mode:
            self.source_files = self._load_audio_files_dict(source_path)
            if len(self.source_files) == 0:
                print(f"Warning: No source audio files found, falling back to self-reconstruction mode")
                self.use_paired_mode = False
            else:
                # 找到配对的文件（文件名相同）
                self._build_paired_data()
        
        if not self.use_paired_mode:
            # 自重建模式
            self.paired_data = [(path, path) for path in self.target_files.values()]
            self.source_files = {}
        
        # 采样率和 mel 参数
        self.sr = sr
        self.mel_fn_args = {
            "n_fft": spect_params['n_fft'],
            "win_size": spect_params.get('win_length', spect_params.get('win_size', 1024)),
            "hop_size": spect_params.get('hop_length', spect_params.get('hop_size', 256)),
            "num_mels": spect_params.get('n_mels', spect_params.get('n_mels', 80)),
            "sampling_rate": sr,
            "fmin": spect_params['fmin'],
            "fmax": None if spect_params['fmax'] == "None" else spect_params['fmax'],
            "center": False
        }
        
        # 确保数据足够批量大小
        while len(self.paired_data) < batch_size:
            self.paired_data += self.paired_data
        
        # epoch 相关
        self.current_epoch_indices = list(range(len(self.paired_data)))
        self.current_epoch = 0
    
    def _load_audio_files_dict(self, data_path):
        """加载目录中的所有音频文件，返回 {basename: full_path} 字典"""
        if not os.path.exists(data_path):
            print(f"Warning: Path does not exist: {data_path}")
            return {}
        
        audio_files = {}
        for root, _, files in os.walk(data_path):
            for file in files:
                if file.endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus")):
                    # 用文件名（不含扩展名）作为 key
                    basename = os.path.splitext(file)[0]
                    audio_files[basename] = os.path.join(root, file)
        
        return audio_files
    
    def _build_paired_data(self):
        """构建配对数据，匹配 source 和 target 文件名"""
        self.paired_data = []
        matched_count = 0
        unmatched_source = []
        unmatched_target = []
        
        # 找到同名的配对
        for basename, source_path in self.source_files.items():
            if basename in self.target_files:
                target_path = self.target_files[basename]
                self.paired_data.append((source_path, target_path))
                matched_count += 1
            else:
                unmatched_source.append(basename)
        
        for basename in self.target_files:
            if basename not in self.source_files:
                unmatched_target.append(basename)
        
        print(f"\n=== 配对结果 ===")
        print(f"成功配对: {matched_count} 对")
        if unmatched_source:
            print(f"source 未配对: {len(unmatched_source)} 个")
            if len(unmatched_source) <= 5:
                print(f"  {unmatched_source}")
        if unmatched_target:
            print(f"target 未配对: {len(unmatched_target)} 个")
            if len(unmatched_target) <= 5:
                print(f"  {unmatched_target}")
        print(f"================\n")
        
        if matched_count == 0:
            print("Warning: 没有找到配对的文件，退化为自重建模式")
            self.use_paired_mode = False
            self.paired_data = [(path, path) for path in self.target_files.values()]
    
    def _safe_load_audio(self, wav_path):
        """安全地加载音频文件"""
        try:
            speech, orig_sr = sf.read(wav_path)
            if speech.dtype != np.float32:
                speech = speech.astype(np.float32)
            if speech.ndim > 1:
                speech = np.mean(speech, axis=1)
            return speech, orig_sr
        except Exception as e:
            print(f"Soundfile failed to load {wav_path}: {e}")
        
        try:
            speech, orig_sr = librosa.load(wav_path, sr=None, mono=True)
            return speech, orig_sr
        except Exception as e:
            print(f"Librosa failed to load {wav_path}: {e}")
            raise e
    
    def _load_and_process_audio(self, wav_path):
        """加载并处理音频到目标采样率"""
        speech, orig_sr = self._safe_load_audio(wav_path)
        
        # 重采样
        if orig_sr != self.sr:
            speech = librosa.resample(speech, orig_sr=orig_sr, target_sr=self.sr)
        
        # 长度检查
        if len(speech) < self.sr * duration_setting["min"]:
            return None, None, "too_short"
        if len(speech) > self.sr * duration_setting["max"]:
            # 截断过长音频
            speech = speech[:int(self.sr * duration_setting["max"])]
        
        wave = torch.from_numpy(speech).float().unsqueeze(0)
        mel = to_mel_fn(wave, self.mel_fn_args, wav_path).squeeze(0)
        
        return wave.squeeze(0), mel, None
    
    def set_epoch(self, epoch):
        """设置当前 epoch 并生成随机索引"""
        self.current_epoch = epoch
        epoch_seed = 1234 + epoch
        rng = random.Random(epoch_seed)
        self.current_epoch_indices = list(range(len(self.paired_data)))
        rng.shuffle(self.current_epoch_indices)
    
    def __len__(self):
        return len(self.paired_data)
    
    def __getitem__(self, idx):
        # 获取配对的音频
        shuffled_idx = self.current_epoch_indices[idx % len(self.paired_data)]
        source_path, target_path = self.paired_data[shuffled_idx]
        
        # 加载 target 音频（目标 mel）
        target_wave, target_mel, error = self._load_and_process_audio(target_path)
        if error:
            return self.__getitem__((idx + 1) % len(self))
        
        # 加载 source 音频（内容）
        source_wave, source_mel, error = self._load_and_process_audio(source_path)
        if error:
            source_wave = target_wave
            source_mel = target_mel
            source_path = target_path
        
        # 加载 reference 音频（风格）
        if self.same_reference or not self.use_paired_mode:
            # 参考音频 = target 本身
            ref_wave = target_wave
            ref_path = target_path
        else:
            # 参考音频 = target 集合中的其他音频（模拟推理时的场景）
            ref_seed = 9999 + self.current_epoch * 1000 + idx
            ref_rng = random.Random(ref_seed)
            ref_idx = ref_rng.randint(0, len(self.target_list) - 1)
            ref_path = self.target_list[ref_idx]
            
            ref_wave, _, error = self._load_and_process_audio(ref_path)
            if error:
                ref_wave = target_wave
                ref_path = target_path
        
        # 返回: source_wave, target_wave, ref_wave, target_mel, source_path, target_path, ref_path
        return source_wave, target_wave, ref_wave, target_mel, source_path, target_path, ref_path


def paired_collate(batch):
    """配对数据集的 collate 函数"""
    batch_size = len(batch)
    
    # batch[i] = (source_wave, target_wave, ref_wave, target_mel, source_path, target_path, ref_path)
    
    nmels = batch[0][3].size(0)  # target_mel 的 mel 维度
    
    # 计算最大长度
    max_source_wave_len = max([b[0].size(0) for b in batch])
    max_target_wave_len = max([b[1].size(0) for b in batch])
    max_ref_wave_len = max([b[2].size(0) for b in batch])
    max_target_mel_len = max([b[3].shape[1] for b in batch])
    
    # 初始化张量
    source_waves = torch.zeros((batch_size, max_source_wave_len)).float()
    target_waves = torch.zeros((batch_size, max_target_wave_len)).float()
    ref_waves = torch.zeros((batch_size, max_ref_wave_len)).float()
    target_mels = torch.zeros((batch_size, nmels, max_target_mel_len)).float() - 10
    
    source_wave_lens = torch.zeros(batch_size).long()
    target_wave_lens = torch.zeros(batch_size).long()
    ref_wave_lens = torch.zeros(batch_size).long()
    target_mel_lens = torch.zeros(batch_size).long()
    
    source_paths = []
    target_paths = []
    ref_paths = []
    
    for bid, item in enumerate(batch):
        source_wave, target_wave, ref_wave, target_mel, source_path, target_path, ref_path = item
        
        # Source
        source_waves[bid, :source_wave.size(0)] = source_wave
        source_wave_lens[bid] = source_wave.size(0)
        source_paths.append(source_path)
        
        # Target
        target_waves[bid, :target_wave.size(0)] = target_wave
        target_wave_lens[bid] = target_wave.size(0)
        target_mel_size = target_mel.size(1)
        target_mels[bid, :, :target_mel_size] = target_mel
        target_mel_lens[bid] = target_mel_size
        target_paths.append(target_path)
        
        # Reference
        ref_waves[bid, :ref_wave.size(0)] = ref_wave
        ref_wave_lens[bid] = ref_wave.size(0)
        ref_paths.append(ref_path)
    
    return {
        'source_waves': source_waves,
        'source_wave_lens': source_wave_lens,
        'target_waves': target_waves,
        'target_wave_lens': target_wave_lens,
        'ref_waves': ref_waves,
        'ref_wave_lens': ref_wave_lens,
        'target_mels': target_mels,
        'target_mel_lens': target_mel_lens,
        'source_paths': source_paths,
        'target_paths': target_paths,
        'ref_paths': ref_paths,
    }


def build_paired_dataloader(target_path, spect_params, sr, batch_size=1, num_workers=0, 
                            shuffle=False, source_path=None, same_reference=True):
    """
    构建配对训练数据加载器
    
    Args:
        target_path: 目标说话人音频目录
        source_path: 源说话人音频目录（可选）
        same_reference: 参考音频是否与 target 相同
    """
    dataset = PairedFTDataset(
        target_path=target_path,
        spect_params=spect_params,
        sr=sr,
        batch_size=batch_size,
        source_path=source_path,
        same_reference=same_reference,
    )
    
    if shuffle:
        dataset.set_epoch(0)
    
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=paired_collate,
        generator=torch.Generator().manual_seed(1234) if shuffle else None,
    )
    
    return dataloader


if __name__ == "__main__":
    # 测试代码
    target_path = "./examples/reference"
    source_path = "./examples/source"
    sr = 22050
    spect_params = {
        "n_fft": 1024,
        "win_length": 1024,
        "hop_length": 256,
        "n_mels": 80,
        "fmin": 0,
        "fmax": 8000,
    }
    
    # 测试配对模式
    if os.path.exists(source_path):
        dataloader = build_paired_dataloader(
            target_path, spect_params, sr, 
            batch_size=2, num_workers=0, 
            source_path=source_path
        )
        print("Testing paired mode:")
        for idx, batch in enumerate(dataloader):
            print(f"Source waves: {batch['source_waves'].shape}")
            print(f"Target waves: {batch['target_waves'].shape}")
            if idx == 2:
                break
    
    # 测试自重建模式
    dataloader = build_paired_dataloader(
        target_path, spect_params, sr, 
        batch_size=2, num_workers=0
    )
    print("\nTesting self-reconstruction mode:")
    for idx, batch in enumerate(dataloader):
        print(f"Source waves: {batch['source_waves'].shape}")
        print(f"Target waves: {batch['target_waves'].shape}")
        if idx == 2:
            break
