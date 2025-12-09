import torch
import librosa
import numpy as np
import random
import os
import hashlib
import soundfile as sf
from torch.utils.data import DataLoader
from modules.audio import mel_spectrogram


duration_setting = {
    "min": 1.0,
    "max": 30.0,
}
# assume single speaker
def to_mel_fn(wave, mel_fn_args):
    return mel_spectrogram(wave, **mel_fn_args)

class FT_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        spect_params,
        sr=22050,
        batch_size=1,
    ):
        self.data_path = data_path
        print(f"Loading data from: {data_path}")
        print(f"Absolute path: {os.path.abspath(data_path)}")
        
        # Check if directory exists
        if not os.path.exists(data_path):
            raise ValueError(f"Data path does not exist: {data_path}")
            
        if not os.path.isdir(data_path):
            raise ValueError(f"Data path is not a directory: {data_path}")
        
        self.data = []
        file_count = 0
        
        # Debug: List directory contents first
        try:
            contents = os.listdir(data_path)
            # Removed debug print statement
        except Exception as e:
            print(f"Error listing directory: {e}")
        
        for root, _, files in os.walk(data_path):
            # Removed debug print statement
            for file in files:
                file_count += 1
                if file.endswith((".wav", ".mp3", ".flac", ".ogg", ".m4a", ".opus")):
                    full_path = os.path.join(root, file)
                    # Removed print statement
                    self.data.append(full_path)
        
        # Removed debug print statements
        
        # Additional check with glob
        if len(self.data) == 0:
            import glob
            # Removed debug print statement
            for ext in ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a", "*.opus"]:
                pattern = os.path.join(data_path, "**", ext)
                try:
                    matches = glob.glob(pattern, recursive=True)
                    # Removed debug print statement
                    self.data.extend(matches)
                except Exception as e:
                    print(f"Error with glob pattern {pattern}: {e}")
        
        # Sort by filename for consistency
        self.data.sort()
        
        if len(self.data) == 0:
            raise AssertionError(f"No valid audio files found in {data_path}. "
                                 f"Checked directory and subdirectories for files with extensions: "
                                 f".wav, .mp3, .flac, .ogg, .m4a, .opus. "
                                 f"Total files checked: {file_count}")

        # 添加用于存储当前epoch打乱顺序的索引列表
        self.current_epoch_indices = list(range(len(self.data)))
        self.current_epoch = 0

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

        assert len(self.data) != 0
        while len(self.data) < batch_size:
            self.data += self.data

    def __len__(self):
        return len(self.data)

    def _safe_load_audio(self, wav_path):
        """
        安全地加载音频文件，防止递归错误
        """
        # 首先尝试使用soundfile直接加载，避免librosa的复杂处理
        try:
            speech, orig_sr = sf.read(wav_path)
            # 转换为浮点型
            if speech.dtype != np.float32:
                speech = speech.astype(np.float32)
            # 如果是立体声，转换为单声道
            if speech.ndim > 1:
                speech = np.mean(speech, axis=1)
            return speech, orig_sr
        except Exception as e:
            print(f"Soundfile failed to load {wav_path}: {e}")
        
        # 如果soundfile失败，再尝试使用librosa加载
        try:
            # 设置较小的块大小以减少内存使用
            speech, orig_sr = librosa.load(wav_path, sr=None, mono=True)
            return speech, orig_sr
        except Exception as e:
            print(f"Librosa failed to load {wav_path}: {e}")
            raise e

    def set_epoch(self, epoch):
        """设置当前epoch并生成该epoch的随机索引序列"""
        self.current_epoch = epoch
        # 为每个epoch生成确定性的随机种子
        epoch_seed = 1234 + epoch
        rng = random.Random(epoch_seed)
        # 创建该epoch的索引副本并打乱
        self.current_epoch_indices = list(range(len(self.data)))
        rng.shuffle(self.current_epoch_indices)

    def __getitem__(self, idx):
        # 使用当前epoch的打乱索引
        shuffled_idx = self.current_epoch_indices[idx % len(self.data)]
        idx = shuffled_idx
        wav_path = self.data[idx]
        try:
            # 使用更安全的音频加载方法
            speech, orig_sr = self._safe_load_audio(wav_path)
        except RecursionError as e:
            print(f"RecursionError when loading {wav_path}: {e}")
            print(f"Skipping file {wav_path} due to recursion error")
            # 使用确定性选择而不是随机选择
            import hashlib
            hash_input = f"{idx}_{len(self)}_{str(e)}".encode()
            deterministic_idx = int(hashlib.md5(hash_input).hexdigest(), 16) % len(self)
            # 避免无限递归，如果连续多次出现递归错误则跳过
            if deterministic_idx == idx:
                # 如果计算出的索引与当前索引相同，尝试下一个索引
                deterministic_idx = (idx + 1) % len(self)
            return self.__getitem__(deterministic_idx)
        except Exception as e:
            print(f"Failed to load wav file with error {e}")
            # 使用确定性选择而不是随机选择
            import hashlib
            hash_input = f"{idx}_{len(self)}_{str(e)}".encode()
            deterministic_idx = int(hashlib.md5(hash_input).hexdigest(), 16) % len(self)
            # 避免无限递归
            if deterministic_idx == idx:
                # 如果计算出的索引与当前索引相同，尝试下一个索引
                deterministic_idx = (idx + 1) % len(self)
            return self.__getitem__(deterministic_idx)
        
        # 重采样到目标采样率
        if orig_sr != self.sr:
            try:
                # 使用更安全的重采样方法
                speech = librosa.resample(speech, orig_sr=orig_sr, target_sr=self.sr)
            except RecursionError as e:
                print(f"RecursionError when resampling {wav_path}: {e}")
                print(f"Skipping file {wav_path} due to recursion error in resampling")
                # 使用确定性选择而不是随机选择
                import hashlib
                hash_input = f"{idx}_{len(self)}_resample_recursion_{str(e)}".encode()
                deterministic_idx = int(hashlib.md5(hash_input).hexdigest(), 16) % len(self)
                # 避免无限递归
                if deterministic_idx == idx:
                    # 如果计算出的索引与当前索引相同，尝试下一个索引
                    deterministic_idx = (idx + 1) % len(self)
                return self.__getitem__(deterministic_idx)
            except Exception as e:
                print(f"Failed to resample {wav_path}: {e}")
                # 使用确定性选择而不是随机选择
                import hashlib
                hash_input = f"{idx}_{len(self)}_resample_{str(e)}".encode()
                deterministic_idx = int(hashlib.md5(hash_input).hexdigest(), 16) % len(self)
                # 避免无限递归
                if deterministic_idx == idx:
                    # 如果计算出的索引与当前索引相同，尝试下一个索引
                    deterministic_idx = (idx + 1) % len(self)
                return self.__getitem__(deterministic_idx)

        # 检查音频长度
        if len(speech) < self.sr * duration_setting["min"] or len(speech) > self.sr * duration_setting["max"]:
            print(f"Audio {wav_path} is too short or too long, skipping")
            # 使用确定性选择而不是随机选择
            import hashlib
            hash_input = f"{idx}_{len(self)}_{wav_path}".encode()
            deterministic_idx = int(hashlib.md5(hash_input).hexdigest(), 16) % len(self)
            # 避免无限递归
            if deterministic_idx == idx:
                # 如果计算出的索引与当前索引相同，尝试下一个索引
                deterministic_idx = (idx + 1) % len(self)
            return self.__getitem__(deterministic_idx)

        wave = torch.from_numpy(speech).float().unsqueeze(0)
        mel = to_mel_fn(wave, self.mel_fn_args).squeeze(0)

        return wave.squeeze(0), mel


def build_ft_dataloader(data_path, spect_params, sr, batch_size=1, num_workers=0, shuffle=False):
    dataset = FT_Dataset(data_path, spect_params, sr, batch_size)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        generator=torch.Generator().manual_seed(1234) if shuffle else None,
    )
    return dataloader

def collate(batch):
    batch_size = len(batch)

    # Keep original order instead of sorting by mel length for reproducibility
    # This ensures that the same batches are processed in the same order across runs
    batch_indexes = list(range(batch_size))
    batch = [batch[bid] for bid in batch_indexes]

    nmels = batch[0][1].size(0)
    max_mel_length = max([b[1].shape[1] for b in batch])
    max_wave_length = max([b[0].size(0) for b in batch])

    mels = torch.zeros((batch_size, nmels, max_mel_length)).float() - 10
    waves = torch.zeros((batch_size, max_wave_length)).float()

    mel_lengths = torch.zeros(batch_size).long()
    wave_lengths = torch.zeros(batch_size).long()

    for bid, (wave, mel) in enumerate(batch):
        mel_size = mel.size(1)
        mels[bid, :, :mel_size] = mel
        waves[bid, : wave.size(0)] = wave
        mel_lengths[bid] = mel_size
        wave_lengths[bid] = wave.size(0)

    return waves, mels, wave_lengths, mel_lengths

if __name__ == "__main__":
    data_path = "./example/reference"
    sr = 22050
    spect_params = {
        "n_fft": 1024,
        "win_length": 1024,
        "hop_length": 256,
        "n_mels": 80,
        "fmin": 0,
        "fmax": 8000,
    }
    dataloader = build_ft_dataloader(data_path, spect_params, sr, batch_size=2, num_workers=0)
    for idx, batch in enumerate(dataloader):
        wave, mel, wave_lengths, mel_lengths = batch
        print(wave.shape, mel.shape)
        if idx == 10:
            break