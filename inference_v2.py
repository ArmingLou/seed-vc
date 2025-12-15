import os
import argparse
import torch
import yaml
import soundfile as sf
import time
import numpy as np
from modules.commons import str2bool

# Set up device and torch configurations
# 根据环境变量决定是否强制使用 CPU
if os.environ.get("FORCE_CPU", "0") == "1":
    device = torch.device("cpu")
elif hasattr(torch, 'xpu') and torch.xpu.is_available():
    device = torch.device("xpu")
elif torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# 初始化全局变量
fp16 = False
dtype = torch.float32

# Global variables to store model instances
vc_wrapper_v2 = None


def load_v2_models(args):
    """Load V2 models using the wrapper from app.py"""
    global fp16, dtype
    from hydra.utils import instantiate
    from omegaconf import DictConfig
    # 使用传入的配置文件路径，如果没有指定则使用默认路径
    config_path = args.config if hasattr(args, 'config') and args.config else "configs/v2/vc_wrapper.yaml"
    cfg = DictConfig(yaml.safe_load(open(config_path, "r")))
    vc_wrapper = instantiate(cfg)
    
    print(f"正在尝试加载V2模型到{device}设备...")
    
    try:
        vc_wrapper.load_checkpoints(ar_checkpoint_path=args.ar_checkpoint_path,
                                    cfm_checkpoint_path=args.cfm_checkpoint_path)
    except Exception as e:
        print(f"警告: 在{device}设备上无法加载V2模型: {e}")
        print(f"正在回退到float32精度加载V2模型...")
        fp16 = False
        dtype = torch.float32
        vc_wrapper.load_checkpoints(ar_checkpoint_path=args.ar_checkpoint_path,
                                    cfm_checkpoint_path=args.cfm_checkpoint_path)
        print(f"信息: 已将内部fp16标志设置为False，以保持一致性")
        print(f"V2模型已加载到{device}设备，使用float32数据类型")
    
    vc_wrapper.to(device)
    vc_wrapper.eval()

    vc_wrapper.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=dtype, device=device)

    if args.compile:
        torch._inductor.config.coordinate_descent_tuning = True
        torch._inductor.config.triton.unique_kernel_names = True

        if hasattr(torch._inductor.config, "fx_graph_cache"):
            # Experimental feature to reduce compilation times, will be on by default in future
            torch._inductor.config.fx_graph_cache = True
        vc_wrapper.compile_ar()
        # vc_wrapper.compile_cfm()

    return vc_wrapper

# 重新加载模型为float32精度的函数
def reload_v2_model(vc_wrapper, args):
    """重新加载V2模型为float32精度"""
    global fp16, dtype
    fp16 = False
    dtype = torch.float32
    print(f"信息: 已将内部fp16标志设置为False，以保持一致性")
    
    # 重新加载模型为float32精度
    vc_wrapper.load_checkpoints(ar_checkpoint_path=args.ar_checkpoint_path,
                                cfm_checkpoint_path=args.cfm_checkpoint_path)
    vc_wrapper.to(device)
    vc_wrapper.eval()
    vc_wrapper.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=dtype, device=device)
    
    print(f"信息: 已成功回退到float32精度并重新加载模型")
    return vc_wrapper


def convert_voice_v2(source_audio_path, target_audio_path, args):
    """Convert voice using V2 model"""
    global vc_wrapper_v2, fp16, dtype
    # 更新全局fp16标志和dtype
    fp16 = args.fp16
    # 根据fp16参数决定数据类型，但不与设备关联
    dtype = torch.float16 if fp16 else torch.float32
    
    if vc_wrapper_v2 is None:
        vc_wrapper_v2 = load_v2_models(args)

    # Use the generator function but collect all outputs
    try:
        generator = vc_wrapper_v2.convert_voice_with_streaming(
            source_audio_path=source_audio_path,
            target_audio_path=target_audio_path,
            diffusion_steps=args.diffusion_steps,
            length_adjust=args.length_adjust,
            intelligebility_cfg_rate=args.intelligibility_cfg_rate,
            similarity_cfg_rate=args.similarity_cfg_rate,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            convert_style=args.convert_style,
            anonymization_only=args.anonymization_only,
            device=device,
            dtype=dtype,
            stream_output=True
        )

        # Collect all outputs from the generator
        for output in generator:
            _, full_audio = output
        return full_audio
    except RuntimeError as e:
        if "LayerNormKernelImpl" in str(e) and device.type == "cpu" and fp16:
            print(f"警告: 在CPU设备上使用fp16时遇到LayerNorm错误，正在回退到float32精度...")
            # 回退到float32精度重新加载模型
            vc_wrapper_v2 = reload_v2_model(vc_wrapper_v2, args)
            # 重新执行推理
            generator = vc_wrapper_v2.convert_voice_with_streaming(
                source_audio_path=source_audio_path,
                target_audio_path=target_audio_path,
                diffusion_steps=args.diffusion_steps,
                length_adjust=args.length_adjust,
                intelligebility_cfg_rate=args.intelligibility_cfg_rate,
                similarity_cfg_rate=args.similarity_cfg_rate,
                top_p=args.top_p,
                temperature=args.temperature,
                repetition_penalty=args.repetition_penalty,
                convert_style=args.convert_style,
                anonymization_only=args.anonymization_only,
                device=device,
                dtype=dtype,
                stream_output=True
            )

            # Collect all outputs from the generator
            for output in generator:
                _, full_audio = output
            print(f"信息: 已成功回退到float32精度并重新执行推理")
            return full_audio
        else:
            # 如果不是预期的LayerNorm错误，则重新抛出异常
            raise e


def main(args):
    # 更新全局fp16标志和dtype
    global fp16, dtype
    fp16 = args.fp16
    # 根据fp16参数决定数据类型，但不与设备关联
    dtype = torch.float16 if fp16 else torch.float32
    
    # 自动切换fp16逻辑
    # 仅在需要时打印警告信息
    if (device.type == "cpu" or device.type == "mps") and fp16:
        print(f"Warning: fp16 is enabled for {device.type} device, which may cause issues")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)

    start_time = time.time()
    converted_audio = convert_voice_v2(args.source, args.target, args)
    end_time = time.time()

    if converted_audio is None:
        print("Error: Failed to convert voice")
        return

    # Save the converted audio
    source_name = os.path.basename(args.source).split(".")[0]
    target_name = os.path.basename(args.target).split(".")[0]

    # Create a descriptive filename
    filename = f"vc_v2_{source_name}_{target_name}_{args.length_adjust}_{args.diffusion_steps}_{args.similarity_cfg_rate}.wav"

    output_path = os.path.join(args.output, filename)
    save_sr, converted_audio = converted_audio
    
    # 确保音频数据是numpy数组并且是正确的数据类型
    if isinstance(converted_audio, torch.Tensor):
        converted_audio = converted_audio.cpu().numpy()
    
    # print(f"原始音频数据形状: {converted_audio.shape}")
    # print(f"原始音频数据类型: {converted_audio.dtype}")
    # print(f"原始音频数据范围: [{converted_audio.min():.6f}, {converted_audio.max():.6f}]")
    
    # 确保音频数据在一维或二维（立体声）范围内
    if converted_audio.ndim == 1:
        # 单声道音频
        audio_data = converted_audio
        # print("处理为单声道音频")
    elif converted_audio.ndim == 2:
        # 立体声音频，需要转置
        audio_data = converted_audio.T
        # print("处理为立体声音频")
    else:
        print(f"警告: 音频数据维度异常 ({converted_audio.ndim}D)，尝试将其转换为一维")
        audio_data = converted_audio.flatten()
    
    # 确保数据类型为float32并且在[-1, 1]范围内
    if audio_data.dtype != np.float32:
        audio_data = audio_data.astype(np.float32)
    
    # 检查是否有NaN或Inf值
    if np.isnan(audio_data).any():
        print("警告: 音频数据包含NaN值，替换为0")
        audio_data = np.nan_to_num(audio_data)
    
    if np.isinf(audio_data).any():
        print("警告: 音频数据包含Inf值，替换为有限值")
        audio_data = np.nan_to_num(audio_data)
    
    # 归一化到[-1, 1]范围
    if np.abs(audio_data).max() > 1.0:
        audio_data = audio_data / np.abs(audio_data).max()
        print("警告: 音频数据已被归一化到[-1, 1]范围")
    
    # print(f"处理后音频数据形状: {audio_data.shape}")
    # print(f"处理后音频数据类型: {audio_data.dtype}")
    # print(f"处理后音频数据范围: [{audio_data.min():.6f}, {audio_data.max():.6f}]")
    
    try:
        print(f"尝试使用float32格式保存音频文件: {output_path}")
        sf.write(output_path, audio_data, save_sr)
        print(f"Voice conversion completed in {end_time - start_time:.2f} seconds")
        print(f"Output saved to: {output_path}")
    except Exception as e:
        print(f"使用float32格式保存音频文件时出错: {e}")
        # 尝试使用不同的数据类型保存
        try:
            print(f"尝试使用int16格式保存音频文件: {output_path}")
            audio_data_int16 = (audio_data * 32767).astype(np.int16)
            print(f"int16数据范围: [{audio_data_int16.min()}, {audio_data_int16.max()}]")
            sf.write(output_path, audio_data_int16, save_sr, subtype='PCM_16')
            print(f"Voice conversion completed in {end_time - start_time:.2f} seconds")
            print(f"Output saved to: {output_path}")
        except Exception as e2:
            print(f"使用int16格式保存音频文件时也出错: {e2}")
            # 最后的备用方案：尝试创建目录并重新保存
            try:
                output_dir = os.path.dirname(output_path)
                if output_dir and not os.path.exists(output_dir):
                    os.makedirs(output_dir, exist_ok=True)
                    print(f"创建输出目录: {output_dir}")
                
                # 尝试使用不同的subtype
                print("尝试使用不同的音频子类型保存")
                sf.write(output_path, audio_data, save_sr, subtype='FLOAT')
                print(f"Voice conversion completed in {end_time - start_time:.2f} seconds")
                print(f"Output saved to: {output_path}")
            except Exception as e3:
                print(f"所有保存方法都失败了: {e3}")
                # 如果所有方法都失败，保存为.npy文件用于调试
                try:
                    npy_path = output_path.replace('.wav', '.npy')
                    np.save(npy_path, audio_data)
                    print(f"作为备用方案，音频数据已保存为.npy文件: {npy_path}")
                    print("您可以使用 np.load() 加载此文件进行进一步分析")
                except Exception as e4:
                    print(f"连.npy文件保存也失败了: {e4}")
                raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Voice Conversion Inference Script")
    parser.add_argument("--source", type=str, required=True,
                        help="Path to source audio file")
    parser.add_argument("--target", type=str, required=True,
                        help="Path to target/reference audio file")
    parser.add_argument("--output", type=str, default="./output",
                        help="Output directory for converted audio")
    parser.add_argument("--diffusion-steps", type=int, default=30,
                        help="Number of diffusion steps")
    parser.add_argument("--length-adjust", type=float, default=1.0,
                        help="Length adjustment factor (<1.0 for speed-up, >1.0 for slow-down)")
    parser.add_argument("--compile", type=bool, default=False,
                        help="Whether to compile the model for faster inference")
    parser.add_argument("--config", type=str, default="configs/v2/vc_wrapper.yaml",
                        help="Path to the configuration file")

    # V2 specific arguments
    parser.add_argument("--intelligibility-cfg-rate", type=float, default=0.7,
                        help="Intelligibility CFG rate for V2 model")
    parser.add_argument("--similarity-cfg-rate", type=float, default=0.7,
                        help="Similarity CFG rate for V2 model")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p sampling parameter for V2 model")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature sampling parameter for V2 model")
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="Repetition penalty for V2 model")
    parser.add_argument("--convert-style", type=str2bool, default=False,
                        help="Convert style/emotion/accent for V2 model")
    parser.add_argument("--anonymization-only", type=str2bool, default=False,
                        help="Anonymization only mode for V2 model")

    # V2 custom checkpoints
    parser.add_argument("--ar-checkpoint-path", type=str, default=None,
                        help="Path to custom checkpoint file")
    parser.add_argument("--cfm-checkpoint-path", type=str, default=None,
                        help="Path to custom checkpoint file")
    parser.add_argument("--fp16", type=str2bool, default=False,
                        help="Use fp16 precision for inference")

    args = parser.parse_args()
    main(args)