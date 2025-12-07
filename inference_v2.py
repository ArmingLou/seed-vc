import os
import argparse
import torch
import yaml
import soundfile as sf
import time
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
    from hydra.utils import instantiate
    from omegaconf import DictConfig
    # 使用传入的配置文件路径，如果没有指定则使用默认路径
    config_path = args.config if hasattr(args, 'config') and args.config else "configs/v2/vc_wrapper.yaml"
    cfg = DictConfig(yaml.safe_load(open(config_path, "r")))
    vc_wrapper = instantiate(cfg)
    
    # 根据fp16参数决定加载模型时的数据类型
    model_dtype = torch.float16 if fp16 else torch.float32
    print(f"正在尝试使用{model_dtype}精度加载V2模型到{device}设备...")
    
    try:
        vc_wrapper.load_checkpoints(ar_checkpoint_path=args.ar_checkpoint_path,
                                    cfm_checkpoint_path=args.cfm_checkpoint_path,
                                    dtype=model_dtype)
    except Exception as e:
        print(f"警告: 在{device}设备上无法使用{model_dtype}精度加载V2模型: {e}")
        print(f"正在回退到float32精度加载V2模型...")
        global fp16, dtype
        fp16 = False
        dtype = torch.float32
        model_dtype = torch.float32
        vc_wrapper.load_checkpoints(ar_checkpoint_path=args.ar_checkpoint_path,
                                    cfm_checkpoint_path=args.cfm_checkpoint_path,
                                    dtype=model_dtype)
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
    model_dtype = torch.float32
    vc_wrapper.load_checkpoints(ar_checkpoint_path=args.ar_checkpoint_path,
                                cfm_checkpoint_path=args.cfm_checkpoint_path,
                                dtype=model_dtype)
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
    sf.write(output_path, converted_audio, save_sr)

    print(f"Voice conversion completed in {end_time - start_time:.2f} seconds")
    print(f"Output saved to: {output_path}")


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