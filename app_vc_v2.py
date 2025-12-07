import gradio as gr
import torch
import yaml
import os
from modules.commons import str2bool

# 初始化全局变量
fp16 = False
device = None
dtype = torch.float32
vc_wrapper = None

def load_models(args):
    global fp16, dtype
    from hydra.utils import instantiate
    from omegaconf import DictConfig
    
    # 更新全局fp16标志
    fp16 = args.fp16
    print(f"Using device: {device}")
    print(f"Using fp16: {fp16}")
    
    # 根据fp16参数决定数据类型
    dtype = torch.float16 if fp16 else torch.float32
    
    # 自动切换fp16逻辑
    # 仅在需要时打印警告信息
    if (device.type == "cpu" or device.type == "mps") and fp16:
        print(f"Warning: fp16 is enabled for {device.type} device, which may cause issues")
    
    # 使用传入的配置文件路径，如果没有指定则使用默认路径
    config_path = args.config if hasattr(args, 'config') and args.config else "configs/v2/vc_wrapper.yaml"
    cfg = DictConfig(yaml.safe_load(open(config_path, "r")))
    vc_wrapper = instantiate(cfg)
    
    # 根据fp16参数决定加载模型时的数据类型
    model_dtype = torch.float16 if fp16 else torch.float32
    print(f"正在尝试使用{model_dtype}精度加载V2模型到{device}设备...")
    
    try:
        vc_wrapper.load_checkpoints(ar_checkpoint_path=args.ar_checkpoint_path,
                                   cfm_checkpoint_path=args.cfm_checkpoint_path)
    except Exception as e:
        print(f"警告: 在{device}设备上无法使用{model_dtype}精度加载V2模型: {e}")
        print(f"正在回退到float32精度加载V2模型...")
        fp16 = False
        dtype = torch.float32
        model_dtype = torch.float32
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
    model_dtype = torch.float32
    vc_wrapper.load_checkpoints(ar_checkpoint_path=args.ar_checkpoint_path,
                               cfm_checkpoint_path=args.cfm_checkpoint_path)
    vc_wrapper.to(device)
    vc_wrapper.eval()
    vc_wrapper.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=dtype, device=device)
    
    print(f"信息: 已成功回退到float32精度并重新加载模型")
    return vc_wrapper

def main(args):
    global vc_wrapper
    vc_wrapper = load_models(args)
    
    # 创建一个包装函数，传递正确的设备和数据类型参数，并处理生成器返回值
    def convert_voice_with_streaming_wrapper(source_audio_path, target_audio_path, diffusion_steps=30,
                                           length_adjust=1.0, intelligebility_cfg_rate=0.7, similarity_cfg_rate=0.7,
                                           top_p=0.7, temperature=0.7, repetition_penalty=1.5,
                                           convert_style=False, anonymization_only=False):
        global vc_wrapper
        try:
            # 调用生成器函数并收集所有输出
            generator = vc_wrapper.convert_voice_with_streaming(
                source_audio_path, target_audio_path, diffusion_steps,
                length_adjust, intelligebility_cfg_rate, similarity_cfg_rate,
                top_p, temperature, repetition_penalty,
                convert_style, anonymization_only,
                device=device, dtype=dtype, stream_output=True
            )
            
            # 收集生成器的所有输出，返回最后一个完整的音频
            mp3_bytes = None
            full_audio = None
            for mp3_bytes, full_audio in generator:
                pass  # 我们只关心最后一个值
            
            # 返回两个值：流式输出和完整输出
            return mp3_bytes, full_audio
        except RuntimeError as e:
            if "LayerNormKernelImpl" in str(e) and device.type == "cpu" and fp16:
                print(f"警告: 在CPU设备上使用fp16时遇到LayerNorm错误，正在回退到float32精度...")
                # 回退到float32精度重新加载模型
                vc_wrapper = reload_v2_model(vc_wrapper, args)
                # 重新执行推理
                generator = vc_wrapper.convert_voice_with_streaming(
                    source_audio_path, target_audio_path, diffusion_steps,
                    length_adjust, intelligebility_cfg_rate, similarity_cfg_rate,
                    top_p, temperature, repetition_penalty,
                    convert_style, anonymization_only,
                    device=device, dtype=dtype, stream_output=True
                )
                
                # 收集生成器的所有输出，返回最后一个完整的音频
                mp3_bytes = None
                full_audio = None
                for mp3_bytes, full_audio in generator:
                    pass  # 我们只关心最后一个值
                
                print(f"信息: 已成功回退到float32精度并重新执行推理")
                # 返回两个值：流式输出和完整输出
                return mp3_bytes, full_audio
            else:
                # 如果不是预期的LayerNorm错误，则重新抛出异常
                raise e
    
    # Set up Gradio interface
    description = ("Zero-shot voice conversion with in-context learning. For local deployment please check [GitHub repository](https://github.com/Plachtaa/seed-vc) "
                   "for details and updates.<br>Note that any reference audio will be forcefully clipped to 25s if beyond this length.<br> "
                   "If total duration of source and reference audio exceeds 30s, source audio will be processed in chunks.<br> "
                   "无需训练的 zero-shot 语音/歌声转换模型，若需本地部署查看[GitHub页面](https://github.com/Plachtaa/seed-vc)<br>"
                   "请注意，参考音频若超过 25 秒，则会被自动裁剪至此长度。<br>若源音频和参考音频的总时长超过 30 秒，源音频将被分段处理。")
    
    inputs = [
        gr.Audio(type="filepath", label="Source Audio / 源音频"),
        gr.Audio(type="filepath", label="Reference Audio / 参考音频"),
        gr.Slider(minimum=1, maximum=200, value=30, step=1, label="Diffusion Steps / 扩散步数", 
                 info="30 by default, 50~100 for best quality / 默认为 30，50~100 为最佳质量"),
        gr.Slider(minimum=0.5, maximum=2.0, step=0.1, value=1.0, label="Length Adjust / 长度调整", 
                 info="<1.0 for speed-up speech, >1.0 for slow-down speech / <1.0 加速语速，>1.0 减慢语速"),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.5, label="Intelligibility CFG Rate",
                 info="has subtle influence / 有微小影响"),
        gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.5, label="Similarity CFG Rate",
                  info="has subtle influence / 有微小影响"),
        gr.Slider(minimum=0.1, maximum=1.0, step=0.1, value=0.9, label="Top-p",
                 info="Controls diversity of generated audio / 控制生成音频的多样性"),
        gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Temperature",
                 info="Controls randomness of generated audio / 控制生成音频的随机性"),
        gr.Slider(minimum=1.0, maximum=3.0, step=0.1, value=1.0, label="Repetition Penalty",
                 info="Penalizes repetition in generated audio / 惩罚生成音频中的重复"),
        gr.Checkbox(label="convert style", value=False),
        gr.Checkbox(label="anonymization only", value=False),
    ]
    
    examples = [
        ["examples/source/yae_0.wav", "examples/reference/dingzhen_0.wav", 50, 1.0, 0.5, 0.5, 0.9, 1.0, 1.0, False, False],
        ["examples/source/jay_0.wav", "examples/reference/azuma_0.wav", 50, 1.0, 0.5, 0.5, 0.9, 1.0, 1.0, False, False],
    ]
    
    outputs = [
        gr.Audio(label="Stream Output Audio / 流式输出", streaming=True, format='mp3'),
        gr.Audio(label="Full Output Audio / 完整输出", streaming=False, format='wav')
    ]
    
    # Launch the Gradio interface
    gr.Interface(
        fn=convert_voice_with_streaming_wrapper,  # 使用包装函数
        description=description,
        inputs=inputs,
        outputs=outputs,
        title="Seed Voice Conversion V2",
        examples=examples,
        cache_examples=False,
    ).launch()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="Compile the model using torch.compile")
    parser.add_argument("--config", type=str, default="configs/v2/vc_wrapper.yaml",
                        help="Path to the configuration file")
    parser.add_argument("--fp16", type=str2bool, nargs="?", const=True, help="Whether to use fp16", default=False)
    # V2 custom checkpoints
    parser.add_argument("--ar-checkpoint-path", type=str, default=None,
                        help="Path to custom checkpoint file")
    parser.add_argument("--cfm-checkpoint-path", type=str, default=None,
                        help="Path to custom checkpoint file")
    args = parser.parse_args()
    
    # 根据环境变量决定是否强制使用 CPU
    if os.environ.get("FORCE_CPU", "0") == "1":
        device = torch.device("cpu")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    main(args)