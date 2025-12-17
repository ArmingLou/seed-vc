import gradio as gr
import torch
import yaml
import os
import glob
from modules.commons import str2bool
from hf_utils import load_custom_model_from_hf

# 初始化全局变量
fp16 = False
device = None
dtype = torch.float32
vc_wrapper = None

# 全局变量存储配置和检查点文件信息
config_file_names = []
config_file_dict = {}
ar_checkpoint_file_names = []
ar_checkpoint_file_dict = {}
cfm_checkpoint_file_names = []
cfm_checkpoint_file_dict = {}

def load_models(args, config_path=None, ar_checkpoint_path=None, cfm_checkpoint_path=None):
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
    
    # 使用传入的配置文件路径，如果没有指定或设置为"None"则使用默认路径
    if config_path is not None and config_path != "None":
        cfg = DictConfig(yaml.safe_load(open(config_path, "r")))
        print(f"Loading config from {config_path}")
    else:
        # Use default local config
        default_config_path = args.config if hasattr(args, 'config') and args.config else "configs/v2/vc_wrapper.yaml"
        cfg = DictConfig(yaml.safe_load(open(default_config_path, "r")))
        print(f"Loading config from {default_config_path}")
    vc_wrapper = instantiate(cfg)
    
    # 根据fp16参数决定加载模型时的数据类型
    model_dtype = torch.float16 if fp16 else torch.float32
    print(f"正在尝试使用{model_dtype}精度加载V2模型到{device}设备...")
    
    # 使用传入的检查点路径，如果没有指定或设置为"None"则从HF加载默认路径
    if ar_checkpoint_path is None or ar_checkpoint_path == "None":
        if args.ar_checkpoint_path is not None and args.ar_checkpoint_path != "":
            ar_checkpoint_path = args.ar_checkpoint_path
        else:
            # Load default AR checkpoint from HF for V2
            ar_checkpoint_path = load_custom_model_from_hf("Plachta/Seed-VC",
                                                   "v2/ar_base.pth",
                                                   None)
    if cfm_checkpoint_path is None or cfm_checkpoint_path == "None":
        if args.cfm_checkpoint_path is not None and args.cfm_checkpoint_path != "":
            cfm_checkpoint_path = args.cfm_checkpoint_path
        else:
            # Load default CFM checkpoint from HF for V2
            cfm_checkpoint_path = load_custom_model_from_hf("Plachta/Seed-VC",
                                                     "v2/cfm_small.pth",
                                                     None)
    
    try:
        vc_wrapper.load_checkpoints(ar_checkpoint_path=ar_checkpoint_path,
                                   cfm_checkpoint_path=cfm_checkpoint_path)
    except Exception as e:
        print(f"警告: 在{device}设备上无法使用{model_dtype}精度加载V2模型: {e}")
        print(f"正在回退到float32精度加载V2模型...")
        fp16 = False
        dtype = torch.float32
        model_dtype = torch.float32
        vc_wrapper.load_checkpoints(ar_checkpoint_path=ar_checkpoint_path,
                                   cfm_checkpoint_path=cfm_checkpoint_path)
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
def reload_v2_model(vc_wrapper, args, config_path=None, ar_checkpoint_path=None, cfm_checkpoint_path=None):
    """重新加载V2模型为float32精度"""
    global fp16, dtype
    fp16 = False
    dtype = torch.float32
    print(f"信息: 已将内部fp16标志设置为False，以保持一致性")
    
    # 重新加载模型为float32精度
    model_dtype = torch.float32
    # 使用传入的检查点路径，如果没有指定或设置为"None"则从HF加载默认路径
    if ar_checkpoint_path is None or ar_checkpoint_path == "None":
        if args.ar_checkpoint_path is not None and args.ar_checkpoint_path != "":
            ar_checkpoint_path = args.ar_checkpoint_path
        else:
            # Load default AR checkpoint from HF for V2
            ar_checkpoint_path = load_custom_model_from_hf("Plachta/Seed-VC",
                                                   "v2/ar_base.pth",
                                                   None)
    if cfm_checkpoint_path is None or cfm_checkpoint_path == "None":
        if args.cfm_checkpoint_path is not None and args.cfm_checkpoint_path != "":
            cfm_checkpoint_path = args.cfm_checkpoint_path
        else:
            # Load default CFM checkpoint from HF for V2
            cfm_checkpoint_path = load_custom_model_from_hf("Plachta/Seed-VC",
                                                     "v2/cfm_small.pth",
                                                     None)
    vc_wrapper.load_checkpoints(ar_checkpoint_path=ar_checkpoint_path,
                               cfm_checkpoint_path=cfm_checkpoint_path)
    vc_wrapper.to(device)
    vc_wrapper.eval()
    vc_wrapper.setup_ar_caches(max_batch_size=1, max_seq_len=4096, dtype=dtype, device=device)
    
    print(f"信息: 已成功回退到float32精度并重新加载模型")
    return vc_wrapper

def main(args):
    global vc_wrapper, config_file_names, config_file_dict, ar_checkpoint_file_names, ar_checkpoint_file_dict, cfm_checkpoint_file_names, cfm_checkpoint_file_dict
    
    # Get list of config files and checkpoint files from conf_dir
    if args.conf_dir and os.path.exists(args.conf_dir):
        config_files = glob.glob(os.path.join(args.conf_dir, "*.yaml")) + glob.glob(os.path.join(args.conf_dir, "*.yml"))
        config_file_names = [os.path.basename(f) for f in config_files]
        config_file_dict = {name: path for name, path in zip(config_file_names, config_files)}
        
        ar_checkpoint_files = glob.glob(os.path.join(args.conf_dir, "*ar*.pth"))
        ar_checkpoint_file_names = [os.path.basename(f) for f in ar_checkpoint_files]
        ar_checkpoint_file_dict = {name: path for name, path in zip(ar_checkpoint_file_names, ar_checkpoint_files)}
        
        cfm_checkpoint_files = glob.glob(os.path.join(args.conf_dir, "*cfm*.pth"))
        cfm_checkpoint_file_names = [os.path.basename(f) for f in cfm_checkpoint_files]
        cfm_checkpoint_file_dict = {name: path for name, path in zip(cfm_checkpoint_file_names, cfm_checkpoint_files)}
    else:
        # Default to v2 configs directory
        config_dir = "configs/v2"
        config_files = glob.glob(os.path.join(config_dir, "*.yaml")) + glob.glob(os.path.join(config_dir, "*.yml"))
        config_file_names = [os.path.basename(f) for f in config_files]
        config_file_dict = {name: path for name, path in zip(config_file_names, config_files)}
        
        # No checkpoint files by default
        ar_checkpoint_file_names = []
        ar_checkpoint_file_dict = {}
        cfm_checkpoint_file_names = []
        cfm_checkpoint_file_dict = {}
    
    # Add "None" option to all lists
    config_file_names = ["None"] + config_file_names if config_file_names else ["None"]
    ar_checkpoint_file_names = ["None"] + ar_checkpoint_file_names if ar_checkpoint_file_names else ["None"]
    cfm_checkpoint_file_names = ["None"] + cfm_checkpoint_file_names if cfm_checkpoint_file_names else ["None"]
    
    vc_wrapper = load_models(args)
    
    # 创建一个包装函数，传递正确的设备和数据类型参数，并处理生成器返回值
    def convert_voice_with_streaming_wrapper(source_audio_path, target_audio_path, diffusion_steps=30,
                                           length_adjust=1.0, intelligebility_cfg_rate=0.7, similarity_cfg_rate=0.7,
                                           top_p=0.7, temperature=0.7, repetition_penalty=1.5,
                                           convert_style=False, anonymization_only=False):
        global vc_wrapper
        yield None, None, gr.update(interactive=False)
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
            yield mp3_bytes, full_audio, gr.update(interactive=True)
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
                yield mp3_bytes, full_audio, gr.update(interactive=True)
            else:
                # 如果不是预期的LayerNorm错误，则重新抛出异常
                yield None, None, gr.update(interactive=True)
                raise e
    
    # Reload model function
    def reload_model(config_name, ar_checkpoint_name, cfm_checkpoint_name):
        print(f"====Reloading model with config {config_name} and ar checkpoint {ar_checkpoint_name} cfm checkpoint {cfm_checkpoint_name}")
        global vc_wrapper, config_file_dict, ar_checkpoint_file_dict, cfm_checkpoint_file_dict
        config_path = config_file_dict.get(config_name) if config_name != "None" else None
        ar_checkpoint_path = ar_checkpoint_file_dict.get(ar_checkpoint_name) if ar_checkpoint_name != "None" else None
        cfm_checkpoint_path = cfm_checkpoint_file_dict.get(cfm_checkpoint_name) if cfm_checkpoint_name != "None" else None
        
        if config_path or ar_checkpoint_path or cfm_checkpoint_path:
            # Reload models with new config/checkpoint
            vc_wrapper = load_models(args, config_path=config_path, ar_checkpoint_path=ar_checkpoint_path, cfm_checkpoint_path=cfm_checkpoint_path)
            status_msg = f""
            status_msg += f" 配置文件:   {config_name if config_name != 'None' else 'Default'}     |    "
            status_msg += f" AR检查点文件:   {ar_checkpoint_name if ar_checkpoint_name != 'None' else 'Default'}     |    "
            status_msg += f" CFM检查点文件:   {cfm_checkpoint_name if cfm_checkpoint_name != 'None' else 'Default'}"
            return status_msg, config_name, ar_checkpoint_name, cfm_checkpoint_name
        else:
            # Reload with default settings
            vc_wrapper = load_models(args)
            return "模型已使用默认设置重新加载", config_name, ar_checkpoint_name, cfm_checkpoint_name
    
    # Set up Gradio interface
    description = ("Zero-shot voice conversion with in-context learning. For local deployment please check [GitHub repository](https://github.com/Plachtaa/seed-vc) "
                   "for details and updates.<br>Note that any reference audio will be forcefully clipped to 25s if beyond this length.<br> "
                   "If total duration of source and reference audio exceeds 30s, source audio will be processed in chunks.<br> "
                   "无需训练的 zero-shot 语音/歌声转换模型，若需本地部署查看[GitHub页面](https://github.com/Plachtaa/seed-vc)<br>"
                   "请注意，参考音频若超过 25 秒，则会被自动裁剪至此长度。<br>若源音频和参考音频的总时长超过 30 秒，源音频将被分段处理。")
    
    with gr.Blocks() as demo:
        gr.Markdown("# Seed Voice Conversion V2")
        gr.Markdown(description)
        
        with gr.Row():
            config_choice = gr.Dropdown(choices=config_file_names, value="None", label="选择配置文件 / Select Config File")
            ar_checkpoint_choice = gr.Dropdown(choices=ar_checkpoint_file_names, value="None", label="选择AR检查点文件 / Select AR Checkpoint File")
            cfm_checkpoint_choice = gr.Dropdown(choices=cfm_checkpoint_file_names, value="None", label="选择CFM检查点文件 / Select CFM Checkpoint File")
            reload_btn = gr.Button("重新加载模型 / Reload Model")
        
        with gr.Row():
            status_bar = gr.Text(show_label=False)
        with gr.Row():
            source_audio = gr.Audio(type="filepath", label="Source Audio / 源音频")
            reference_audio = gr.Audio(type="filepath", label="Reference Audio / 参考音频")
        with gr.Row():
            convert_btn = gr.Button("开始转换 / Convert")
        with gr.Row():
            stream_output = gr.Audio(label="Stream Output Audio / 流式输出", streaming=True, format='mp3')
            full_output = gr.Audio(label="Full Output Audio / 完整输出", streaming=False, format='wav')
        with gr.Row():
            with gr.Column():
                diffusion_steps = gr.Slider(minimum=1, maximum=200, value=30, step=1, label="Diffusion Steps / 扩散步数", 
                         info="30 by default, 50~100 for best quality / 默认为 30，50~100 为最佳质量")
                length_adjust = gr.Slider(minimum=0.5, maximum=2.0, step=0.1, value=1.0, label="Length Adjust / 长度调整", 
                         info="<1.0 for speed-up speech, >1.0 for slow-down speech / <1.0 加速语速，>1.0 减慢语速")
                intelligibility_cfg_rate = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.5, label="Intelligibility CFG Rate",
                         info="has subtle influence / 有微小影响")
                similarity_cfg_rate = gr.Slider(minimum=0.0, maximum=1.0, step=0.1, value=0.5, label="Similarity CFG Rate",
                          info="has subtle influence / 有微小影响")
                top_p = gr.Slider(minimum=0.1, maximum=1.0, step=0.1, value=0.9, label="Top-p",
                         info="Controls diversity of generated audio / 控制生成音频的多样性")
                temperature = gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Temperature",
                         info="Controls randomness of generated audio / 控制生成音频的随机性")
                repetition_penalty = gr.Slider(minimum=1.0, maximum=3.0, step=0.1, value=1.0, label="Repetition Penalty",
                         info="Penalizes repetition in generated audio / 惩罚生成音频中的重复")
                convert_style = gr.Checkbox(label="convert style", value=False)
                anonymization_only = gr.Checkbox(label="anonymization only", value=False)
        
        examples = [
            [None, "examples/reference/wise_9347.mp3", 100, 1.0, 1.0, 0.0, 0.9, 1.0, 1.0, False, False],
            [None, "examples/reference/wise_9494.mp3", 100, 1.0, 1.0, 0.0, 0.9, 1.0, 1.0, False, False],
            [None, "examples/reference/wise_9495.mp3", 100, 1.0, 1.0, 0.0, 0.9, 1.0, 1.0, False, False],
            ["examples/source/yae_0.wav", "examples/reference/dingzhen_0.wav", 50, 1.0, 0.5, 0.5, 0.9, 1.0, 1.0, False, False],
            ["examples/source/jay_0.wav", "examples/reference/azuma_0.wav", 50, 1.0, 0.5, 0.5, 0.9, 1.0, 1.0, False, False],
        ]
        
        gr.Examples(
            examples=examples,
            inputs=[source_audio, reference_audio, diffusion_steps, length_adjust, intelligibility_cfg_rate, similarity_cfg_rate, top_p, temperature, repetition_penalty, convert_style, anonymization_only],
        )
        
        reload_btn.click(fn=reload_model, inputs=[config_choice, ar_checkpoint_choice, cfm_checkpoint_choice], outputs=[status_bar, config_choice, ar_checkpoint_choice, cfm_checkpoint_choice])
        convert_btn.click(fn=convert_voice_with_streaming_wrapper, 
                         inputs=[source_audio, reference_audio, diffusion_steps, length_adjust, intelligibility_cfg_rate, similarity_cfg_rate, top_p, temperature, repetition_penalty, convert_style, anonymization_only],
                         outputs=[stream_output, full_output, convert_btn])
    
    demo.launch(share=args.share)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="Compile the model using torch.compile")
    parser.add_argument("--config", type=str, default="configs/v2/vc_wrapper.yaml",
                        help="Path to the configuration file")
    parser.add_argument("--fp16", type=str2bool, nargs="?", const=True, help="Whether to use fp16", default=False)
    parser.add_argument("--conf-dir", type=str, help="Directory containing config and checkpoint files", default=None)
    parser.add_argument("--share", type=str2bool, nargs="?", const=True, default=False, help="Whether to share the app")
    # V2 custom checkpoints
    parser.add_argument("--ar-checkpoint-path", type=str, default=None,
                        help="Path to custom checkpoint file")
    parser.add_argument("--cfm-checkpoint-path", type=str, default=None,
                        help="Path to custom checkpoint file")
    args = parser.parse_args()
    
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
    
    main(args)