#!/bin/bash

# 初始化 conda
eval "$(conda shell.bash hook)"

# 激活 seedvc_env 环境
conda activate seedvc_env

# 将HFS路径转换为Unix路径
convert_hfs_to_unix() {
    local hfs_path="$1"
    if [[ -n "$hfs_path" ]]; then
        # 清理HFS路径，移除开头的"alias "前缀
        local clean_path="${hfs_path#alias }"
        
        # 使用AppleScript将HFS路径转换为Unix路径
        local unix_path=$(osascript -e "POSIX path of \"$clean_path\"" 2>/dev/null)
        if [[ $? -eq 0 && -n "$unix_path" ]]; then
            echo "$unix_path"
            return 0
        else
            # 如果直接转换失败，尝试另一种方法
            local unix_path2=$(osascript -e "do shell script \"echo \\\"$clean_path\\\" | sed 's/:/\\//g' | sed 's/^\\(Macintosh HD\\)/\\/\\/'\"" 2>/dev/null)
            if [[ $? -eq 0 && -n "$unix_path2" ]]; then
                # 确保路径以/开头
                if [[ "$unix_path2" != /* ]]; then
                    unix_path2="/$unix_path2"
                fi
                echo "$unix_path2"
                return 0
            fi
        fi
    fi
    # 如果转换失败，返回原始路径
    echo "$hfs_path"
}

# 交互式选择任意文件函数（不限制文件类型）
select_any_file() {
    local prompt="$1"
    local default_path="$2"
    
    # 检查是否在macOS上运行且支持图形界面
    if command -v osascript &> /dev/null; then
        # 使用AppleScript显示文件选择对话框，不限制文件类型
        local script="choose file with prompt \"$prompt\""
        if [[ -n "$default_path" && -d "$default_path" ]]; then
            script+=" default location POSIX file \"$default_path\""
        fi
        local result=$(osascript -e "$script" 2>/dev/null)
        if [[ $? -eq 0 && -n "$result" ]]; then
            # 将HFS路径转换为Unix路径
            local unix_path=$(convert_hfs_to_unix "$result")
            echo "$unix_path"
            return 0
        fi
    fi
    
    # 如果没有图形界面或AppleScript失败，提示用户手动输入路径
    read -p "$prompt: " path
    echo "$path"
}

# 解析命令行参数
INTERACTIVE_MODE=false
APP_TYPE="vc"  # 默认应用类型: vc (voice conversion)
USE_CPU=true
CHECKPOINT=""
CONFIG=""
CFM_CHECKPOINT=""
AR_CHECKPOINT=""
FP16="False"
COMPILE=false
SHARE=false

show_help() {
    echo "用法: $0 [选项]"
    echo "选项:"
    echo "  -v, --vc              运行语音转换 Web UI (默认)"
    echo "  -s, --svc             运行歌声转换 Web UI"
    echo "  -2, --v2              运行 V2 模型 Web UI"
    echo "  -G, --gpu             使用 GPU 运行 (默认使用 CPU)"
    echo "  -p, --checkpoint PATH 指定模型检查点路径"
    echo "  -c, --config PATH     指定模型配置文件路径"
    echo "  -m, --cfm-checkpoint PATH 指定 CFM 模型检查点路径 (仅 V2)"
    echo "  -a, --ar-checkpoint PATH  指定 AR 模型检查点路径 (仅 V2)"
    echo "  -f, --fp16            是否使用 FP16 (默认: False)"
    echo "  -P, --compile         启用编译优化 (仅 V2)"
    echo "  --share               共享 Gradio 应用"
    echo "  -I, --interactive     交互式选择参数"
    echo "  -h, --help            显示此帮助信息"
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -G|--gpu)
            USE_CPU=false
            echo "使用 GPU 运行"
            shift
            ;;
        -v|--vc)
            APP_TYPE="vc"
            echo "运行语音转换 Web UI"
            shift
            ;;
        -s|--svc)
            APP_TYPE="svc"
            echo "运行歌声转换 Web UI"
            shift
            ;;
        -2|--v2)
            APP_TYPE="v2"
            echo "运行 V2 模型 Web UI"
            shift
            ;;
        -p|--checkpoint)
            CHECKPOINT="$2"
            echo "使用检查点: $CHECKPOINT"
            shift 2
            ;;
        -c|--config)
            CONFIG="$2"
            echo "使用配置: $CONFIG"
            shift 2
            ;;
        -m|--cfm-checkpoint)
            CFM_CHECKPOINT="$2"
            echo "使用 CFM 检查点: $CFM_CHECKPOINT"
            shift 2
            ;;
        -a|--ar-checkpoint)
            AR_CHECKPOINT="$2"
            echo "使用 AR 检查点: $AR_CHECKPOINT"
            shift 2
            ;;
        -f|--fp16)
            FP16="True"
            echo "启用 FP16"
            shift
            ;;
        -P|--compile)
            COMPILE=true
            echo "启用编译优化"
            shift
            ;;
        --share)
            SHARE=true
            echo "启用共享"
            shift
            ;;
        -I|--interactive)
            INTERACTIVE_MODE=true
            shift
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "未知参数: $1"
            show_help
            exit 1
            ;;
    esac
done

# 如果启用了交互模式，则让用户选择参数
if [[ "$INTERACTIVE_MODE" = true ]]; then
    echo "=== 交互式参数选择 ==="
    
    # 询问应用类型
    echo "请选择应用类型:"
    echo "1) V1语音转换 Web UI (vc) (默认)"
    echo "2) V1歌声转换 Web UI (svc)"
    echo "3) V2语音转换 Web UI (v2)"
    read -p "请输入选择 (1/2/3): " -n 1 -r
    echo
    if [[ $REPLY == "2" ]]; then
        APP_TYPE="svc"
        echo "已选 V1歌声转换 Web UI"
    elif [[ $REPLY == "3" ]]; then
        APP_TYPE="v2"
        echo "已选 V2语音转换 Web UI"
    else
        APP_TYPE="vc"
        echo "已选 V1语音转换 Web UI"
    fi
    
    # 询问是否强制使用 CPU
    read -p "是否使用 GPU 运行？(y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        USE_CPU=false
        echo "已启用使用 GPU 运行"
    else
        USE_CPU=true
        echo "将使用 CPU 运行"
    fi
    
    # 询问是否共享应用
    read -p "是否共享 Gradio 应用？(y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        SHARE=true
        echo "已启用共享 Gradio 应用"
    fi
    
    # 根据应用类型询问相应的参数
    if [[ "$APP_TYPE" = "vc" ]] || [[ "$APP_TYPE" = "svc" ]]; then
        # VC 和 SVC 共享的参数
        read -p "是否指定模型检查点文件 (--checkpoint)？(y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "请选择模型检查点文件:"
            SELECTED_CHECKPOINT=$(select_any_file "请选择模型检查点文件" "./runs")
            if [[ -n "$SELECTED_CHECKPOINT" ]]; then
                CHECKPOINT="$SELECTED_CHECKPOINT"
                echo "已选择模型检查点文件: $CHECKPOINT"
            else
                echo "未指定模型检查点文件"
            fi
        fi
        
        read -p "是否指定配置文件 (--config)？(y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "请选择配置文件:"
            SELECTED_CONFIG=$(select_any_file "请选择配置文件" "./runs")
            if [[ -n "$SELECTED_CONFIG" ]]; then
                CONFIG="$SELECTED_CONFIG"
                echo "已选择配置文件: $CONFIG"
            else
                echo "未指定配置文件"
            fi
        fi
        
        # 询问 FP16 设置
        read -p "是否使用 FP16 (默认: False)？(y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            FP16="True"
            echo "已启用 FP16"
        else
            FP16="False"
            echo "已禁用 FP16"
        fi
    elif [[ "$APP_TYPE" = "v2" ]]; then
        # V2 特有的参数
        read -p "是否指定 CFM 模型检查点文件 (--cfm-checkpoint)？(y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "请选择 CFM 模型检查点文件:"
            SELECTED_CFM_CHECKPOINT=$(select_any_file "请选择 CFM 模型检查点文件" "./runs")
            if [[ -n "$SELECTED_CFM_CHECKPOINT" ]]; then
                CFM_CHECKPOINT="$SELECTED_CFM_CHECKPOINT"
                echo "已选择 CFM 模型检查点文件: $CFM_CHECKPOINT"
            else
                echo "未指定 CFM 模型检查点文件"
            fi
        fi
        
        read -p "是否指定 AR 模型检查点文件 (--ar-checkpoint)？(y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "请选择 AR 模型检查点文件:"
            SELECTED_AR_CHECKPOINT=$(select_any_file "请选择 AR 模型检查点文件" "./runs")
            if [[ -n "$SELECTED_AR_CHECKPOINT" ]]; then
                AR_CHECKPOINT="$SELECTED_AR_CHECKPOINT"
                echo "已选择 AR 模型检查点文件: $AR_CHECKPOINT"
            else
                echo "未指定 AR 模型检查点文件"
            fi
        fi
        
        read -p "是否指定配置文件 (--config)？(y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "请选择配置文件:"
            SELECTED_CONFIG=$(select_any_file "请选择配置文件" "./runs")
            if [[ -n "$SELECTED_CONFIG" ]]; then
                CONFIG="$SELECTED_CONFIG"
                echo "已选择配置文件: $CONFIG"
            else
                echo "未指定配置文件"
            fi
        fi
        
        # 询问是否启用编译优化
        read -p "是否启用编译优化 (--compile)？(y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            COMPILE=true
            echo "已启用编译优化"
        fi
    fi
    
    # 显示所有选择的参数并请求确认
    echo ""
    echo "=== 参数确认 ==="
    echo "应用类型: $APP_TYPE"
    echo "强制使用 CPU: $USE_CPU"
    echo "共享应用: $SHARE"

    if [[ "$APP_TYPE" = "vc" ]] || [[ "$APP_TYPE" = "svc" ]]; then
        if [[ -n "$CHECKPOINT" ]]; then
            echo "模型检查点: $CHECKPOINT"
        fi
        if [[ -n "$CONFIG" ]]; then
            echo "配置文件: $CONFIG"
        fi
        echo "FP16: $FP16"
    elif [[ "$APP_TYPE" = "v2" ]]; then
        if [[ -n "$CFM_CHECKPOINT" ]]; then
            echo "CFM 模型检查点: $CFM_CHECKPOINT"
        fi
        if [[ -n "$AR_CHECKPOINT" ]]; then
            echo "AR 模型检查点: $AR_CHECKPOINT"
        fi
        if [[ -n "$CONFIG" ]]; then
            echo "配置文件: $CONFIG"
        fi
        echo "编译优化: $COMPILE"
    fi

    # 生成等效的非交互式命令行命令
    echo ""
    echo "=== 等效的非交互式命令行 ==="
    CMD="./web-ui.sh"

    case $APP_TYPE in
        "vc")
            CMD+=" --vc"
            ;;
        "svc")
            CMD+=" --svc"
            ;;
        "v2")
            CMD+=" --v2"
            ;;
    esac
    
    if [[ -n "$CONFIG" ]]; then
        CMD+=" --config \"$CONFIG\""
    fi

    if [[ "$APP_TYPE" = "vc" ]] || [[ "$APP_TYPE" = "svc" ]]; then
        if [[ -n "$CHECKPOINT" ]]; then
            CMD+=" --checkpoint \"$CHECKPOINT\""
        fi
    elif [[ "$APP_TYPE" = "v2" ]]; then
        if [[ -n "$CFM_CHECKPOINT" ]]; then
            CMD+=" --cfm-checkpoint \"$CFM_CHECKPOINT\""
        fi
        if [[ -n "$AR_CHECKPOINT" ]]; then
            CMD+=" --ar-checkpoint \"$AR_CHECKPOINT\""
        fi
        if [[ "$COMPILE" = true ]]; then
            CMD+=" --compile"
        fi
    fi

    if [[ "$USE_CPU" = false ]]; then
        CMD+=" --gpu"
    fi
    
    if [[ "$FP16" = "True" ]]; then
        CMD+=" --fp16"
    fi

    if [[ "$SHARE" = true ]]; then
        CMD+=" --share"
    fi

    echo "$CMD"

    echo ""
    read -p "是否确认启动应用？(y/N/c): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        if [[ $REPLY =~ ^[Cc]$ ]]; then
            echo "已取消执行"
        else
            echo "已取消执行"
        fi
        exit 0
    fi
    echo "启动应用..."
fi

# 设置环境变量
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HUB_OFFLINE=0

# export HUGGING_FACE_HUB_TOKEN={从https://huggingface.co/settings/tokens获取}
export HUGGING_FACE_HUB_TOKEN=

# 如果未指定 --gpu 或运行 V2 版本，则强制使用 CPU
if [ "$USE_CPU" = false ]; then
    export FORCE_CPU=0
    echo "设置 FORCE_CPU=0"
else
    export FORCE_CPU=1
    echo "设置 FORCE_CPU=1"
fi

# 构建命令参数
ARGS=""
if [ "$SHARE" = true ]; then
    ARGS="$ARGS --share True"
fi

if [[ "$FP16" = "True" ]]; then
    ARGS="$ARGS --fp16 True"
else
    ARGS="$ARGS --fp16 False"
fi

if [ -n "$CONFIG" ]; then
    ARGS="$ARGS --config $CONFIG"
fi

# 根据应用类型运行相应的应用程序
if [ "$APP_TYPE" = "vc" ]; then
    if [ -n "$CHECKPOINT" ]; then
        ARGS="$ARGS --checkpoint $CHECKPOINT"
    fi
    echo "运行命令: python app_vc.py $ARGS"
    python app_vc.py $ARGS
elif [ "$APP_TYPE" = "svc" ]; then
    if [ -n "$CHECKPOINT" ]; then
        ARGS="$ARGS --checkpoint $CHECKPOINT"
    fi
    echo "运行命令: python app_svc.py $ARGS"
    python app_svc.py $ARGS
elif [ "$APP_TYPE" = "v2" ]; then
    if [ -n "$CFM_CHECKPOINT" ]; then
        ARGS="$ARGS --cfm-checkpoint-path $CFM_CHECKPOINT"
    fi
    if [ -n "$AR_CHECKPOINT" ]; then
        ARGS="$ARGS --ar-checkpoint-path $AR_CHECKPOINT"
    fi
    if [ "$COMPILE" = true ]; then
        ARGS="$ARGS --compile"
    fi
    echo "运行命令: python app_vc_v2.py $ARGS"
    python app_vc_v2.py $ARGS
fi