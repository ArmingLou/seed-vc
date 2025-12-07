#!/bin/bash

# 批量语音变声处理脚本
# 支持V1和V2版本，支持wav和mp3格式，可选择使用CPU或GPU

# ====================
# Conda环境设置（可选）
# ====================
# 如果使用conda环境，请取消下面几行的注释并根据实际情况修改
# 初始化 conda
eval "$(conda shell.bash hook)" || echo "警告: conda初始化失败"

# 激活指定的conda环境
conda activate seedvc_env 2>/dev/null || echo "警告: 无法激活seedvc_env环境"

# 检查是否安装了必要的命令
check_dependencies() {
    if ! command -v python &> /dev/null; then
        echo "错误: 未找到python命令，请确保已安装Python"
        exit 1
    fi
    
    if ! command -v find &> /dev/null; then
        echo "错误: 未找到find命令"
        exit 1
    fi
}

# 交互式选择目录函数
select_directory() {
    local prompt="$1"
    local default_path="$2"
    
    # 检查是否在macOS上运行
    if command -v osascript &> /dev/null; then
        # 使用AppleScript显示目录选择对话框
        local script="choose folder with prompt \"$prompt\""
        if [[ -n "$default_path" && -d "$default_path" ]]; then
            script+=" default location POSIX file \"$default_path\""
        fi
        local result=$(osascript -e "$script" 2>/dev/null)
        if [[ $? -eq 0 && -n "$result" ]]; then
            # 将AppleScript的HFS路径转换为Unix路径
            local unix_path=$(convert_hfs_to_unix "$result")
            echo "$unix_path"
        else
            echo ""
        fi
    else
        # 如果没有图形界面，提示用户手动输入路径
        read -p "$prompt: " -e -i "$default_path" path
        echo "$path"
    fi
}

# 交互式选择文件函数
select_file() {
    local prompt="$1"
    local default_path="$2"
    local file_types="$3"  # 文件类型过滤器，如"wav,mp3"
    
    # 检查是否在macOS上运行
    if command -v osascript &> /dev/null; then
        # 使用AppleScript显示文件选择对话框
        local script="choose file with prompt \"$prompt\""
        if [[ -n "$default_path" && -d "$default_path" ]]; then
            script+=" default location POSIX file \"$default_path\""
        fi
        if [[ -n "$file_types" ]]; then
            # 添加文件类型过滤器
            local type_list=""
            IFS=',' read -ra TYPES <<< "$file_types"
            for type in "${TYPES[@]}"; do
                type_list+="\"${type}\", "
            done
            type_list=${type_list%, }
            script+=" of type {$type_list}"
        fi
        local result=$(osascript -e "$script" 2>/dev/null)
        if [[ $? -eq 0 && -n "$result" ]]; then
            # 将AppleScript的HFS路径转换为Unix路径
            local unix_path=$(convert_hfs_to_unix "$result")
            echo "$unix_path"
        else
            echo ""
        fi
    else
        # 如果没有图形界面，提示用户手动输入路径
        read -p "$prompt: " -e -i "$default_path" path
        echo "$path"
    fi
}

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

# 默认参数设置
VERSION="v1"              # 默认使用V1版本
USE_CPU=true             # 默认使用CPU
INPUT_DIR=""              # 输入目录
REFERENCE_FILE=""         # 参考音频文件
OUTPUT_DIR=""             # 输出目录，默认为输入目录的父目录下的seedvc-output
INTELLIGIBILITY_RATE=1.0  # V2版本的intelligibility-cfg-rate，默认1.0
SIMILARITY_RATE=0.0       # V2版本的similarity-cfg-rate，默认0.0
INFERENCE_CFG_RATE=1.0    # V1版本的inference-cfg-rate，默认1.0，范围0.0~1.0
SEMI_TONE_SHIFT=0         # V1版本的semi-tone-shift，默认0(-24~24)
SONG=false                # V1版本是否歌手转换
F0_CONDITION="False"      # V1版本的--f0-condition，默认False
AUTO_F0_ADJUST="False"    # V1版本的--auto-f0-adjust，默认False
FP16="False"                # 是否使用fp16精度，默认false

# 新增参数
CHECKPOINT=""            # 模型检查点路径
CONFIG=""                # 模型配置文件路径
CFM_CHECKPOINT=""        # V2 CFM模型检查点路径
AR_CHECKPOINT=""         # V2 AR模型检查点路径

# 固定参数
DIFFUSION_STEPS=100       # 固定扩散步数为100

# 显示帮助信息
show_help() {
    echo "用法: $0 [选项] --input-dir <目录> --reference <参考音频> --output-dir <输出目录>"
    echo ""
    echo "通用选项:"
    echo "  -i, --input-dir DIR          指定包含待处理音频文件的输入目录"
    echo "  -r, --reference FILE         指定参考音频文件"
    echo "  -o, --output-dir DIR         指定输出目录 (默认: 输入目录的父目录下，格式为 输入目录名-converted-时间戳)"
    echo "      --v1                     使用V1版本 (默认)"
    echo "      --v2                     使用V2版本"
    echo "      --gpu                    使用GPU运行 (默认使用CPU)"
    echo "      --fp16                   使用fp16精度 (默认: false)"
    echo ""
    echo "V1版本专用参数:"
    echo "  -S, --song                   使用V1的歌声转换"
    echo "  -P, --pitch SHIFT            控制音调(默认: 0 recommended -24~24. 同时 使用了--song才有效)"
    echo "  -L, --inference-cfg-rate RATE 控制推理CFG率(默认: 1.0 recommended 0.0~1.0)"
    echo "  -A, --auto-f0-adjust         自动调整F0 (默认: false)"
    echo "  -p, --checkpoint PATH        指定模型检查点路径"
    echo "  -c, --config PATH            指定模型配置文件路径"
    echo ""
    echo "V2版本专用参数:"
    echo "  -l, --intelligibility RATE   控制发音清晰度(默认: 1.0 recommended 0.0~1.0)"
    echo "  -s, --similarity RATE        控制与参考音频的相似度(默认: 0.0 recommended 0.0~1.0)"
    echo "  -f, --cfm-checkpoint PATH    指定 CFM 模型检查点路径"
    echo "  -a, --ar-checkpoint PATH     指定 AR 模型检查点路径"
    echo "  -c, --config PATH            指定模型配置文件路径"
    echo ""
    echo "其他选项:"
    echo "  -h, --help                   显示此帮助信息"
    echo "  -I, --interactive            交互式选择输入目录、参考文件和输出目录，以及其他可选参数"
    echo ""
    echo "离线模式:"
    echo "  如果您已经下载了所有模型文件，可以通过设置以下环境变量来启用离线模式:"
    echo "    export HF_HUB_OFFLINE=1"
    echo "  模型文件应位于以下目录之一:"
    echo "    系统默认Hugging Face缓存: $HOME/cache/huggingface/hub/"
    echo "    自定义模型: ./checkpoints/"
    echo ""
    echo "示例:"
    echo "  $0 -i ./sources -r ./ref.wav -o ./converted"
    echo "  $0 -i ./sources -r ./ref.wav --v2 --cpu"
    echo "  HF_HUB_OFFLINE=1 $0 -i ./sources -r ./ref.wav"
    echo "  $0 -I"
    echo "  $0 -I --v2"
}

# 解析命令行参数
INTERACTIVE_MODE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--input-dir)
            INPUT_DIR="$2"
            shift 2
            ;;
        -r|--reference)
            REFERENCE_FILE="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --v1)
            VERSION="v1"
            shift
            ;;
        --v2)
            VERSION="v2"
            shift
            ;;
        --gpu)
            USE_CPU=false
            shift
            ;;
        -S|--song)
            SONG=true
            shift
            ;;
        -L|--inference-cfg-rate)
            INFERENCE_CFG_RATE="$2"
            shift 2
            ;;
        --fp16)
            FP16="True"
            shift
            ;;
        -A|--auto-f0-adjust)
            AUTO_F0_ADJUST="True"
            shift
            ;;
        -l|--intelligibility)
            INTELLIGIBILITY_RATE="$2"
            shift 2
            ;;
        -s|--similarity)
            SIMILARITY_RATE="$2"
            shift 2
            ;;
        -P|--pitch)
            SEMI_TONE_SHIFT="$2"
            shift 2
            ;;
        -p|--checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        -c|--config)
            CONFIG="$2"
            shift 2
            ;;
        -f|--cfm-checkpoint)
            CFM_CHECKPOINT="$2"
            shift 2
            ;;
        -a|--ar-checkpoint)
            AR_CHECKPOINT="$2"
            shift 2
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

# 检查必填参数（非交互模式下）
if [[ "$INTERACTIVE_MODE" = false ]]; then
    if [[ -z "$INPUT_DIR" ]]; then
        echo "错误: 输入目录 (--input-dir|-i) 是必填参数"
        show_help
        exit 1
    fi
    
    if [[ -z "$REFERENCE_FILE" ]]; then
        echo "错误: 参考音频文件 (--reference|-r) 是必填参数"
        show_help
        exit 1
    fi
    
    # 输出目录不是必填项，有默认逻辑
fi

# 如果启用了交互模式，则让用户选择路径
if [[ "$INTERACTIVE_MODE" = true ]]; then
    echo "=== 交互式路径选择 ==="
    
    # 询问使用哪个版本
    echo "请选择使用的版本:"
    echo "1) V1版本 (默认)"
    echo "2) V2版本"
    read -p "请输入选择 (1/2): " -n 1 -r
    echo
    if [[ $REPLY == "2" ]]; then
        VERSION="v2"
        echo "已选择V2版本"
    else
        VERSION="v1"
        echo "已选择V1版本"
    fi
    
    # 询问是否使用CPU（仅在非交互模式下未指定时）
    if [[ -z "$USE_CPU" || "$USE_CPU" = "false" ]]; then
        read -p "是否使用GPU运行？(y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            USE_CPU=false
            echo "已选择使用GPU运行"
        else
            USE_CPU=true
            echo "将使用CPU运行"
        fi
    fi
    
    # 选择输入目录（必填）
    while [[ -z "$SELECTED_INPUT_DIR" ]]; do
        echo "请选择输入目录（包含待处理音频文件的目录） (必填):"
        SELECTED_INPUT_DIR=$(select_directory "请选择输入目录" "$INPUT_DIR")
        if [[ -n "$SELECTED_INPUT_DIR" ]]; then
            INPUT_DIR="$SELECTED_INPUT_DIR"
            echo "已选择输入目录: $INPUT_DIR"
        else
            echo "输入目录为必填项，请重新选择。"
        fi
    done
    
    # 选择参考音频文件（必填）
    while [[ -z "$SELECTED_REFERENCE_FILE" ]]; do
        echo "请选择参考音频文件 (必填):"
        SELECTED_REFERENCE_FILE=$(select_file "请选择参考音频文件" "$REFERENCE_FILE" "wav,mp3")
        if [[ -n "$SELECTED_REFERENCE_FILE" ]]; then
            REFERENCE_FILE="$SELECTED_REFERENCE_FILE"
            echo "已选择参考音频文件: $REFERENCE_FILE"
        else
            echo "参考音频文件为必填项，请重新选择。"
        fi
    done
    
    # 询问是否指定输出目录
    read -p "是否指定输出目录？(y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "请选择输出目录:"
        SELECTED_OUTPUT_DIR=$(select_directory "请选择输出目录" "$OUTPUT_DIR")
        if [[ -n "$SELECTED_OUTPUT_DIR" ]]; then
            OUTPUT_DIR="$SELECTED_OUTPUT_DIR"
            echo "已选择输出目录: $OUTPUT_DIR"
        else
            echo "未指定输出目录，将使用默认逻辑生成输出目录"
        fi
    else
        echo "将使用默认逻辑生成输出目录"
    fi
    
    # 询问是否使用fp16精度
    read -p "是否使用fp16精度 (--fp16)？(y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        FP16="True"
        echo "已启用fp16精度"
    else
        FP16="False"
        echo "已禁用fp16精度"
    fi
    
    # 根据版本选择显示不同的参数选项
    if [[ "$VERSION" = "v1" ]]; then
        # V1版本特有的参数
        # 询问是否使用歌声转换
        read -p "是否使用歌声转换 (--song)？(y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            SONG=true
            echo "已启用歌声转换"
            
            # 询问音调调整值（有默认值和范围）
            read -p "请输入音调调整值 (-24~24, 默认: 0): " pitch_value
            if [[ -n "$pitch_value" ]]; then
                SEMI_TONE_SHIFT="$pitch_value"
            else
                SEMI_TONE_SHIFT=0
            fi
            echo "已设置音调调整值: $SEMI_TONE_SHIFT"
        else
            SONG=false
        fi
        
        # 询问是否自动调整F0
        read -p "是否自动调整F0 (--auto-f0-adjust)？(y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            AUTO_F0_ADJUST="True"
            echo "已启用自动调整F0"
        else
            AUTO_F0_ADJUST="False"
            echo "已关闭自动调整F0"
        fi
        
        # 询问是否指定checkpoint文件（仅V1）
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
        
        # 询问是否指定config文件（V1）
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
        
        # 询问推理CFG率（有默认值和范围）
        read -p "请输入推理CFG率 (0.0~1.0, 默认: 1.0): " inference_cfg_value
        if [[ -n "$inference_cfg_value" ]]; then
            INFERENCE_CFG_RATE="$inference_cfg_value"
        else
            INFERENCE_CFG_RATE=1.0
        fi
        echo "已设置推理CFG率: $INFERENCE_CFG_RATE"
        
        
    else
        # V2版本特有的参数
        # 询问发音清晰度（有默认值和范围）
        read -p "请输入发音清晰度 (0.0~1.0, 默认: 1.0): " intelligibility_value
        if [[ -n "$intelligibility_value" ]]; then
            INTELLIGIBILITY_RATE="$intelligibility_value"
        else
            INTELLIGIBILITY_RATE=1.0
        fi
        echo "已设置发音清晰度: $INTELLIGIBILITY_RATE"
        
        # 询问相似度（有默认值和范围）
        read -p "请输入相似度 (0.0~1.0, 默认: 0.0): " similarity_value
        if [[ -n "$similarity_value" ]]; then
            SIMILARITY_RATE="$similarity_value"
        else
            SIMILARITY_RATE=0.0
        fi
        echo "已设置相似度: $SIMILARITY_RATE"
        
        # 询问是否指定CFM检查点文件
        read -p "是否指定CFM模型检查点文件 (--cfm-checkpoint)？(y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "请选择CFM模型检查点文件:"
            SELECTED_CFM_CHECKPOINT=$(select_any_file "请选择CFM模型检查点文件" "./runs")
            if [[ -n "$SELECTED_CFM_CHECKPOINT" ]]; then
                CFM_CHECKPOINT="$SELECTED_CFM_CHECKPOINT"
                echo "已选择CFM模型检查点文件: $CFM_CHECKPOINT"
            else
                echo "未指定CFM模型检查点文件"
            fi
        fi
        
        # 询问是否指定AR检查点文件
        read -p "是否指定AR模型检查点文件 (--ar-checkpoint)？(y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "请选择AR模型检查点文件:"
            SELECTED_AR_CHECKPOINT=$(select_any_file "请选择AR模型检查点文件" "./runs")
            if [[ -n "$SELECTED_AR_CHECKPOINT" ]]; then
                AR_CHECKPOINT="$SELECTED_AR_CHECKPOINT"
                echo "已选择AR模型检查点文件: $AR_CHECKPOINT"
            else
                echo "未指定AR模型检查点文件"
            fi
        fi
        
        # 询问是否指定config文件（V2）
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
    fi
    
    # 显示所有选择的参数并请求确认
    echo ""
    echo "=== 参数确认 ==="
    echo "版本: $VERSION"
    echo "输入目录: $INPUT_DIR"
    echo "参考音频: $REFERENCE_FILE"

    # 如果输出目录未指定，显示默认生成的路径
    if [[ -z "$OUTPUT_DIR" ]]; then
        # 生成默认输出目录路径
        INPUT_DIR_NAME=$(basename "$INPUT_DIR")
        TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
        DEFAULT_OUTPUT_DIR="${INPUT_DIR%/*}/seedvc-output/${INPUT_DIR_NAME}-converted-${TIMESTAMP}"
        echo "输出目录: $DEFAULT_OUTPUT_DIR (默认)"
    else
        echo "输出目录: $OUTPUT_DIR"
    fi

    # 显示CPU使用情况
    if [[ "$USE_CPU" = true ]]; then
        echo "强制使用CPU: 是"
    elif [[ "$VERSION" = "v2" ]]; then
        echo "强制使用CPU: 是 (V2版本自动强制在CPU上运行)"
    else
        echo "强制使用CPU: 否（将使用GPU，如果可用）"
    fi

    if [[ "$VERSION" = "v1" ]]; then
        if [[ "$SONG" = true ]]; then
            echo "歌声转换: 是"
            echo "音调调整: $SEMI_TONE_SHIFT"
        else
            echo "歌声转换: 否"
        fi
        echo "自动调整F0: $AUTO_F0_ADJUST"
        echo "推理CFG率: $INFERENCE_CFG_RATE"
        echo "使用fp16精度: $FP16"
        if [[ -n "$CHECKPOINT" ]]; then
            echo "模型检查点: $CHECKPOINT"
        fi
    else
        echo "发音清晰度: $INTELLIGIBILITY_RATE"
        echo "相似度: $SIMILARITY_RATE"
        if [[ -n "$CFM_CHECKPOINT" ]]; then
            echo "CFM模型检查点: $CFM_CHECKPOINT"
        fi
        if [[ -n "$AR_CHECKPOINT" ]]; then
            echo "AR模型检查点: $AR_CHECKPOINT"
        fi
    fi

    if [[ -n "$CONFIG" ]]; then
        echo "配置文件: $CONFIG"
    fi

    # 生成等效的非交互式命令行命令
    echo ""
    echo "=== 等效的非交互式命令行 ==="
    CMD="./batch_voice_conversion.sh"
    CMD+=" --${VERSION}"

    CMD+=" --input-dir \"$INPUT_DIR\""
    CMD+=" --reference \"$REFERENCE_FILE\""

    if [[ -n "$OUTPUT_DIR" ]]; then
        CMD+=" --output-dir \"$OUTPUT_DIR\""
    fi
    
    if [[ -n "$CONFIG" ]]; then
        CMD+=" --config \"$CONFIG\""
    fi

    if [[ "$VERSION" = "v1" ]]; then
        if [[ -n "$CHECKPOINT" ]]; then
            CMD+=" --checkpoint \"$CHECKPOINT\""
        fi
        if [[ "$SONG" = true ]]; then
            CMD+=" --song"
            if [[ "$SEMI_TONE_SHIFT" != "0" ]]; then
                CMD+=" --semi-tone-shift $SEMI_TONE_SHIFT"
            fi
        fi
        if [[ "$AUTO_F0_ADJUST" = "True" ]]; then
            CMD+=" --auto-f0-adjust"
        fi
        if [[ "$INFERENCE_CFG_RATE" != "0.0" ]]; then
            CMD+=" --inference-cfg-rate $INFERENCE_CFG_RATE"
        fi
    else
        if [[ -n "$CFM_CHECKPOINT" ]]; then
            CMD+=" --cfm-checkpoint \"$CFM_CHECKPOINT\""
        fi
        if [[ -n "$AR_CHECKPOINT" ]]; then
            CMD+=" --ar-checkpoint \"$AR_CHECKPOINT\""
        fi
        if [[ "$INTELLIGIBILITY_RATE" != "1.0" ]]; then
            CMD+=" --intelligibility $INTELLIGIBILITY_RATE"
        fi
        if [[ "$SIMILARITY_RATE" != "0.0" ]]; then
            CMD+=" --similarity $SIMILARITY_RATE"
        fi
    fi
    
    if [[ "$USE_CPU" = false ]]; then
        CMD+=" --gpu"
    fi
    
    if [[ "$FP16" = "True" ]]; then
        CMD+=" --fp16"
    fi

    echo "$CMD"

    echo ""
    read -p "是否确认开始处理？(y/N/c): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        if [[ $REPLY =~ ^[Cc]$ ]]; then
            echo "已取消执行"
        else
            echo "已取消执行"
        fi
        exit 0
    fi
    echo "开始处理..."
fi

# 检查依赖
check_dependencies

# 检查必需参数
if [[ -z "$INPUT_DIR" ]] || [[ -z "$REFERENCE_FILE" ]]; then
    echo "错误: 必须指定 --input-dir 和 --reference 参数"
    if [[ "$INTERACTIVE_MODE" = false ]]; then
        show_help
    fi
    exit 1
fi

# 检查输入目录是否存在
if [[ ! -d "$INPUT_DIR" ]]; then
    echo "错误: 输入目录 '$INPUT_DIR' 不存在"
    exit 1
fi

# 检查参考音频文件是否存在
if [[ ! -f "$REFERENCE_FILE" ]]; then
    echo "错误: 参考音频文件 '$REFERENCE_FILE' 不存在"
    exit 1
fi

# 如果未指定输出目录，则使用输入目录的父目录下，目录名格式为"输入目录名-converted-动态时间值"
if [[ -z "$OUTPUT_DIR" ]]; then
    INPUT_DIR_PARENT=$(dirname "$INPUT_DIR")
    INPUT_DIR_NAME=$(basename "$INPUT_DIR")
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="$INPUT_DIR_PARENT/${INPUT_DIR_NAME}-converted-${TIMESTAMP}"
fi

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 设置环境变量
export PYTORCH_ENABLE_MPS_FALLBACK=1

# 设置Hugging Face相关环境变量以改善网络连接
# 只有在没有设置HF_ENDPOINT时才设置默认值
# export HF_ENDPOINT="https://hf-mirror.com"

export HUGGING_FACE_HUB_TOKEN=
# 强制设置为在线模式以确保能下载缺失的模型
export HF_HUB_OFFLINE=0
# export HF_HUB_CACHE="$HOME/cache/huggingface/hub"
# export HF_DATASETS_CACHE="$HOME/cache/huggingface/datasets"
# export HF_HUB_DISABLE_TELEMETRY=1
# export HF_HUB_DISABLE_PROGRESS_BARS=1

# 智能检测离线模式
# 如果用户明确设置了HF_HUB_OFFLINE，则使用用户设置
# 否则检查是否存在缓存文件来决定是否启用离线模式
# if [[ -z "$HF_HUB_OFFLINE" ]]; then
#     # 只检查$HOME/cache/huggingface/hub目录
#     if [[ -d "$HOME/cache/huggingface/hub" ]] && [[ "$(ls -A $HOME/cache/huggingface/hub 2>/dev/null)" ]]; then
#         echo "检测到本地缓存目录中存在模型文件，自动启用离线模式"
#         export HF_HUB_OFFLINE=1
#         echo "系统默认缓存目录: $HOME/cache/huggingface/hub"
#     else
#         echo "未检测到本地缓存文件，将尝试在线下载模型"
#         echo "系统默认缓存目录: $HOME/cache/huggingface/hub"
#         export HF_HUB_OFFLINE=0
#     fi
# else
#     echo "使用用户设置的HF_HUB_OFFLINE值: $HF_HUB_OFFLINE"
# fi

# 检查离线模式下必要的模型文件是否存在
# if [[ "$HF_HUB_OFFLINE" == "1" ]]; then
#     required_models=(
#         "models--openai--whisper-small"
#         "models--nvidia--bigvgan_v2_22khz_80band_256x"
#         "models--nvidia--bigvgan_v2_44khz_128band_512x"
#         # "models--facebook--hubert-large-ls960-ft"
#         # "models--facebook--wav2vec2-xls-r-300m"
#     )
#     
#     missing_models=()
#     for model in "${required_models[@]}"; do
#         model_found=false
#         
#         # 只检查$HOME/cache/huggingface/hub目录
#         model_path="$HOME/cache/huggingface/hub/${model}"
#         echo "检查默认缓存目录: $model_path"
#         if [[ -d "$model_path" ]] && [[ -n "$(ls -A "$model_path" 2>/dev/null)" ]]; then
#             echo "  目录存在且不为空"
#             # 检查models子目录或snapshots子目录是否存在
#             models_dir="$model_path/models"
#             snapshots_dir="$model_path/snapshots"
#             echo "  检查models目录: $models_dir"
#             echo "  检查snapshots目录: $snapshots_dir"
#             if [[ -d "$models_dir" ]] && [[ -n "$(ls -A "$models_dir" 2>/dev/null)" ]]; then
#                 echo "  models目录存在且不为空"
#                 model_found=true
#             elif [[ -d "$snapshots_dir" ]] && [[ -n "$(ls -A "$snapshots_dir" 2>/dev/null)" ]]; then
#                 echo "  snapshots目录存在且不为空"
#                 model_found=true
#             else
#                 echo "  models和snapshots目录都不存在或为空"
#             fi
#         else
#             echo "  目录不存在或为空"
#         fi
#         
#         # 如果没有找到模型，添加到缺失列表中
#         if [[ "$model_found" = false ]]; then
#             echo "  模型未找到，添加到缺失列表"
#             missing_models+=("$model")
#         else
#             echo "  模型已找到"
#         fi
#         echo ""
#     done
#     
#     if [[ ${#missing_models[@]} -gt 0 ]]; then
#         echo "警告: 离线模式下缺少以下模型文件:"
#         for model in "${missing_models[@]}"; do
#             echo "  - $model"
#         done
#         echo "请确保所有必要的模型文件都已下载到 $HOME/cache/huggingface/hub 目录中"
#         echo "或者设置HF_HUB_OFFLINE=0以在线下载模型"
#         read -p "是否继续执行？(y/N): " -n 1 -r
#         echo
#         if [[ ! $REPLY =~ ^[Yy]$ ]]; then
#             echo "操作已取消"
#             exit 1
#         fi
#     fi
# fi

echo ""
echo "如果您已经下载了所有必要的模型文件，请确保:"
echo "1. Hugging Face模型文件位于以下目录之一:"
echo "   - 系统默认缓存目录: $HOME/cache/huggingface/hub/"
echo "2. 自定义模型文件位于 ./checkpoints/ 目录中"
echo "3. 或者手动设置 export HF_HUB_OFFLINE=1 来强制启用离线模式"
echo ""

# 添加网络调试信息
echo "当前HF_ENDPOINT设置为: $HF_ENDPOINT"

# 如果存在网络问题，尝试使用不同的镜像或直接访问
if [[ -n "$http_proxy" ]] || [[ -n "$https_proxy" ]]; then
    echo "检测到代理设置，请检查代理配置是否正确"
    echo "HTTP_PROXY: $http_proxy"
    echo "HTTPS_PROXY: $https_proxy"
fi

# 如果遇到持续的网络问题，您可以手动下载以下模型文件:
echo "如果遇到持续的网络问题，您可以手动下载以下模型文件:"
echo "1. Whisper模型: https://huggingface.co/openai/whisper-small"
echo "2. Hubert模型: https://huggingface.co/facebook/hubert-large-ll60k"
echo "3. BigVGAN模型: https://huggingface.co/nvidia/bigvgan_v2_22khz_80band_256x"
echo "下载后将文件放置在正确的缓存目录中。"
echo ""

# 设置FORCE_CPU环境变量来强制使用CPU
if [[ "$USE_CPU" = true ]]; then
    export FORCE_CPU=1
    echo "使用 CPU 运行"
else
    export FORCE_CPU=0
    echo "使用 GPU 运行 (如果可用)"
fi

# 设置FORCE_CPU环境变量来强制使用CPU
if [[ "$SONG" = true ]]; then
    F0_CONDITION="True"
fi

# 获取输入目录中的所有wav和mp3文件
AUDIO_FILES=()
while IFS= read -r -d '' file; do
    AUDIO_FILES+=("$file")
done < <(find "$INPUT_DIR" -type f \( -iname "*.wav" -o -iname "*.mp3" \) -print0)

# 检查是否有找到音频文件
if [[ ${#AUDIO_FILES[@]} -eq 0 ]]; then
    echo "警告: 在目录 '$INPUT_DIR' 中未找到任何wav或mp3文件"
    exit 0
fi

echo "找到 ${#AUDIO_FILES[@]} 个音频文件"
echo "参考音频: $REFERENCE_FILE"
echo "输出目录: $OUTPUT_DIR"
echo "模型版本: $VERSION"
echo "扩散步数: $DIFFUSION_STEPS"
if [[ "$VERSION" = "v2" ]]; then
    echo "Intelligibility CFG Rate: $INTELLIGIBILITY_RATE"
    echo "Similarity CFG Rate: $SIMILARITY_RATE"
    if [[ -n "$CFM_CHECKPOINT" ]]; then
        echo "CFM Checkpoint: $CFM_CHECKPOINT"
    fi
    if [[ -n "$AR_CHECKPOINT" ]]; then
        echo "AR Checkpoint: $AR_CHECKPOINT"
    fi
else
    echo "inference-cfg-rate: $INFERENCE_CFG_RATE"
    echo "f0-condition: $F0_CONDITION"
    echo "auto-f0-adjust: $AUTO_F0_ADJUST"
    echo "semi-tone-shift: $SEMI_TONE_SHIFT"
    if [[ -n "$CHECKPOINT" ]]; then
        echo "Checkpoint: $CHECKPOINT"
    fi
    if [[ -n "$CONFIG" ]]; then
        echo "Config: $CONFIG"
    fi
fi
echo ""

# 处理每个音频文件
PROCESSED_COUNT=0
FAILED_COUNT=0

for audio_file in "${AUDIO_FILES[@]}"; do
    ((PROCESSED_COUNT++))
    echo "[$PROCESSED_COUNT/${#AUDIO_FILES[@]}] 处理: $(basename "$audio_file")"
    
    # 获取文件名（不含扩展名）
    filename=$(basename "$audio_file")
    filename="${filename%.*}"
    
    # 构建输出文件路径
    output_file="$OUTPUT_DIR/${filename}_converted.wav"
    
    # 根据版本选择不同的处理命令
    if [[ "$VERSION" = "v1" ]]; then
        # V1版本处理命令
        CMD="python inference.py \
            --source \"$audio_file\" \
            --target \"$REFERENCE_FILE\" \
            --output \"$OUTPUT_DIR\" \
            --f0-condition $F0_CONDITION \
            --auto-f0-adjust $AUTO_F0_ADJUST \
            --semi-tone-shift $SEMI_TONE_SHIFT \
            --diffusion-steps $DIFFUSION_STEPS \
            --inference-cfg-rate $INFERENCE_CFG_RATE \
            --fp16 $FP16"
        
        # 添加checkpoint和config参数（如果指定）
        if [[ -n "$CHECKPOINT" ]]; then
            CMD="$CMD --checkpoint \"$CHECKPOINT\""
        fi
        if [[ -n "$CONFIG" ]]; then
            CMD="$CMD --config \"$CONFIG\""
        fi
        
        # 执行命令
        eval $CMD
    else
        # V2版本处理命令
        CMD="python inference_v2.py \
            --source \"$audio_file\" \
            --target \"$REFERENCE_FILE\" \
            --output \"$OUTPUT_DIR\" \
            --diffusion-steps $DIFFUSION_STEPS \
            --intelligibility-cfg-rate $INTELLIGIBILITY_RATE \
            --similarity-cfg-rate $SIMILARITY_RATE \
            --fp16 $FP16"
        
        # 添加checkpoint参数（如果指定）
        if [[ -n "$CFM_CHECKPOINT" ]]; then
            CMD="$CMD --cfm-checkpoint-path \"$CFM_CHECKPOINT\""
        fi
        if [[ -n "$AR_CHECKPOINT" ]]; then
            CMD="$CMD --ar-checkpoint-path \"$AR_CHECKPOINT\""
        fi
        if [[ -n "$CONFIG" ]]; then
            CMD="$CMD --config \"$CONFIG\""
        fi
        
        # 执行命令
        eval $CMD
    fi
    
    # 检查命令执行结果
    if [[ $? -eq 0 ]]; then
        echo "  ✓ 处理成功"
    else
        echo "  ✗ 处理失败"
        ((FAILED_COUNT++))
    fi
done

# 输出处理结果统计
echo ""
echo "处理完成!"
echo "总共处理: $PROCESSED_COUNT 个文件"
if [[ $FAILED_COUNT -eq 0 ]]; then
    echo "全部成功! 🎉"
else
    echo "成功: $((PROCESSED_COUNT - FAILED_COUNT)) 个文件"
    echo "失败: $FAILED_COUNT 个文件"
fi