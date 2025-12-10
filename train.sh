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

# 交互式选择目录函数
select_directory() {
    local prompt="$1"
    local default_path="$2"
    
    # 检查是否在macOS上运行且支持图形界面
    if command -v osascript &> /dev/null; then
        # 使用AppleScript显示目录选择对话框
        local script="choose folder with prompt \"$prompt\""
        if [[ -n "$default_path" && -d "$default_path" ]]; then
            script+=" default location POSIX file \"$default_path\""
        fi
        local result=$(osascript -e "$script" 2>/dev/null)
        if [[ $? -eq 0 && -n "$result" ]]; then
            # 将HFS路径转换为Unix路径
            local unix_path=$(convert_hfs_to_unix "$result")
            echo "$unix_path"
            return 0
        else
            echo ""
            return 1
        fi
    fi
    
    # 如果没有图形界面或AppleScript失败，提示用户手动输入路径
    read -p "$prompt: " path
    echo "$path"
}

# 交互式选择任意文件函数（不限制文件类型）
select_any_file() {
    local prompt="$1"
    local default_path="$2"
    
    # 检查是否在macOS上运行且支持图形界面
    if command -v osascript &> /dev/null; then
        # 使用AppleScript显示文件选择对话框，不限制文件类型
        local script="choose file with prompt \"$prompt\""
        if [[ -n "$default_path" && -f "$default_path" ]]; then
            script+=" default location POSIX file \"$default_path\""
        elif [[ -n "$default_path" && -d "$default_path" ]]; then
            # 如果默认路径是目录，则使用该目录作为默认位置
            script+=" default location POSIX file \"$default_path\""
        fi
        local result=$(osascript -e "$script" 2>/dev/null)
        if [[ $? -eq 0 && -n "$result" ]]; then
            # 将HFS路径转换为Unix路径
            local unix_path=$(convert_hfs_to_unix "$result")
            echo "$unix_path"
            return 0
        else
            echo ""
            return 1
        fi
    fi
    
    # 如果没有图形界面或AppleScript失败，提示用户手动输入路径
    read -p "$prompt: " path
    echo "$path"
}

# 解析命令行参数
INTERACTIVE_MODE=false
VERSION="v1"  # 默认版本
USE_CPU=true
RUN_NAME="Test_ft"
CONFIG=""
DATASET_DIR=""  # 移除默认值，使其成为必填项
VAL_DATASET_DIR=""  # 验证集目录
MAX_STEPS=1000
MAX_EPOCHS=1000
SAVE_EVERY=100
TRAIN_CFM=false
TRAIN_AR=false
FP16=false  # 是否使用FP16，默认false
PATIENCE=25 # 建议至少覆盖一个epoch。
VALIDATION_INTERVAL=10 # 建议约验证样本数/batch_size 。validation_interval × patience × batch_size >= 训练样本数 。validation_interval × batch_size = 验证样本数
DISTILL=false
# V2版本的蒸馏参数
DISTILL_AR=false
DISTILL_CFM=false
MIN_LR=1e-7
LR_ADJUST_INTERVAL=10
INITIAL_LR=1e-5 # batch×2 → lr÷4 目前粤语如果batch size 8 。声调敏感学习率更低用1e-5
WARMUP_STEPS=100 # batch×2 → steps÷2 总样本数的4%左右
RESUME_LR=0.0 # 恢复训练时的学习率，默认为0.0表示使用检查点中的学习率

# 新增语言参数变量
LANGUAGE=""

# 新增预训练模型检查点变量
PRETRAINED_CKPT=""
PRETRAINED_CFM_CKPT=""
PRETRAINED_AR_CKPT=""

# 日志文件路径
LOG_FILE=""



while [[ $# -gt 0 ]]; do
    case $1 in
        --gpu|-G)
            USE_CPU=false
            echo "使用 GPU 运行"
            shift
            ;;
        --v1)
            VERSION="v1"
            echo "运行 V1 版本"
            shift
            ;;
        --v2)
            VERSION="v2"
            echo "运行 V2 版本"
            shift
            ;;
        --run-name|-n)
            RUN_NAME="$2"
            echo "设置运行名称: $RUN_NAME"
            shift 2
            ;;
        --config|-c)
            CONFIG="$2"
            echo "设置配置文件: $CONFIG"
            shift 2
            ;;
        --dataset-dir|-d)
            DATASET_DIR="$2"
            echo "设置数据集目录: $DATASET_DIR"
            shift 2
            ;;
        --max-steps|-s)
            MAX_STEPS="$2"
            echo "设置最大步数: $MAX_STEPS"
            shift 2
            ;;
        --max-epochs|-e)
            MAX_EPOCHS="$2"
            echo "设置最大学习轮数: $MAX_EPOCHS"
            shift 2
            ;;
        --save-every|-S)
            SAVE_EVERY="$2"
            echo "设置保存间隔: $SAVE_EVERY"
            shift 2
            ;;
        --val-dataset-dir|--val-dir)
            VAL_DATASET_DIR="$2"
            echo "设置验证集目录: $VAL_DATASET_DIR"
            shift 2
            ;;
        --patience|-p)
            PATIENCE="$2"
            echo "设置早停耐心值: $PATIENCE"
            shift 2
            ;;
        --validation-interval|-v)
            VALIDATION_INTERVAL="$2"
            echo "设置验证间隔: $VALIDATION_INTERVAL"
            shift 2
            ;;
        --train-cfm)
            TRAIN_CFM=true
            echo "设置训练 CFM 模型"
            shift
            ;;
        --train-ar)
            TRAIN_AR=true
            echo "设置训练 AR 模型"
            shift
            ;;
        --fp16)
            FP16=true
            echo "启用 FP16"
            shift
            ;;
        -I|--interactive)
            INTERACTIVE_MODE=true
            shift
            ;;
        --distill)
            DISTILL=true
            echo "启用知识蒸馏"
            shift
            ;;        
        --distill-ar)
            DISTILL_AR=true
            echo "启用 AR 模型知识蒸馏"
            shift
            ;;
        --distill-cfm)
            DISTILL_CFM=true
            echo "启用 CFM 模型知识蒸馏"
            shift
            ;;
        --min-lr)
            MIN_LR="$2"
            echo "设置最小学习率: $MIN_LR"
            shift 2
            ;;
        --lr-adjust-interval)
            LR_ADJUST_INTERVAL="$2"
            echo "设置学习率调整日志打印间隔step: $LR_ADJUST_INTERVAL"
            shift 2
            ;;
        --initial-lr)
            INITIAL_LR="$2"
            echo "设置初始学习率: $INITIAL_LR"
            shift 2
            ;;
        --warmup-steps)
            WARMUP_STEPS="$2"
            echo "设置预热步数: $WARMUP_STEPS"
            shift 2
            ;;
        --resume-lr)
            RESUME_LR="$2"
            echo "设置恢复学习率: $RESUME_LR"
            shift 2
            ;;
        
        --language)
            LANGUAGE="$2"
            echo "设置语言参数: $LANGUAGE"
            shift 2
            ;;
        
        --pretrained-ckpt)
            PRETRAINED_CKPT="$2"
            echo "设置 V1 预训练检查点: $PRETRAINED_CKPT"
            shift 2
            ;;
        
        --pretrained-cfm-ckpt)
            PRETRAINED_CFM_CKPT="$2"
            echo "设置 V2 CFM 预训练检查点: $PRETRAINED_CFM_CKPT"
            shift 2
            ;;
        
        --pretrained-ar-ckpt)
            PRETRAINED_AR_CKPT="$2"
            echo "设置 V2 AR 预训练检查点: $PRETRAINED_AR_CKPT"
            shift 2
            ;;
        --log-file)
            LOG_FILE="$2"
            echo "设置日志文件: $LOG_FILE"
            shift 2
            ;;

        *)
            echo "未知参数: $1"
            echo "用法: $0 [--gpu|-G] [--v1|--v2] [--run-name|-n NAME] [--config|-c CONFIG_PATH] [--dataset-dir|-d DATASET_PATH] [--val-dataset-dir|--val-dir VAL_DATASET_PATH] [--max-steps|-s STEPS] [--max-epochs|-e EPOCHS] [--save-every|-S INTERVAL] [--patience|-p PATIENCE] [--validation-interval|-v INTERVAL] [--train-cfm] [--train-ar] [--distill] [--distill-ar] [--distill-cfm] [--min-lr MIN_LR] [--lr-adjust-interval LR_ADJUST_INTERVAL] [--initial-lr INITIAL_LR] [--warmup-steps WARMUP_STEPS] [--pretrained-ckpt CKPT_PATH] [--pretrained-cfm-ckpt CFM_CKPT_PATH] [--pretrained-ar-ckpt AR_CKPT_PATH]"
            echo "  --gpu|-G        使用 GPU 运行 (默认使用 CPU)"
            echo "  --v1            运行 V1 版本 (默认)"
            echo "  --v2            运行 V2 版本"
            echo "  --run-name|-n   设置运行名称 (默认: Test_ft)"
            echo "  --config|-c     设置配置文件路径"
            echo "  --dataset-dir|-d 设置数据集目录 (必填)"
            echo "  --val-dataset-dir|--val-dir 设置验证集目录"
            echo "  --max-steps|-s  设置最大步数 (默认: 1000)"
            echo "  --max-epochs|-e 设置最大学习轮数 (默认: 1000)"
            echo "  --save-every|-S 设置保存间隔 (默认: 100，建议与验证间隔一致)"
            echo "  --patience|-p   设置早停耐心值 (默认: 10，建议至少覆盖一个epoch)"
            echo "  --validation-interval|-v 设置验证间隔step (默认: 10，建议约验证样本数/batch_size)"
            echo "  --train-cfm     训练 CFM 模型 (仅 V2)"
            echo "  --train-ar      训练 AR 模型 (仅 V2)"
            echo "  --fp16          使用 FP16 精度 (默认: false)"
            echo "  --distill       启用知识蒸馏 (V1版本)"
            echo "  --distill-ar    启用 AR 模型知识蒸馏 (V2版本)"
            echo "  --distill-cfm   启用 CFM 模型知识蒸馏 (V2版本)"
            
            echo "  --min-lr        设置最小学习率 (默认: 1e-7)"
            echo "  --lr-adjust-interval 设置学习率调整间隔 (默认: 10)"
            echo "  --initial-lr    设置初始学习率 (默认: 4e-5)"
            echo "  --warmup-steps  设置预热步数 (默认: 50)"
            echo "  --resume-lr     设置恢复学习率 (默认: 0.0)"
            
            echo "  --pretrained-ckpt       V1 版本预训练检查点路径"
            echo "  --pretrained-cfm-ckpt   V2 版本 CFM 预训练检查点路径"
            echo "  --pretrained-ar-ckpt    V2 版本 AR 预训练检查点路径"
            
            echo "  --language              设置语言参数 (例如: zh, yue, en)"
            
            echo "  --log-file              设置日志文件路径 (默认: 不保存日志文件)"
            
            echo "  -I, --interactive 交互式选择参数"
            exit 1
            ;;
    esac
done

# 检查必填参数
if [[ -z "$DATASET_DIR" && "$INTERACTIVE_MODE" = false ]]; then
    echo "错误: 数据集目录 (--dataset-dir|-d) 是必填参数"
    echo "用法: $0 [--gpu|-G] [--v1|--v2] [--run-name|-n NAME] [--config|-c CONFIG_PATH] --dataset-dir|-d DATASET_PATH [--val-dataset-dir|--val-dir VAL_DATASET_PATH] [--max-steps|-s STEPS] [--max-epochs|-e EPOCHS] [--save-every|-S INTERVAL] [--patience|-p PATIENCE] [--validation-interval|-v INTERVAL] [--pretrained-ckpt CKPT_PATH] [--pretrained-cfm-ckpt CFM_CKPT_PATH] [--pretrained-ar-ckpt AR_CKPT_PATH]"
    exit 1
fi

# 如果启用了交互模式，则让用户选择参数
if [[ "$INTERACTIVE_MODE" = true ]]; then
    echo "=== 交互式参数选择 ==="
    
    # 询问版本
    echo "请选择版本:"
    echo "1) V1 版本 (默认)"
    echo "2) V2 版本"
    read -p "请输入选择 (1/2): " -n 1 -r
    echo
    if [[ $REPLY == "2" ]]; then
        VERSION="v2"
        echo "已选择 V2 版本"
    else
        VERSION="v1"
        echo "已选择 V1 版本"
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
    
    # 询问运行名称
    read -p "请输入运行名称 (默认: Test_ft): " run_name_input
    if [[ -n "$run_name_input" ]]; then
        RUN_NAME="$run_name_input"
    else
        RUN_NAME="Test_ft"
    fi
    echo "运行名称: $RUN_NAME"
    
    # 询问配置文件
    read -p "是否指定配置文件 (--config)？(y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "请选择配置文件:"
        SELECTED_CONFIG=$(select_any_file "请选择配置文件" "./configs")
        if [[ -n "$SELECTED_CONFIG" ]]; then
            CONFIG="$SELECTED_CONFIG"
            echo "已选择配置文件: $CONFIG"
        else
            echo "未指定配置文件"
        fi
    fi
    
    # 询问数据集目录（必填）
    while [[ -z "$SELECTED_DATASET_DIR" ]]; do
        echo "请选择数据集目录 (必填):"
        SELECTED_DATASET_DIR=$(select_directory "请选择数据集目录" "$DATASET_DIR")
        if [[ -n "$SELECTED_DATASET_DIR" ]]; then
            DATASET_DIR="$SELECTED_DATASET_DIR"
            echo "已选择数据集目录: $DATASET_DIR"
        else
            echo "数据集目录为必填项，请重新选择。"
        fi
    done
    
    # 询问最大步数
    read -p "请输入最大步数 (默认: 1000): " max_steps_input
    if [[ -n "$max_steps_input" ]]; then
        MAX_STEPS="$max_steps_input"
    else
        MAX_STEPS=1000
    fi
    echo "最大步数: $MAX_STEPS"
    
    # 询问最大学习轮数
    read -p "请输入最大学习轮数 (默认: 1000): " max_epochs_input
    if [[ -n "$max_epochs_input" ]]; then
        MAX_EPOCHS="$max_epochs_input"
    else
        MAX_EPOCHS=1000
    fi
    echo "最大学习轮数: $MAX_EPOCHS"
    
    # 询问保存间隔
    read -p "请输入保存间隔 (默认: 100，建议与验证间隔一致): " save_every_input
    if [[ -n "$save_every_input" ]]; then
        SAVE_EVERY="$save_every_input"
    else
        SAVE_EVERY=100
    fi
    echo "保存间隔: $SAVE_EVERY"
    
    # 询问验证集目录
    read -p "是否指定验证集目录 (--val-dataset-dir)？(y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "请选择验证集目录:"
        SELECTED_VAL_DATASET_DIR=$(select_directory "请选择验证集目录" "$VAL_DATASET_DIR")
        if [[ -n "$SELECTED_VAL_DATASET_DIR" ]]; then
            VAL_DATASET_DIR="$SELECTED_VAL_DATASET_DIR"
            echo "已选择验证集目录: $VAL_DATASET_DIR"
        else
            echo "未指定验证集目录"
        fi
    fi
    
    # 询问早停耐心值
    read -p "请输入早停耐心值 (默认: 10，建议至少覆盖一个epoch): " patience_input
    if [[ -n "$patience_input" ]]; then
        PATIENCE="$patience_input"
    else
        PATIENCE=10
    fi
    echo "早停耐心值: $PATIENCE"
    
    # 询问验证间隔
    read -p "请输入验证间隔step (默认: 10，建议约验证样本数/batch_size): " validation_interval_input
    if [[ -n "$validation_interval_input" ]]; then
        VALIDATION_INTERVAL="$validation_interval_input"
    else
        VALIDATION_INTERVAL=10
    fi
    echo "验证间隔: $VALIDATION_INTERVAL"
    
    # 询问是否使用 FP16
    read -p "是否使用 FP16 精度 (默认: false)？(y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        FP16=true
        echo "已启用 FP16 精度"
    else
        FP16=false
        echo "已禁用 FP16 精度"
    fi
    
    # 询问是否启用知识蒸馏（根据版本不同询问不同的参数）
    if [[ "$VERSION" = "v1" ]]; then
        read -p "是否启用知识蒸馏？(y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            DISTILL=true
            echo "已启用知识蒸馏"
        else
            DISTILL=false
            echo "已禁用知识蒸馏"
        fi
    else
        echo "是否启用知识蒸馏？(V2版本支持独立控制AR和CFM模型的蒸馏)"
        read -p "是否启用 AR 模型知识蒸馏？(y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            DISTILL_AR=true
            echo "已启用 AR 模型知识蒸馏"
        else
            DISTILL_AR=false
            echo "已禁用 AR 模型知识蒸馏"
        fi
        
        read -p "是否启用 CFM 模型知识蒸馏？(y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            DISTILL_CFM=true
            echo "已启用 CFM 模型知识蒸馏"
        else
            DISTILL_CFM=false
            echo "已禁用 CFM 模型知识蒸馏"
        fi
    fi
    
    # 询问预训练检查点路径
    echo ""
    echo "=== 预训练检查点 ==="
    read -p "是否指定预训练检查点？(y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [[ "$VERSION" = "v1" ]]; then
            echo "请选择 V1 预训练检查点文件:"
            SELECTED_PRETRAINED_CKPT=$(select_any_file "请选择预训练检查点文件" "./runs")
            if [[ -n "$SELECTED_PRETRAINED_CKPT" ]]; then
                PRETRAINED_CKPT="$SELECTED_PRETRAINED_CKPT"
                echo "已选择预训练检查点: $PRETRAINED_CKPT"
            else
                echo "未指定预训练检查点"
            fi
        else
            echo "请选择 V2 CFM 预训练检查点文件 (可选):"
            SELECTED_PRETRAINED_CFM_CKPT=$(select_any_file "请选择 CFM 预训练检查点文件" "./runs")
            if [[ -n "$SELECTED_PRETRAINED_CFM_CKPT" ]]; then
                PRETRAINED_CFM_CKPT="$SELECTED_PRETRAINED_CFM_CKPT"
                echo "已选择 CFM 预训练检查点: $PRETRAINED_CFM_CKPT"
            else
                echo "未指定 CFM 预训练检查点"
            fi
            
            echo "请选择 V2 AR 预训练检查点文件 (可选):"
            SELECTED_PRETRAINED_AR_CKPT=$(select_any_file "请选择 AR 预训练检查点文件" "./runs")
            if [[ -n "$SELECTED_PRETRAINED_AR_CKPT" ]]; then
                PRETRAINED_AR_CKPT="$SELECTED_PRETRAINED_AR_CKPT"
                echo "已选择 AR 预训练检查点: $PRETRAINED_AR_CKPT"
            else
                echo "未指定 AR 预训练检查点"
            fi
        fi
    else
        echo "将使用随机初始化训练"
    fi
    
    # 询问学习率相关参数
    echo ""
    echo "=== 学习率相关参数 ==="
    read -p "请输入最小学习率 (默认: 1e-7): " min_lr_input
    if [[ -n "$min_lr_input" ]]; then
        MIN_LR="$min_lr_input"
    else
        MIN_LR=1e-7
    fi
    echo "最小学习率: $MIN_LR"
    
    read -p "设置学习率调整日志打印间隔step (默认: 10): " lr_adjust_interval_input
    if [[ -n "$lr_adjust_interval_input" ]]; then
        LR_ADJUST_INTERVAL="$lr_adjust_interval_input"
    else
        LR_ADJUST_INTERVAL=10
    fi
    echo "学习率调整间隔: $LR_ADJUST_INTERVAL"
    
    read -p "请输入初始学习率 (默认: 4e-5): " initial_lr_input
    if [[ -n "$initial_lr_input" ]]; then
        INITIAL_LR="$initial_lr_input"
    else
        INITIAL_LR=4e-5
    fi
    echo "初始学习率: $INITIAL_LR"
    
    read -p "请输入预热步数 (默认: 50): " warmup_steps_input
    if [[ -n "$warmup_steps_input" ]]; then
        WARMUP_STEPS="$warmup_steps_input"
    else
        WARMUP_STEPS=50
    fi
    echo "预热步数: $WARMUP_STEPS"
    
    read -p "请输入恢复学习率 (默认: 0.0): " resume_lr_input
    if [[ -n "$resume_lr_input" ]]; then
        RESUME_LR="$resume_lr_input"
    else
        RESUME_LR=0.0
    fi
    echo "恢复学习率: $RESUME_LR"
    
    # 询问语言参数
    echo ""
    echo "=== 语言参数 ==="
    read -p "请输入语言参数 (例如: zh, yue, en，留空表示自动检测): " language_input
    if [[ -n "$language_input" ]]; then
        LANGUAGE="$language_input"
        echo "语言参数: $LANGUAGE"
    else
        echo "使用自动语言检测"
    fi
    
    # 询问日志文件路径
    echo ""
    echo "=== 日志文件 ==="
    read -p "请输入日志文件路径 (留空表示不保存日志文件): " log_file_input
    if [[ -n "$log_file_input" ]]; then
        LOG_FILE="$log_file_input"
        echo "日志文件: $LOG_FILE"
    else
        LOG_FILE=""
        echo "不保存日志文件"
    fi
    

    
    # 如果是V2版本，询问训练目标
    if [[ "$VERSION" = "v2" ]]; then
        echo ""
        echo "请选择要训练的模型:"
        echo "1) CFM 模型"
        echo "2) AR 模型"
        echo "3) CFM 和 AR 模型"
        read -p "请输入选项 (1/2/3): " choice
        
        TRAIN_CFM_ARG=""
        TRAIN_AR_ARG=""
        
        case $choice in
            1)
                TRAIN_CFM_ARG="--train-cfm"
                echo "您选择了训练 CFM 模型"
                ;;
            2)
                TRAIN_AR_ARG="--train-ar"
                echo "您选择了训练 AR 模型"
                ;;
            3)
                TRAIN_CFM_ARG="--train-cfm"
                TRAIN_AR_ARG="--train-ar"
                echo "您选择了同时训练 CFM 和 AR 模型"
                ;;
            *)
                TRAIN_CFM_ARG="--train-cfm"
                echo "无效选项，将默认训练 CFM 模型"
                ;;
        esac
        
    fi
    
    # 显示所有选择的参数并请求确认
    echo ""
    echo "=== 参数确认 ==="
    echo "版本: $VERSION"
    echo "运行名称: $RUN_NAME"

    # 显示CPU使用情况
    if [[ "$USE_CPU" = true ]]; then
        echo "使用CPU: 是"
    else
        echo "使用CPU: 否（将使用GPU，如果可用）"
    fi

    if [[ -n "$CONFIG" ]]; then
        echo "配置文件: $CONFIG"
    fi
    echo "数据集目录: $DATASET_DIR"
    echo "最大步数: $MAX_STEPS"
    echo "最大学习轮数: $MAX_EPOCHS"
    echo "保存间隔: $SAVE_EVERY"
    echo "使用FP16: $FP16"
    
    # 显示知识蒸馏参数（根据版本不同显示不同的参数）
    if [[ "$VERSION" = "v1" ]]; then
        if [[ "$DISTILL" = true ]]; then
            echo "知识蒸馏: 启用"
        else
            echo "知识蒸馏: 禁用"
        fi
    else
        if [[ "$DISTILL_AR" = true ]]; then
            echo "AR 模型知识蒸馏: 启用"
        else
            echo "AR 模型知识蒸馏: 禁用"
        fi
        if [[ "$DISTILL_CFM" = true ]]; then
            echo "CFM 模型知识蒸馏: 启用"
        else
            echo "CFM 模型知识蒸馏: 禁用"
        fi
    fi
    
    # 显示预训练检查点
    if [[ "$VERSION" = "v1" ]]; then
        if [[ -n "$PRETRAINED_CKPT" ]]; then
            echo "预训练检查点: $PRETRAINED_CKPT"
        fi
    else
        if [[ -n "$PRETRAINED_CFM_CKPT" ]]; then
            echo "CFM 预训练检查点: $PRETRAINED_CFM_CKPT"
        fi
        if [[ -n "$PRETRAINED_AR_CKPT" ]]; then
            echo "AR 预训练检查点: $PRETRAINED_AR_CKPT"
        fi
    fi
    

    
    if [[ -n "$VAL_DATASET_DIR" ]]; then
        echo "验证集目录: $VAL_DATASET_DIR"
    fi
    echo "早停耐心值: $PATIENCE"
    echo "验证间隔: $VALIDATION_INTERVAL"

    if [[ "$VERSION" = "v2" ]]; then
        if [[ -n "$TRAIN_CFM_ARG" ]] && [[ -n "$TRAIN_AR_ARG" ]]; then
            echo "训练目标: CFM 和 AR 模型"
        elif [[ -n "$TRAIN_CFM_ARG" ]]; then
            echo "训练目标: CFM 模型"
        elif [[ -n "$TRAIN_AR_ARG" ]]; then
            echo "训练目标: AR 模型"
        fi
    fi
    
    # 显示学习率相关参数
    echo "最小学习率: $MIN_LR"
    echo "学习率调整日志打印间隔step: $LR_ADJUST_INTERVAL"
    echo "初始学习率: $INITIAL_LR"
    echo "预热步数: $WARMUP_STEPS"
    echo "恢复学习率: $RESUME_LR"
    
    # 显示语言参数
    if [[ -n "$LANGUAGE" ]]; then
        echo "语言参数: $LANGUAGE"
    else
        echo "语言参数: 自动检测"
    fi
    
    # 显示日志文件
    if [[ -n "$LOG_FILE" ]]; then
        echo "日志文件: $LOG_FILE"
    else
        echo "日志文件: 不保存日志文件"
    fi
    

    
    # 生成等效的非交互式命令行命令
    echo ""
    echo "=== 等效的非交互式命令行 ==="
    CMD="./train.sh"

    CMD+=" --dataset-dir $DATASET_DIR"
    
    if [[ -n "$VAL_DATASET_DIR" ]]; then
        CMD+=" --val-dataset-dir $VAL_DATASET_DIR"
    fi
    
    # 添加预训练检查点参数
    if [[ "$VERSION" = "v1" ]]; then
        if [[ -n "$PRETRAINED_CKPT" ]]; then
            CMD+=" --pretrained-ckpt $PRETRAINED_CKPT"
        fi
    else
        if [[ -n "$PRETRAINED_CFM_CKPT" ]]; then
            CMD+=" --pretrained-cfm-ckpt $PRETRAINED_CFM_CKPT"
        fi
        if [[ -n "$PRETRAINED_AR_CKPT" ]]; then
            CMD+=" --pretrained-ar-ckpt $PRETRAINED_AR_CKPT"
        fi
    fi
    
    if [[ -n "$CONFIG" ]]; then
        CMD+=" --config \"$CONFIG\""
    fi
    
    # 添加日志文件参数
    if [[ -n "$LOG_FILE" ]]; then
        CMD+=" --log-file $LOG_FILE"
    fi
    
    CMD+=" --${VERSION}"
    
    if [[ "$RUN_NAME" != "Test_ft" ]]; then
        CMD+=" --run-name \"$RUN_NAME\""
    fi

    if [[ "$USE_CPU" = false ]]; then
        CMD+=" --gpu"
    fi
    
    if [[ "$FP16" = true ]]; then
        CMD+=" --fp16"
    fi
    
    # 添加语言参数
    if [[ -n "$LANGUAGE" ]]; then
        CMD+=" --language $LANGUAGE"
    fi
    
    # if [[ "$LR_ADJUST_INTERVAL" != "50" ]]; then
        CMD+=" --lr-adjust-interval $LR_ADJUST_INTERVAL"
    # fi
    
    # if [[ "$MAX_EPOCHS" != "1000" ]]; then
        CMD+=" --max-epochs $MAX_EPOCHS"
    # fi
    
    # 添加学习率相关参数
    # if [[ "$MIN_LR" != "1e-7" ]]; then
        CMD+=" --min-lr $MIN_LR"
    # fi
    
     # if [[ "$VALIDATION_INTERVAL" != "10" ]]; then
        CMD+=" --validation-interval $VALIDATION_INTERVAL"
    # fi
    
    # if [[ "$PATIENCE" != "10" ]]; then
        CMD+=" --patience $PATIENCE"
    # fi
    
    # if [[ "$SAVE_EVERY" != "100" ]]; then
        CMD+=" --save-every $SAVE_EVERY"
    # fi
    
    # if [[ "$WARMUP_STEPS" != "50" ]]; then
        CMD+=" --warmup-steps $WARMUP_STEPS"
    # fi
    
    # if [[ "$MAX_STEPS" != "1000" ]]; then
        CMD+=" --max-steps $MAX_STEPS"
    # fi
    
    # if [[ "$RESUME_LR" != "0.0" ]]; then
        CMD+=" --resume-lr $RESUME_LR"
    # fi
    
    # if [[ "$INITIAL_LR" != "4e-5" ]]; then
        CMD+=" --initial-lr $INITIAL_LR"
    # fi
    
    # 添加知识蒸馏参数（根据版本不同添加不同的参数）
    if [[ "$VERSION" = "v1" ]]; then
        if [[ "$DISTILL" = true ]]; then
            CMD+=" --distill"
        fi
    else
        if [[ "$DISTILL_AR" = true ]]; then
            CMD+=" --distill-ar"
        fi
        if [[ "$DISTILL_CFM" = true ]]; then
            CMD+=" --distill-cfm"
        fi
    fi
    
    if [[ "$VERSION" = "v2" ]]; then
        if [[ -n "$TRAIN_CFM_ARG" ]]; then
            CMD+=" $TRAIN_CFM_ARG"
        fi
        if [[ -n "$TRAIN_AR_ARG" ]]; then
            CMD+=" $TRAIN_AR_ARG"
        fi
    fi

    echo "$CMD"

    echo ""
    read -p "是否确认开始训练？(y/N/c): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        if [[ $REPLY =~ ^[Cc]$ ]]; then
            echo "已取消执行"
        else
            echo "已取消执行"
        fi
        exit 0
    fi
    echo "开始训练..."
fi

# 设置环境变量
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0
export PYTORCH_ENABLE_MPS_FALLBACK=1
export HF_HUB_OFFLINE=0

# 日志输出函数
run_with_logging() {
    # 默认始终在控制台显示日志
    # 只有当指定了日志文件路径且不为空时才同时保存到文件
    if [[ -n "$LOG_FILE" && "$LOG_FILE" != "" ]]; then
        # 使用 stdbuf 禁用缓冲来实现实时日志记录，过滤进度条
        stdbuf -oL -eL "$@" 2>&1 | tee >(awk '!/%\|/ {print $0; fflush()}' >> "$LOG_FILE")
    else
        # 只在控制台显示日志，同样禁用缓冲
        stdbuf -oL -eL "$@"
    fi
}

# export HUGGING_FACE_HUB_TOKEN={从https://huggingface.co/settings/tokens获取}
export HUGGING_FACE_HUB_TOKEN=

# 根据版本运行相应的应用程序
if [ "$VERSION" = "v1" ]; then
    if [[ "$USE_CPU" = true ]]; then
        export FORCE_CPU=1
    else
        export FORCE_CPU=0
    fi
    # 为V1版本构建配置参数
    CONFIG_PARAM=""
    if [ -n "$CONFIG" ]; then
        CONFIG_PARAM="--config $CONFIG"
    fi
    
    # 构建训练参数
    TRAIN_ARGS="$CONFIG_PARAM \
        --dataset-dir $DATASET_DIR \
        --run-name $RUN_NAME \
        --batch-size 8 \
        --max-steps $MAX_STEPS \
        --max-epochs $MAX_EPOCHS \
        --save-every $SAVE_EVERY \
        --num-workers 0 \
        --patience $PATIENCE \
        --validation-interval $VALIDATION_INTERVAL \
        --min-lr $MIN_LR \
        --lr-adjust-interval $LR_ADJUST_INTERVAL \
        --initial-lr $INITIAL_LR \
        --warmup-steps $WARMUP_STEPS \
        --resume-lr $RESUME_LR"
    
    # 添加验证集目录参数（如果提供了的话）
    if [[ -n "$VAL_DATASET_DIR" ]]; then
        TRAIN_ARGS+=" --val-dataset-dir $VAL_DATASET_DIR"
    fi
    
    # 只有在启用FP16时才添加--fp16参数
    if [[ "$FP16" = true ]]; then
        TRAIN_ARGS+=" --fp16"
    fi
    
    # 添加知识蒸馏参数（V1版本使用--distill）
    if [[ "$DISTILL" = true ]]; then
        TRAIN_ARGS+=" --distill"
    fi
    
    # 添加预训练检查点参数
    if [[ -n "$PRETRAINED_CKPT" ]]; then
        TRAIN_ARGS+=" --pretrained-ckpt $PRETRAINED_CKPT"
    fi
    
    # 添加语言参数
    if [[ -n "$LANGUAGE" ]]; then
        TRAIN_ARGS+=" --language $LANGUAGE"
    fi
    
    run_with_logging python train.py $TRAIN_ARGS
else
    
    # 构建V2的训练参数
    V2_TRAIN_ARGS=""
    # 默认使用标准的V2训练脚本
    V2_SCRIPT="train_v2.py"
    
    # 添加FP16支持
    if [[ "$FP16" = true ]]; then
        V2_TRAIN_ARGS+=" --fp16"
    fi
    
    # 添加验证集目录参数（如果提供了的话）
    if [[ -n "$VAL_DATASET_DIR" ]]; then
        V2_TRAIN_ARGS+=" --val-dataset-dir $VAL_DATASET_DIR"
    fi
    
    # 添加早停参数
    V2_TRAIN_ARGS+=" --patience $PATIENCE"
    V2_TRAIN_ARGS+=" --validation-interval $VALIDATION_INTERVAL"
    
    # 添加学习率相关参数
    
    V2_TRAIN_ARGS+=" --min-lr $MIN_LR"
    V2_TRAIN_ARGS+=" --lr-adjust-interval $LR_ADJUST_INTERVAL"
    V2_TRAIN_ARGS+=" --initial-lr $INITIAL_LR"
    V2_TRAIN_ARGS+=" --warmup-steps $WARMUP_STEPS"
    V2_TRAIN_ARGS+=" --resume-lr $RESUME_LR"
    
    if [[ "$TRAIN_CFM" = true ]] || [[ -n "$TRAIN_CFM_ARG" ]]; then
        V2_TRAIN_ARGS+=" --train-cfm"
    fi
    if [[ "$TRAIN_AR" = true ]] || [[ -n "$TRAIN_AR_ARG" ]]; then
        V2_TRAIN_ARGS+=" --train-ar"
    fi
    
    # 添加预训练检查点参数
    if [[ -n "$PRETRAINED_CFM_CKPT" ]]; then
        V2_TRAIN_ARGS+=" --pretrained-cfm-ckpt $PRETRAINED_CFM_CKPT"
    fi
    
    if [[ -n "$PRETRAINED_AR_CKPT" ]]; then
        V2_TRAIN_ARGS+=" --pretrained-ar-ckpt $PRETRAINED_AR_CKPT"
    fi
    
    # 添加知识蒸馏参数（V2版本使用--distill-ar和--distill-cfm）
    if [[ "$DISTILL_AR" = true ]]; then
        V2_TRAIN_ARGS+=" --distill-ar"
    fi
    if [[ "$DISTILL_CFM" = true ]]; then
        V2_TRAIN_ARGS+=" --distill-cfm"
    fi
    
    # 使用统一的训练脚本
    V2_SCRIPT="train_v2.py"
    
    # 为V2版本构建配置参数
    CONFIG_PARAM=""
    if [ -n "$CONFIG" ]; then
        CONFIG_PARAM="--config $CONFIG"
    fi

    if [[ "$USE_CPU" = true ]]; then
        # 强制使用cpu
        export FORCE_CPU=1
         # 直接使用Python运行，避免accelerate命令的问题
        run_with_logging python $V2_SCRIPT $CONFIG_PARAM \
            --dataset-dir $DATASET_DIR \
            --run-name $RUN_NAME \
            --batch-size 8 \
            --max-steps $MAX_STEPS \
            --max-epochs $MAX_EPOCHS \
            --save-every $SAVE_EVERY \
            --num-workers 1 \
            $V2_TRAIN_ARGS
    else
        export FORCE_CPU=0
        # 检查是否有Intel GPU可用
        if python -c "import torch; exit(0 if hasattr(torch, 'xpu') and torch.xpu.is_available() else 1)"; then
            # 使用Intel GPU运行
            run_with_logging accelerate launch $V2_SCRIPT $CONFIG_PARAM \
                --dataset-dir $DATASET_DIR \
                --run-name $RUN_NAME \
                --batch-size 8 \
                --max-steps $MAX_STEPS \
                --max-epochs $MAX_EPOCHS \
                --save-every $SAVE_EVERY \
                --num-workers 0 \
                $V2_TRAIN_ARGS
        else
            # 检查是否有NVIDIA GPU可用
            if nvidia-smi &> /dev/null; then
                # 使用NVIDIA GPU运行
                run_with_logging accelerate launch $V2_SCRIPT $CONFIG_PARAM \
                    --dataset-dir $DATASET_DIR \
                    --run-name $RUN_NAME \
                    --batch-size 8 \
                    --max-steps $MAX_STEPS \
                    --max-epochs $MAX_EPOCHS \
                    --save-every $SAVE_EVERY \
                    --num-workers 0 \
                    $V2_TRAIN_ARGS
            else
                # 回退到CPU运行
                export FORCE_CPU=1
                run_with_logging python $V2_SCRIPT $CONFIG_PARAM \
                    --dataset-dir $DATASET_DIR \
                    --run-name $RUN_NAME \
                    --batch-size 8 \
                    --max-steps $MAX_STEPS \
                    --max-epochs $MAX_EPOCHS \
                    --save-every $SAVE_EVERY \
                    --num-workers 1 \
                    $V2_TRAIN_ARGS
            fi
        fi
    fi
fi