#!/usr/bin/env python3
"""
粤语拼音转换工具
用于将粤语文字转换为粤拼（Jyutping），方便编写 --yue-fix 文件

使用方法:
    python tools/yue_pinyin_converter.py "你好世界"
    python tools/yue_pinyin_converter.py --file input.txt --output output.txt
    python tools/yue_pinyin_converter.py --interactive

依赖安装:
    pip install pycantonese
"""

import argparse
import sys
import multiprocessing


def check_pycantonese():
    try:
        import pycantonese
        return pycantonese
    except ImportError:
        print("错误: 请先安装 pycantonese")
        print("运行: pip install pycantonese")
        sys.exit(1)


def get_opencc():
    """尝试加载简繁转换库"""
    try:
        from opencc import OpenCC
        return OpenCC('s2t')  # 简体转繁体
    except ImportError:
        return None


def split_jyutping(chars: str, jyutping: str) -> list:
    """
    将多字词的粤拼拆分为单字粤拼
    
    例如: "世界", "sai3gaai3" -> [("世", "sai3"), ("界", "gaai3")]
    
    Args:
        chars: 字符串（如 "世界"）
        jyutping: 对应的粤拼（如 "sai3gaai3"）
        
    Returns:
        list of tuples: [(字, 粤拼), ...]
    """
    import re
    
    if not jyutping:
        return [(c, None) for c in chars]
    
    # 用声调数字(1-6)作为分隔符拆分粤拼
    # 匹配模式: 辅音 + 声调数字
    syllables = re.findall(r'[a-z]+[1-6]', jyutping.lower())
    
    if len(syllables) == len(chars):
        return list(zip(chars, syllables))
    elif len(syllables) == 1 and len(chars) == 1:
        return [(chars, jyutping)]
    else:
        # 无法完美拆分，返回原始结果
        return [(chars, jyutping)]


def text_to_jyutping(text: str, with_tone: bool = True, split_words: bool = False) -> list:
    """
    将粤语文字转换为粤拼
    
    Args:
        text: 输入的粤语文字
        with_tone: 是否包含声调数字
        split_words: 是否将多字词拆分为单字
        
    Returns:
        list of tuples: [(字, 粤拼), ...]
    """
    pycantonese = check_pycantonese()
    s2t = get_opencc()
    
    def get_jyutping_single(char):
        """获取单字粤拼，支持简繁体"""
        # 先尝试直接查询
        jp_list = pycantonese.characters_to_jyutping(char)
        if jp_list and jp_list[0][1]:
            return jp_list[0][1]
        
        # 如果失败，尝试转繁体后查询
        if s2t:
            trad_char = s2t.convert(char)
            if trad_char != char:
                jp_list = pycantonese.characters_to_jyutping(trad_char)
                if jp_list and jp_list[0][1]:
                    return jp_list[0][1]
        
        return None
    
    # 先尝试整体转换
    raw_result = pycantonese.characters_to_jyutping(text)
    
    # 检查是否有未识别的字，尝试简繁转换
    result = []
    for chars, jp in raw_result:
        if jp:
            result.append((chars, jp))
        else:
            # 尝试逐字查询（支持简繁转换）
            for c in chars:
                if '\u4e00' <= c <= '\u9fff':
                    single_jp = get_jyutping_single(c)
                    result.append((c, single_jp))
                else:
                    result.append((c, None))
    
    if split_words:
        # 拆分多字词为单字
        expanded = []
        for chars, jp in result:
            if len(chars) > 1 and jp:
                split_pairs = split_jyutping(chars, jp)
                expanded.extend(split_pairs)
            else:
                expanded.append((chars, jp))
        result = expanded
    
    if not with_tone:
        # 移除声调数字
        result = [(char, jp.rstrip('123456') if jp else jp) for char, jp in result]
    
    return result


def format_output(jyutping_result: list, format_type: str = "inline") -> str:
    """
    格式化输出结果
    
    Args:
        jyutping_result: text_to_jyutping 的返回结果
        format_type: 输出格式
            - "inline": 行内格式 "nei5 hou2"
            - "table": 表格格式，每行一个字
            - "yuefix": --yue-fix 文件格式（需要后续手动添加时间戳）
    """
    if format_type == "inline":
        parts = []
        for char, jp in jyutping_result:
            if jp:
                parts.append(jp)
            else:
                parts.append(f"[{char}?]")  # 未识别的字符
        return " ".join(parts)
    
    elif format_type == "table":
        lines = []
        lines.append("字符\t粤拼")
        lines.append("-" * 20)
        for char, jp in jyutping_result:
            jp_display = jp if jp else "(未识别)"
            lines.append(f"{char}\t{jp_display}")
        return "\n".join(lines)
    
    elif format_type == "yuefix":
        lines = []
        lines.append("# yue-fix 格式模板")
        lines.append("# 格式: 粤拼 开始时间(秒) [结束时间(秒)]  # 原字")
        lines.append("# 只有添加了时间戳的行才会被处理，未添加时间戳的行会被跳过")
        lines.append("")
        
        for char, jp in jyutping_result:
            if jp:
                lines.append(f"{jp}  # {char}")
            else:
                lines.append(f"# [{char}] 未识别")
        return "\n".join(lines)
    
    else:
        raise ValueError(f"未知格式: {format_type}")


def process_text(text: str, format_type: str = "inline", with_tone: bool = True) -> str:
    """处理文本并返回格式化结果"""
    # yuefix 格式需要拆分多字词
    split_words = (format_type == "yuefix")
    result = text_to_jyutping(text, with_tone, split_words=split_words)
    return format_output(result, format_type)


def interactive_mode():
    """交互模式：引导式输入"""
    import os
    
    print("=" * 50)
    print("粤语拼音转换工具 - 交互模式")
    print("=" * 50)
    
    # 1. 输入文字
    print("\n请输入要转换的粤语文字（多行输入以空行结束）:")
    lines = []
    while True:
        try:
            line = input()
            if line == "":
                break
            lines.append(line)
        except (EOFError, KeyboardInterrupt):
            break
    
    text = "\n".join(lines).strip()
    if not text:
        print("已取消（未输入文字）")
        return
    
    print(f"\n输入文字: {text[:50]}{'...' if len(text) > 50 else ''}")
    
    # 2. 选择输出格式
    print("\n输出格式:")
    print("  1. inline  - 行内格式 (nei5 hou2)")
    print("  2. table   - 表格格式")
    print("  3. yuefix  - yue-fix 模板格式")
    
    format_choice = input("选择格式 [1/2/3] (默认: 3): ").strip()
    format_map = {"1": "inline", "2": "table", "3": "yuefix", "": "yuefix"}
    format_type = format_map.get(format_choice, "yuefix")
    print(f"已选择: {format_type}")
    
    # 3. 是否包含声调
    tone_choice = input("\n是否包含声调数字? [Y/n] (默认: Y): ").strip().lower()
    with_tone = tone_choice not in ['n', 'no', '否']
    
    # 4. 输出方式
    print("\n输出方式:")
    print("  1. 显示在屏幕")
    print("  2. 保存到文件")
    
    output_choice = input("选择输出方式 [1/2] (默认: 1): ").strip()
    
    output_path = None
    if output_choice == "2":
        output_path = input("输入输出文件路径 (默认: output.txt): ").strip()
        if not output_path:
            output_path = "output.txt"
    
    # 5. 确认执行
    print("\n" + "=" * 50)
    print("参数确认")
    print("=" * 50)
    print(f"输入文字: {text[:30]}{'...' if len(text) > 30 else ''}")
    print(f"输出格式: {format_type}")
    print(f"包含声调: {'是' if with_tone else '否'}")
    output_display = f"保存到 {output_path}" if output_path else "显示在屏幕"
    print(f"输出方式: {output_display}")
    print("=" * 50)
    
    confirm = input("\n是否执行？ [Y/n] (默认: Y): ").strip().lower()
    if confirm in ['n', 'no', '否', '取消']:
        print("已取消")
        return
    
    # 6. 执行转换
    print("\n正在转换...")
    try:
        output = process_text(text, format_type, with_tone)
        
        if output_path:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(output)
            print(f"\n已保存到: {output_path}")
        else:
            print("\n" + "-" * 50)
            print(output)
            print("-" * 50)
        
        print("\n转换完成!")
        
    except Exception as e:
        print(f"错误: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="粤语拼音转换工具 - 将粤语文字转换为粤拼(Jyutping)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s "你好"                      # 直接转换
  %(prog)s "你好" --format table       # 表格格式
  %(prog)s "你好" --format yuefix      # yue-fix文件格式
  %(prog)s --file input.txt            # 从文件读取
  %(prog)s --interactive               # 交互模式
        """
    )
    
    parser.add_argument("text", nargs="?", help="要转换的粤语文字")
    parser.add_argument("--file", "-f", help="从文件读取文字")
    parser.add_argument("--output", "-o", help="输出到文件")
    parser.add_argument("--format", "-F", 
                        choices=["inline", "table", "yuefix"],
                        default="inline",
                        help="输出格式 (default: inline)")
    parser.add_argument("--no-tone", action="store_true",
                        help="不包含声调数字")
    parser.add_argument("--interactive", "-i", action="store_true",
                        help="交互模式")
    
    args = parser.parse_args()
    
    # 交互模式
    if args.interactive:
        interactive_mode()
        return
    
    # 确定输入文本
    if args.file:
        try:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
        except FileNotFoundError:
            print(f"错误: 文件不存在 - {args.file}")
            sys.exit(1)
    elif args.text:
        text = args.text
    else:
        parser.print_help()
        sys.exit(1)
    
    # 处理并输出
    with_tone = not args.no_tone
    output = process_text(text, args.format, with_tone)
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"已保存到: {args.output}")
    else:
        print(output)


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
