import os
import re


def find_log_file(directory=None):
    """在指定目录下查找log.txt文件"""
    if directory is None:
        directory = os.getcwd()

    # 首先查找log.txt文件
    log_txt_path = os.path.join(directory, "log.txt")
    if os.path.exists(log_txt_path):
        return log_txt_path

    # 如果没有找到log.txt，则查找.log文件
    for file in os.listdir(directory):
        if file.endswith(".log"):
            return os.path.join(directory, file)
    return None


def extract_f1_scores(log_file_path):
    f1_scores = []

    try:
        with open(log_file_path, "r", encoding="utf-8") as file:
            for line in file:
                # 匹配格式: [2025-06-29 02:42:20,918 INFO] f1: 0.2967
                match = re.search(r"f1:\s*([0-9.]+)", line)
                if match:
                    f1 = float(match.group(1))
                    f1_scores.append(f1)
    except FileNotFoundError:
        print(f"错误：找不到日志文件 {log_file_path}")
        return None, None, None
    except Exception as e:
        print(f"处理日志文件时出错: {e}")
        return None, None, None

    if not f1_scores:
        print("警告：未找到任何f1值")
        return None, None, None

    max_f1 = max(f1_scores)
    avg_f1 = sum(f1_scores) / len(f1_scores)

    print("\n分析结果：")
    print(f"找到的f1值数量: {len(f1_scores)}")
    print(f"最大f1值: {max_f1:.4f}")
    print(f"平均f1值: {avg_f1:.4f}")
    print(f"前10个f1值: {f1_scores[:10]}")
    print(f"后10个f1值: {f1_scores[-10:]}")

    return f1_scores, max_f1, avg_f1


if __name__ == "__main__":
    # 使用当前目录
    current_dir = os.getcwd()
    print(f"当前工作目录: {current_dir}")

    # 查找日志文件
    log_file = find_log_file(current_dir)

    if log_file:
        print(f"找到日志文件: {log_file}")
        f1_scores, max_f1, avg_f1 = extract_f1_scores(log_file)
    else:
        print("错误：在当前目录下未找到log.txt或.log文件")
