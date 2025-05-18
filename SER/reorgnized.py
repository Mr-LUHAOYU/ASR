import os
import shutil
from pathlib import Path
import numpy as np


def reorganize_audio_files(data_dir="datasets/EmoDB", output_dir="datasets/reorganized"):
    # 确保输出目录存在
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    emotions = ['A', 'B', 'D', 'F', 'H', 'S', 'N']
    trans = {
        'W': 'a',
        'L': 'su',
        'E': 'd',
        'A': 'f',
        'F': 'h',
        'T': 'sa',
        'N': 'n',
    }

    # 获取所有.wav文件
    wav_files = list(Path(data_dir).glob("*.wav"))

    # 创建aa部分到字母的映射
    aa_parts = sorted({f.name[:2] for f in wav_files})
    aa_to_letter = {aa: chr(ord('A') + i) for i, aa in enumerate(aa_parts)}

    # 处理每个文件
    for wav_file in wav_files:
        filename = wav_file.name

        # 解析文件名各部分
        aa_part = filename[:2]  # 前两个数字
        bbb_part = filename[2:5]  # 字母+两个数字
        c_part = filename[5]  # 剩余部分(去掉.wav)
        c_part = trans[c_part]

        # 转换各部分
        x_dir = aa_to_letter[aa_part]  # 映射到字母
        b1 = bbb_part[0]
        b23 = bbb_part[1:]
        bb_part = (ord(b1) - ord('a') + 1) * 11 + int(b23)
        cc_part = c_part  # 保留原样

        # 创建目标目录
        target_dir = output_dir / x_dir
        target_dir.mkdir(exist_ok=True)

        # 构建新文件名
        new_filename = f"{cc_part}{bb_part}.wav"

        # 复制文件到新位置
        shutil.copy2(wav_file, target_dir / new_filename)

        print(f"Processed: {filename} -> {x_dir}/{new_filename}")

    print(f"\nAll files reorganized in: {output_dir}")
    print("Mapping of numeric prefixes to letters:")
    for num, letter in aa_to_letter.items():
        print(f"{num} -> {letter}")


if __name__ == "__main__":
    reorganize_audio_files()
