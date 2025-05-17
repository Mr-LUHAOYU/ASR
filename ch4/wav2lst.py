import os

def find_wav_files(directory):
    """递归查找目录下所有 .wav 文件的绝对路径"""
    wav_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith(".wav"):
                wav_files.append(os.path.abspath(os.path.join(root, file)))
    return wav_files

def write_to_file(file_list, output_file="train.lst"):
    """将文件列表写入 train.lst"""
    with open(output_file, "w", encoding="utf-8") as f:
        for file_path in file_list:
            f.write(file_path + "\n")

if __name__ == "__main__":
    # 用户输入要遍历的目录
    search_dir = input("请输入要遍历的目录路径（例如：C:\\audio 或 ./data）: ").strip()
    output_file = input('请输入目标文件').strip()

    # 检查目录是否存在
    if not os.path.isdir(search_dir):
        print("错误：目录不存在！")
        exit(1)

    # 查找所有 .wav 文件
    wav_files = find_wav_files(search_dir)

    if not wav_files:
        print("未找到 .wav 文件！")
        exit(1)

    # 写入 train.lst
    write_to_file(wav_files, output_file)
    print(f"已完成：共找到 {len(wav_files)} 个 .wav 文件，路径已写入 {output_file}")