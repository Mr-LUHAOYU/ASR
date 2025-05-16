#!/bin/bash

# extract features
# python extract_features.py --n_mfcc 30

#!/bin/bash

# 提取ComParE特征的Shell脚本
# 用法：./extract_compare.sh datasets features dataname

# 参数检查
#if [ "$#" -ne 3 ]; then
#    echo "Usage: $0 <datasets_dir> <features_dir> <dataname>"
#    echo "Example: $0 datasets features CREMA-D"
#    exit 1
#fi

datasets_dir='../datasets'
features_dir='../features'
dataname='SAVEE'

# OpenSMILE配置文件路径（根据实际安装位置调整）
opensmile_config="opensmile\config\compare16\ComParE_2016.conf"

# 输入输出路径
input_root="$datasets_dir/$dataname"
output_root="$features_dir/$dataname"

# 检查输入目录是否存在
if [ ! -d "$input_root" ]; then
    echo "Error: Input directory $input_root does not exist!"
    exit 1
fi

# 创建输出目录
mkdir -p "$output_root"

# 计数器用于进度显示
total_files=0
processed_files=0

# 首先计算总文件数
echo "Counting WAV files..."
while IFS= read -r -d '' file; do
    ((total_files++))
done < <(find "$input_root" -type f -name "*.wav" -print0)

echo "Found $total_files WAV files to process."


# 处理每个WAV文件
while IFS= read -r -d '' input_file; do
    # 计算相对路径
    relative_path="${input_file#$input_root/}"
    relative_path="${relative_path%/*}"

    # 构建输出路径（将.wav改为.csv）
    output_file="$output_root/${relative_path}/${input_file##*/}"
    output_file="${output_file%.wav}.csv"

    # 创建输出目录
    mkdir -p "$(dirname "$output_file")"

    # 调用OpenSMILE提取特征
#    SMILExtract -C "$opensmile_config" -I "$input_file" -O "$output_file"
    "opensmile\bin\SMILExtract.exe" -C "$opensmile_config" -I "$input_file" -O "$output_file"

    # 进度显示
    ((processed_files++))
    echo -ne "Processing: $processed_files/$total_files ($((100*processed_files/total_files))%)"\\r
done < <(find "$input_root" -type f -name "*.wav" -print0)

echo -e "\nDone! Processed $processed_files files."
echo "Features saved to $output_root"
