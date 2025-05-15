from pathlib import Path
from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
all_labels = ['a', 'd', 'f', 'h', 'n', 'sa', 'su']
label_encoder.fit(all_labels)


class Config:
    # 数据集参数
    basePath = Path('f:/ASR/大作业/features')
    encoder = label_encoder


