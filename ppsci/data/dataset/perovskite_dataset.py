import paddle
import pandas as pd
from pathlib import Path

class PerovskiteDataset(paddle.io.Dataset):
    def __init__(self, data_path):
        # 初始化数据集，加载数据并进行预处理
        self.data = pd.read_csv(data_path)
        self.features = self.data.drop(columns=["JV_default_Jsc"]).values
        self.labels = self.data["JV_default_Jsc"].values

    def __getitem__(self, index):
        # 返回数据和标签
        return self.features[index], self.labels[index]

    def __len__(self):
        # 返回数据集的长度
        return len(self.data)

def create_dataset():
    # 数据集创建的主要逻辑
    data_dir = Path.cwd() / "data" / "data" / "cleaned"
    output_file = data_dir / "cleaned_data.csv"
    # 在这里可以调用 create_data.py 的主要逻辑生成数据集
    main()
    return PerovskiteDataset(output_file)

if __name__ == "__main__":
    dataset = create_dataset()
    
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

def split_dataset(data_path, output_dir, datasplit=[0.8, 0.1, 0.1], seed=42):
    data = pd.read_csv(data_path)
    target = "JV_default_Jsc"  # 目标列
    features = data.drop(columns=[target])
    labels = data[target]
    
    X_train, X_temp, y_train, y_temp = train_test_split(features, labels, test_size=datasplit[1] + datasplit[2], random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=datasplit[2] / (datasplit[1] + datasplit[2]), random_state=seed)
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    X_train.to_csv(output_dir / "training.csv", index=False)
    y_train.to_csv(output_dir / "training_labels.csv", index=False)
    X_val.to_csv(output_dir / "validation.csv", index=False)
    y_val.to_csv(output_dir / "validation_labels.csv", index=False)
    X_test.to_csv(output_dir / "test.csv", index=False)
    y_test.to_csv(output_dir / "test_labels.csv", index=False)

if __name__ == "__main__":
    data_path = Path.cwd() / "data" / "data" / "cleaned" / "cleaned_data.csv"
    output_dir = Path.cwd() / "data" / "data" / "cleaned"
    split_dataset(data_path, output_dir)
