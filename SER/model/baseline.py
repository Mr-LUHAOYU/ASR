from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from dataset import DataSet


def train8val(depth, mfcc, delta, DATASET: str, noise: str):
    dataset = DataSet(f'{DATASET}_{noise}_{mfcc}_{delta}', mfcc=mfcc * delta)
    X_train, y_train, _, _ = dataset.data('train', msg=False)
    X_val, y_val, _, _ = dataset.data('val', msg=False)
    X_test, y_test, _, _ = dataset.data('test', msg=False)

    # 训练模型
    rf = RandomForestClassifier(
        max_depth=depth,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # 在验证集上评估
    val_pred = rf.predict(X_val)
    val_score = accuracy_score(y_val, val_pred)

    test_pred = rf.predict(X_test)
    test_score = accuracy_score(y_test, test_pred)
    return val_score, test_score


# 定义参数组合
param_grid = {
    'max_depth': [None, 10, 20, 30],
    'mfcc': [13, 26, 39],
    'delta': [1, 2, 3]
}

# 手动网格搜索
for depth in param_grid['max_depth']:
    for mfcc in param_grid['mfcc']:
        for delta in param_grid['delta']:
            vs, ts = train8val(
                depth=depth, mfcc=mfcc, delta=delta,
                DATASET='SAVEE', noise='clean'
            )
            print(
                f'depth={depth}\tmfcc={mfcc}\tdelta={delta}\t'
                f'val_score={vs:.4f}\ttest_score={ts:.4f}'
            )
