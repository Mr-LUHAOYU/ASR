from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from configs import Config
from dataset import DataSet

data = DataSet('SAVEE')
data.setValidSpeaker()

scaler = StandardScaler()

X, y = data.trainData()
X = X.to_numpy()
y = Config.encoder.transform(y.squeeze())
X = scaler.fit_transform(X)
X_train, y_train = X, y

X, y = data.validData()
X = X.to_numpy()
y = Config.encoder.transform(y.squeeze())
X = scaler.transform(X)
X_test, y_test = X, y

pipeline = make_pipeline(
    RobustScaler(),  # 鲁棒标准化
    PCA(n_components=200),  # 降维
    RandomForestClassifier()
)

pipeline.fit(X_train, y_train)
score = pipeline.score(X_train, y_train)
print('train', score)
score = pipeline.score(X_test, y_test)
print(f'Test', score)
