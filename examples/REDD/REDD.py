import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

matplotlib.rcParams["font.family"] = "SimHei"
matplotlib.rcParams["axes.unicode_minus"] = False

# ==== Parameter Configuration (Customizable) ====
EPOCHS = 500
LR = 0.01
CLASS_WEIGHTS = [3.5, 3.5, 2.0, 2.5]  # Class order: Low, Mid-Low, Mid-High, High

# ==== Classification Label Function ====
def classify_emission(value):
    if value < 1500:
        return 0
    elif value < 7800:
        return 1
    elif value < 40000:
        return 2
    else:
        return 3


# ==== FocalLoss Definition ====
class FocalLoss(nn.Layer):
    def __init__(self, gamma=2, weight=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        logpt = F.cross_entropy(input, target, weight=self.weight, reduction="none")
        pt = paddle.exp(-logpt)
        loss = ((1 - pt) ** self.gamma) * logpt
        return loss.mean()


# ==== Load and Clean Data ====
df = pd.read_excel("./Fusion_Data.xlsx")
df = df.dropna(
    subset=[
        "企业CO₂排放量 (kg)",
        "匹配时间",
        "企业省份",
        "卫星中心纬度",
        "卫星中心经度",
        "卫星CO₂浓度 (xco2)",
        "风向",
        "风速",
    ]
)
df["匹配时间"] = pd.to_datetime(df["匹配时间"])
df["hour"] = df["匹配时间"].dt.hour
df["month"] = df["匹配时间"].dt.month

# ==== Feature Processing ====
numeric_features = ["卫星中心纬度", "卫星中心经度", "卫星CO₂浓度 (xco2)", "风向", "风速", "hour", "month"]
categorical_features = ["企业省份"]
X_raw = df[numeric_features + categorical_features]
y_raw = df["企业CO₂排放量 (kg)"].values.reshape(-1, 1)
labels = np.vectorize(classify_emission)(y_raw.flatten())
enterprise_names = df["企业名称"].values

ct = ColumnTransformer(
    [
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)
X = ct.fit_transform(X_raw)

# ==== Stratified Sampling ====
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in sss.split(X, labels):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = labels[train_idx], labels[test_idx]
    name_train, name_test = enterprise_names[train_idx], enterprise_names[test_idx]

X_train = paddle.to_tensor(X_train, dtype="float32")
y_train = paddle.to_tensor(y_train, dtype="int64")
X_test = paddle.to_tensor(X_test, dtype="float32")
y_test = paddle.to_tensor(y_test, dtype="int64")

# ==== Network Architecture ====
class EmissionClassifier(nn.Layer):
    def __init__(self, input_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 4))

    def forward(self, x):
        x = self.shared(x)
        return self.classifier(x)


# ==== Model Training ====
model = EmissionClassifier(input_dim=X.shape[1])
optimizer = paddle.optimizer.Adam(parameters=model.parameters(), learning_rate=LR)
loss_fn = FocalLoss(gamma=2, weight=paddle.to_tensor(CLASS_WEIGHTS, dtype="float32"))

train_loss_record = []
val_acc_record = []

for epoch in range(EPOCHS):
    model.train()
    logits = model(X_train)
    loss = loss_fn(logits, y_train)
    loss.backward()
    optimizer.step()
    optimizer.clear_grad()
    train_loss_record.append(loss.numpy())

    if (epoch + 1) % 20 == 0:
        model.eval()
        with paddle.no_grad():
            val_logits = model(X_test)
            preds = paddle.argmax(val_logits, axis=1)
            acc = accuracy_score(y_test.numpy(), preds.numpy())
            val_acc_record.append(acc)
            print(f"[Epoch {epoch+1}] loss={loss.numpy():.4f}, acc={acc:.4f}")

# ==== Model Evaluation ====
model.eval()
X_all_tensor = paddle.to_tensor(X, dtype="float32")
with paddle.no_grad():
    preds = paddle.argmax(model(X_all_tensor), axis=1).numpy()

print("\n🎯 Overall Accuracy: {:.2f}%".format(accuracy_score(labels, preds) * 100))
print("\n📊 Classification Report:")
report = classification_report(
    labels, preds, target_names=["Low", "Mid-Low", "Mid-High", "High"], output_dict=True
)
print(
    classification_report(
        labels, preds, target_names=["Low", "Mid-Low", "Mid-High", "High"]
    )
)

# ==== 📈 Training Loss Curve ====
plt.figure()
plt.plot(train_loss_record, label="Training Loss")
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# ==== 📊 Recall per Class Bar Chart ====
plt.figure()
target_names = ["Low", "Mid-Low", "Mid-High", "High"]
recalls = [report[name]["recall"] for name in target_names]
plt.bar(target_names, recalls)
plt.title("Recall per Class")
plt.ylabel("Recall")
plt.ylim(0, 1)
plt.grid(axis="y")
plt.tight_layout()
plt.show()

# ==== Confusion Matrix ====
cm = confusion_matrix(labels, preds)
ConfusionMatrixDisplay(
    confusion_matrix=cm, display_labels=["Low", "Mid-Low", "Mid-High", "High"]
).plot(cmap="Blues")
plt.title("Predicted vs Actual Class")
plt.tight_layout()
plt.show()

# ==== Export Results ====
pd.DataFrame(
    {
        "Enterprise Name": enterprise_names,
        "Actual Class": labels,
        "Predicted Class": preds,
    }
).to_csv("carbon_emission_prediction_results.csv", index=False)
print("✅ Results exported to: carbon_emission_prediction_results.csv")
