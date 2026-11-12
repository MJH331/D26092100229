from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


# =========================
# 1. 固定随机性与路径
# =========================
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

project_dir = Path(__file__).resolve().parent.parent

pairs_path = (
    project_dir
    / "data"
    / "processed"
    / "era5_t2m_daily_1940_2x_pairs.npz"
)

output_dir = project_dir / "outputs" / "srcnn"
output_dir.mkdir(parents=True, exist_ok=True)

checkpoint_path = output_dir / "best_srcnn_model.pt"


# =========================
# 2. 读取并划分数据
# =========================
data = np.load(pairs_path)

hr = data["hr"].astype(np.float32)       # (366, 96, 96)，单位：°C
lr = data["lr"].astype(np.float32)       # (366, 48, 48)，单位：°C
valid_time = data["valid_time"]

train_end = 256
val_end = 311

train_hr = hr[:train_end]
val_hr = hr[train_end:val_end]
test_hr = hr[val_end:]

train_lr = lr[:train_end]
val_lr = lr[train_end:val_end]
test_lr = lr[val_end:]

test_time = valid_time[val_end:]


# =========================
# 3. Bicubic 上采样至 96×96
# =========================
def bicubic_upscale(lr_array):
    lr_tensor = torch.from_numpy(
        np.ascontiguousarray(lr_array)
    ).unsqueeze(1)

    with torch.no_grad():
        output = F.interpolate(
            lr_tensor,
            size=(96, 96),
            mode="bicubic",
            align_corners=False,
        )

    return output.squeeze(1).numpy()


bicubic_all = bicubic_upscale(lr)


# =========================
# 4. 只用训练集计算归一化参数
# =========================
train_mean = float(train_hr.mean())
train_std = float(train_hr.std())

print(f"训练集均值：{train_mean:.4f} °C")
print(f"训练集标准差：{train_std:.4f} °C")

def normalize(array):
    return (array - train_mean) / train_std


train_x = normalize(bicubic_all[:train_end])
val_x = normalize(bicubic_all[train_end:val_end])
test_x = normalize(bicubic_all[val_end:])

train_y = normalize(train_hr)
val_y = normalize(val_hr)
test_y = normalize(test_hr)


# =========================
# 5. 制作 PyTorch 数据集
# =========================
def to_tensor(array):
    array = np.ascontiguousarray(array[:, np.newaxis, :, :])
    return torch.from_numpy(array)


train_dataset = TensorDataset(to_tensor(train_x), to_tensor(train_y))

train_loader = DataLoader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    num_workers=0,   # Windows 下保持 0 更稳定
)


# =========================
# 6. 轻量残差式 SRCNN
# =========================
class TemperatureSRCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(8, 1, kernel_size=3, padding=1),
        )

    def forward(self, x):
        # 网络只学习 bicubic 与真实高分辨率场之间的残差
        return x + self.cnn(x)


device = torch.device("cpu")
model = TemperatureSRCNN().to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

print(f"\n模型参数量：{sum(p.numel() for p in model.parameters()):,}")
print(f"训练设备：{device}")


# =========================
# 7. 训练并按验证集损失保存最佳模型
# =========================
max_epochs = 50
patience = 10

best_val_loss = float("inf")
epochs_without_improvement = 0

train_losses = []
val_losses = []

val_x_tensor = to_tensor(val_x).to(device)
val_y_tensor = to_tensor(val_y).to(device)

for epoch in range(1, max_epochs + 1):
    model.train()
    total_train_loss = 0.0

    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()

        predictions = model(inputs)
        loss = criterion(predictions, targets)

        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * inputs.size(0)

    train_loss = total_train_loss / len(train_dataset)

    model.eval()
    with torch.no_grad():
        val_predictions = model(val_x_tensor)
        val_loss = criterion(val_predictions, val_y_tensor).item()

    train_losses.append(train_loss)
    val_losses.append(val_loss)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), checkpoint_path)
    else:
        epochs_without_improvement += 1

    if epoch == 1 or epoch % 5 == 0:
        print(
            f"Epoch {epoch:02d}/{max_epochs} | "
            f"训练损失: {train_loss:.6f} | "
            f"验证损失: {val_loss:.6f}"
        )

    if epochs_without_improvement >= patience:
        print(f"\n验证集连续 {patience} 轮未提升，提前停止训练。")
        break


# =========================
# 8. 加载最佳模型并测试
# =========================
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()

test_x_tensor = to_tensor(test_x).to(device)

with torch.no_grad():
    cnn_prediction_norm = model(test_x_tensor).cpu().squeeze(1).numpy()

# 反归一化，恢复为 °C
cnn_prediction = cnn_prediction_norm * train_std + train_mean
bicubic_prediction = bicubic_all[val_end:]


def calculate_metrics(prediction, target):
    error = prediction - target

    mae = np.mean(np.abs(error))
    rmse = np.sqrt(np.mean(error ** 2))
    bias = np.mean(error)
    corr = np.corrcoef(prediction.ravel(), target.ravel())[0, 1]

    return mae, rmse, bias, corr


bicubic_mae, bicubic_rmse, bicubic_bias, bicubic_corr = calculate_metrics(
    bicubic_prediction,
    test_hr,
)

cnn_mae, cnn_rmse, cnn_bias, cnn_corr = calculate_metrics(
    cnn_prediction,
    test_hr,
)

print("\n===== 测试集结果（单位：°C）=====")
print(
    f"Bicubic | MAE: {bicubic_mae:.4f} | RMSE: {bicubic_rmse:.4f} | "
    f"Bias: {bicubic_bias:.4f} | Corr: {bicubic_corr:.4f}"
)
print(
    f"SRCNN   | MAE: {cnn_mae:.4f} | RMSE: {cnn_rmse:.4f} | "
    f"Bias: {cnn_bias:.4f} | Corr: {cnn_corr:.4f}"
)

mae_improvement = (bicubic_mae - cnn_mae) / bicubic_mae * 100
rmse_improvement = (bicubic_rmse - cnn_rmse) / bicubic_rmse * 100

print(f"\n相对 Bicubic 的 MAE 改善：{mae_improvement:.2f}%")
print(f"相对 Bicubic 的 RMSE 改善：{rmse_improvement:.2f}%")


# =========================
# 9. 保存指标、预测和训练曲线
# =========================
metrics_path = output_dir / "srcnn_test_metrics.txt"

with metrics_path.open("w", encoding="utf-8") as f:
    f.write("SRCNN test results on 1940 daily mean temperature\n\n")
    f.write(
        f"Bicubic | MAE={bicubic_mae:.4f}, RMSE={bicubic_rmse:.4f}, "
        f"Bias={bicubic_bias:.4f}, Corr={bicubic_corr:.4f}\n"
    )
    f.write(
        f"SRCNN   | MAE={cnn_mae:.4f}, RMSE={cnn_rmse:.4f}, "
        f"Bias={cnn_bias:.4f}, Corr={cnn_corr:.4f}\n"
    )
    f.write(f"MAE improvement (%): {mae_improvement:.2f}\n")
    f.write(f"RMSE improvement (%): {rmse_improvement:.2f}\n")

np.savez_compressed(
    output_dir / "srcnn_test_predictions.npz",
    valid_time=test_time,
    hr=test_hr,
    bicubic=bicubic_prediction,
    srcnn=cnn_prediction,
)

np.savez(
    output_dir / "normalization_parameters.npz",
    mean=train_mean,
    std=train_std,
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(train_losses, label="Training loss")
ax.plot(val_losses, label="Validation loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE loss (normalized temperature)")
ax.set_title("SRCNN training curve")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(output_dir / "srcnn_training_curve.png", dpi=300)
plt.close(fig)


# =========================
# 10. 保存一个测试样本对比图
# =========================
sample_index = 0

vmin = min(
    test_hr[sample_index].min(),
    bicubic_prediction[sample_index].min(),
    cnn_prediction[sample_index].min(),
)

vmax = max(
    test_hr[sample_index].max(),
    bicubic_prediction[sample_index].max(),
    cnn_prediction[sample_index].max(),
)

cnn_error = cnn_prediction[sample_index] - test_hr[sample_index]
max_abs_error = np.max(np.abs(cnn_error))

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

img0 = axes[0, 0].imshow(
    bicubic_prediction[sample_index],
    cmap="coolwarm",
    origin="upper",
    vmin=vmin,
    vmax=vmax,
)
axes[0, 0].set_title("Bicubic prediction")

img1 = axes[0, 1].imshow(
    cnn_prediction[sample_index],
    cmap="coolwarm",
    origin="upper",
    vmin=vmin,
    vmax=vmax,
)
axes[0, 1].set_title("SRCNN prediction")

img2 = axes[1, 0].imshow(
    test_hr[sample_index],
    cmap="coolwarm",
    origin="upper",
    vmin=vmin,
    vmax=vmax,
)
axes[1, 0].set_title("HR reference")

img3 = axes[1, 1].imshow(
    cnn_error,
    cmap="RdBu_r",
    origin="upper",
    vmin=-max_abs_error,
    vmax=max_abs_error,
)
axes[1, 1].set_title("SRCNN error: prediction − reference")

for ax in axes.ravel():
    ax.set_xlabel("Grid x")
    ax.set_ylabel("Grid y")

fig.colorbar(img1, ax=axes[:2, :].ravel().tolist(), label="Temperature / Error (°C)")
sample_date = str(test_time[sample_index])[:10]
fig.suptitle(f"SRCNN comparison — {sample_date}", fontsize=14)
fig.tight_layout()

fig.savefig(
    output_dir / "srcnn_first_test_sample.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

print(f"\n最佳模型：{checkpoint_path}")
print(f"训练曲线：{output_dir / 'srcnn_training_curve.png'}")
print(f"测试指标：{metrics_path}")
print(f"测试对比图：{output_dir / 'srcnn_first_test_sample.png'}")