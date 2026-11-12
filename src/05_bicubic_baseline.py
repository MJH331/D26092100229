from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # 直接保存图片，避免图形窗口问题，要放在import matplotlib.pyplot前，否则失效

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


# =========================
# 1. 路径与数据读取
# =========================
project_dir = Path(__file__).resolve().parent.parent

pairs_path = (
    project_dir
    / "data"
    / "processed"
    / "era5_t2m_daily_1940_2x_pairs.npz"
)

output_dir = project_dir / "outputs" / "bicubic_baseline"
output_dir.mkdir(parents=True, exist_ok=True)

data = np.load(pairs_path)

hr = data["hr"]                 # (366, 96, 96)，单位：°C
lr = data["lr"]                 # (366, 48, 48)，单位：°C
valid_time = data["valid_time"]

print("HR 形状：", hr.shape)
print("LR 形状：", lr.shape)


# =========================
# 2. 按时间划分数据集
#    仅用于本次练手
# =========================
train_end = 256
val_end = 311

# 训练集：前 256 天
# 验证集：接下来 55 天
# 测试集：最后 55 天
test_hr = hr[val_end:]
test_lr = lr[val_end:]
test_time = valid_time[val_end:]

print(f"\n训练集样本数：{train_end}")
print(f"验证集样本数：{val_end - train_end}")
print(f"测试集样本数：{len(test_hr)}")
print(f"测试时间范围：{test_time[0]} 至 {test_time[-1]}")


# =========================
# 3. Bicubic 插值：48×48 → 96×96
# =========================
lr_tensor = torch.from_numpy(test_lr).unsqueeze(1)  # (N, 1, 48, 48)

with torch.no_grad():
    bicubic_tensor = F.interpolate(
        lr_tensor,
        size=(96, 96),
        mode="bicubic",
        align_corners=False,
    )

bicubic = bicubic_tensor.squeeze(1).numpy()  # (N, 96, 96)


# =========================
# 4. 在原始摄氏度单位下计算指标
# =========================
error = bicubic - test_hr

mae = np.mean(np.abs(error))
rmse = np.sqrt(np.mean(error ** 2))
bias = np.mean(error)
corr = np.corrcoef(bicubic.ravel(), test_hr.ravel())[0, 1]

print("\n===== Bicubic 测试集结果 =====")
print(f"MAE : {mae:.4f} °C")
print(f"RMSE: {rmse:.4f} °C")
print(f"Bias: {bias:.4f} °C")
print(f"相关系数: {corr:.4f}")


# 保存指标
metrics_path = output_dir / "bicubic_test_metrics.txt"

with metrics_path.open("w", encoding="utf-8") as f:
    f.write("Bicubic baseline on 1940 daily mean temperature\n")
    f.write(f"Test samples: {len(test_hr)}\n")
    f.write(f"MAE : {mae:.4f} °C\n")
    f.write(f"RMSE: {rmse:.4f} °C\n")
    f.write(f"Bias: {bias:.4f} °C\n")
    f.write(f"Correlation: {corr:.4f}\n")


# 保存预测结果，后续与 CNN 使用同一测试集比较
prediction_path = output_dir / "bicubic_test_predictions.npz"

np.savez_compressed(
    prediction_path,
    valid_time=test_time,
    hr=test_hr,
    lr=test_lr,
    bicubic=bicubic,
)


# =========================
# 5. 保存一个测试样本的对比图
# =========================
sample_index = 0

vmin = min(test_hr[sample_index].min(), bicubic[sample_index].min())
vmax = max(test_hr[sample_index].max(), bicubic[sample_index].max())

fig, axes = plt.subplots(2, 2, figsize=(12, 9))

img0 = axes[0, 0].imshow(
    test_lr[sample_index],
    cmap="coolwarm",
    origin="upper",
)
axes[0, 0].set_title("LR input: 0.5° (48×48)")
fig.colorbar(img0, ax=axes[0, 0], label="Temperature (°C)")

img1 = axes[0, 1].imshow(
    bicubic[sample_index],
    cmap="coolwarm",
    origin="upper",
    vmin=vmin,
    vmax=vmax,
)
axes[0, 1].set_title("Bicubic prediction: 0.25°")
fig.colorbar(img1, ax=axes[0, 1], label="Temperature (°C)")

img2 = axes[1, 0].imshow(
    test_hr[sample_index],
    cmap="coolwarm",
    origin="upper",
    vmin=vmin,
    vmax=vmax,
)
axes[1, 0].set_title("HR reference: 0.25°")
fig.colorbar(img2, ax=axes[1, 0], label="Temperature (°C)")

max_abs_error = np.max(np.abs(error[sample_index]))

img3 = axes[1, 1].imshow(
    error[sample_index],
    cmap="RdBu_r",#蓝色是正值，红色是负值
    origin="upper",
    vmin=-max_abs_error,
    vmax=max_abs_error,
)
axes[1, 1].set_title("Bicubic error: prediction − reference")
fig.colorbar(img3, ax=axes[1, 1], label="Error (°C)")

for ax in axes.ravel():
    ax.set_xlabel("Grid x")
    ax.set_ylabel("Grid y")

sample_date = str(test_time[sample_index])[:10]
fig.suptitle(f"Bicubic baseline comparison — {sample_date}", fontsize=14)
fig.tight_layout()

figure_path = output_dir / "bicubic_first_test_sample.png"
fig.savefig(figure_path, dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"\n指标已保存：{metrics_path}")
print(f"预测结果已保存：{prediction_path}")
print(f"对比图已保存：{figure_path}")