from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


project_dir = Path(__file__).resolve().parent.parent

input_path = (
    project_dir
    / "outputs"
    / "srcnn"
    / "srcnn_test_predictions.npz"
)

output_dir = project_dir / "outputs" / "model_evaluation"
output_dir.mkdir(parents=True, exist_ok=True)

data = np.load(input_path)

hr = data["hr"]
bicubic = data["bicubic"]
srcnn = data["srcnn"]
valid_time = data["valid_time"]


def calculate_daily_metrics(prediction, target):
    error = prediction - target

    daily_mae = np.mean(np.abs(error), axis=(1, 2))
    daily_rmse = np.sqrt(np.mean(error ** 2, axis=(1, 2)))

    return daily_mae, daily_rmse


# 每个测试日的 MAE / RMSE
bicubic_mae, bicubic_rmse = calculate_daily_metrics(bicubic, hr)
srcnn_mae, srcnn_rmse = calculate_daily_metrics(srcnn, hr)

# 输出总体结果
print("测试样本数：", len(valid_time))
print(f"SRCNN 在 MAE 上优于 Bicubic 的天数：{np.sum(srcnn_mae < bicubic_mae)} / {len(valid_time)}")
print(f"SRCNN 在 RMSE 上优于 Bicubic 的天数：{np.sum(srcnn_rmse < bicubic_rmse)} / {len(valid_time)}")

print(f"\nBicubic 平均 MAE：{bicubic_mae.mean():.4f} °C")
print(f"SRCNN 平均 MAE：{srcnn_mae.mean():.4f} °C")

print(f"\nBicubic 平均 RMSE：{bicubic_rmse.mean():.4f} °C")
print(f"SRCNN 平均 RMSE：{srcnn_rmse.mean():.4f} °C")


# 保存每个测试日的指标表
csv_path = output_dir / "daily_test_metrics.csv"

with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)

    writer.writerow([
        "date",
        "bicubic_mae_celsius",
        "srcnn_mae_celsius",
        "bicubic_rmse_celsius",
        "srcnn_rmse_celsius",
        "mae_improvement_celsius",
        "rmse_improvement_celsius",
    ])

    for i in range(len(valid_time)):
        writer.writerow([
            str(valid_time[i])[:10],
            f"{bicubic_mae[i]:.6f}",
            f"{srcnn_mae[i]:.6f}",
            f"{bicubic_rmse[i]:.6f}",
            f"{srcnn_rmse[i]:.6f}",
            f"{bicubic_mae[i] - srcnn_mae[i]:.6f}",
            f"{bicubic_rmse[i] - srcnn_rmse[i]:.6f}",
        ])

print(f"\n逐日指标表已保存：{csv_path}")


# 绘制逐日 MAE 曲线
fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)

ax.plot(valid_time, bicubic_mae, label="Bicubic MAE", linewidth=2)
ax.plot(valid_time, srcnn_mae, label="SRCNN MAE", linewidth=2)

ax.set_title("Daily MAE on the test period")
ax.set_xlabel("Date")
ax.set_ylabel("MAE (°C)")
ax.legend()
ax.grid(alpha=0.3)

fig.savefig(
    output_dir / "daily_mae_comparison.png",
    dpi=300,
)
plt.close(fig)


# 计算每个空间格点上的平均绝对误差
bicubic_spatial_mae = np.mean(np.abs(bicubic - hr), axis=0)
srcnn_spatial_mae = np.mean(np.abs(srcnn - hr), axis=0)

# 正值代表 SRCNN 比 bicubic 更好
spatial_improvement = bicubic_spatial_mae - srcnn_spatial_mae

max_mae = max(bicubic_spatial_mae.max(), srcnn_spatial_mae.max())
max_improvement = np.max(np.abs(spatial_improvement))

fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)

img0 = axes[0].imshow(
    bicubic_spatial_mae,
    cmap="viridis",
    origin="upper",
    vmin=0,
    vmax=max_mae,
)
axes[0].set_title("Bicubic spatial MAE")
fig.colorbar(img0, ax=axes[0], label="MAE (°C)")

img1 = axes[1].imshow(
    srcnn_spatial_mae,
    cmap="viridis",
    origin="upper",
    vmin=0,
    vmax=max_mae,
)
axes[1].set_title("SRCNN spatial MAE")
fig.colorbar(img1, ax=axes[1], label="MAE (°C)")

img2 = axes[2].imshow(
    spatial_improvement,
    cmap="RdBu_r",
    origin="upper",
    vmin=-max_improvement,
    vmax=max_improvement,
)
axes[2].set_title("Improvement: Bicubic MAE − SRCNN MAE")
fig.colorbar(img2, ax=axes[2], label="Improvement (°C)")

for ax in axes:
    ax.set_xlabel("Grid x")
    ax.set_ylabel("Grid y")

fig.savefig(
    output_dir / "spatial_mae_comparison.png",
    dpi=300,
)
plt.close(fig)

print(f"逐日 MAE 曲线已保存：{output_dir / 'daily_mae_comparison.png'}")
print(f"空间误差图已保存：{output_dir / 'spatial_mae_comparison.png'}")