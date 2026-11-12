from pathlib import Path

import numpy as np
import xarray as xr

# 项目根目录
project_dir = Path(__file__).resolve().parent.parent

# 已生成的逐日平均温度数据
input_path = (
    project_dir
    / "data"
    / "processed"
    / "era5_t2m_daily_1940.nc"
)

# 输出：供后续 bicubic 和 CNN 使用的低/高分辨率数据对
output_path = (
    project_dir
    / "data"
    / "processed"
    / "era5_t2m_daily_1940_2x_pairs.npz"
)

# 读取逐日平均温度场：366 × 96 × 96，单位为 °C
ds = xr.open_dataset(input_path)
hr = ds["t2m_daily_c"]

# 通过 2×2 块平均，将 0.25° 的 96×96 温度场降采样为
# 0.5° 的 48×48 温度场
lr = hr.coarsen(latitude=2, longitude=2, boundary="trim").mean()

# 检查维度是否符合预期
print("高分辨率 HR 形状：", hr.shape)
print("低分辨率 LR 形状：", lr.shape)

assert hr.shape == (366, 96, 96), "HR 尺寸不是预期的 366×96×96"
assert lr.shape == (366, 48, 48), "LR 尺寸不是预期的 366×48×48"
assert not np.isnan(hr.values).any(), "HR 中发现缺失值"
assert not np.isnan(lr.values).any(), "LR 中发现缺失值"

# 保存为 NumPy 压缩文件。
# 这里暂不归一化；后续必须只利用训练集计算均值和标准差。
np.savez_compressed(
    output_path,
    hr=hr.values.astype(np.float32),
    lr=lr.values.astype(np.float32),
    valid_time=hr["valid_time"].values,
    hr_latitude=hr["latitude"].values,
    hr_longitude=hr["longitude"].values,
    lr_latitude=lr["latitude"].values,
    lr_longitude=lr["longitude"].values,
)

print("\n已保存 2× 超分辨率数据对：")
print(output_path)
print("HR 单位：°C；LR 由 HR 的 2×2 空间块平均得到。")