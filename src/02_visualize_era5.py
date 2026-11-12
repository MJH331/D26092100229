from pathlib import Path

import matplotlib.pyplot as plt
import xarray as xr

# 项目根目录
project_dir = Path(__file__).resolve().parent.parent

# 输入与输出路径
file_path = project_dir / "data" / "data_stream-oper_stepType-instant.nc"
output_dir = project_dir / "outputs"
output_dir.mkdir(exist_ok=True)

# 读取 ERA5 数据
ds = xr.open_dataset(file_path)

# t2m 单位为 K，转换为摄氏度
t2m_c = ds["t2m"] - 273.15

# 选取第一个时刻：1940-01-01 00:00
field = t2m_c.isel(valid_time=0)
sample_time = field["valid_time"].values

# 绘制温度空间分布图
fig, ax = plt.subplots(figsize=(9, 6))#准备画布

field.plot(
    ax=ax,
    cmap="coolwarm",
    cbar_kwargs={"label": "Temperature (°C)"}
)

ax.set_title(f"ERA5 2m Temperature\n{sample_time}")
ax.set_xlabel("Longitude")
ax.set_ylabel("Latitude")

# 保存图片
output_path = output_dir / "t2m_19400101_0000.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print(f"图片已保存至：{output_path}")