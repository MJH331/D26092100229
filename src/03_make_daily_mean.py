from pathlib import Path

import xarray as xr

# 项目目录
project_dir = Path(__file__).resolve().parent.parent

# 原始逐小时 ERA5 数据
input_path = project_dir / "data" / "data_stream-oper_stepType-instant.nc"

# 保存逐日平均结果的位置
processed_dir = project_dir / "data" / "processed"
processed_dir.mkdir(exist_ok=True)

output_path = processed_dir / "era5_t2m_daily_1940.nc"

# 读取原始数据
ds = xr.open_dataset(input_path)

# 开尔文转换为摄氏度
t2m_c = ds["t2m"] - 273.15
t2m_c.name = "t2m_daily_c"

# 按 UTC 自然日做平均：8784 小时 → 366 天
daily_t2m_c = t2m_c.resample(valid_time="1D").mean()

# 添加变量说明
daily_t2m_c.attrs["long_name"] = "Daily mean 2m air temperature"
daily_t2m_c.attrs["units"] = "degree_Celsius"

# 保存为新的 NetCDF 文件
daily_t2m_c.to_dataset().to_netcdf(output_path)

print("逐日平均数据已保存：")
print(output_path)
print("\n结果维度：")
print(daily_t2m_c.sizes)
print("\n温度范围（°C）：")
print(
    f"最低值：{float(daily_t2m_c.min()):.2f} °C，"
    f"最高值：{float(daily_t2m_c.max()):.2f} °C"
)