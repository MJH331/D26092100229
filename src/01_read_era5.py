from pathlib import Path
import xarray as xr

# 项目根目录：scripts 文件夹的上一级
project_dir = Path(__file__).resolve().parent.parent

# ERA5 NetCDF 文件路径
#file_path = project_dir / "data" / "data_stream-oper_stepType-instant.nc"
#读日平均数据
file_path = project_dir / "data"/"processed" / "era5_t2m_daily_1940.nc"

# 读取数据
ds = xr.open_dataset(file_path)

print(ds)
print("\n数据变量：")
print(ds.data_vars)