## 1. 项目简介
本项目针对 ERA5 气温数据 0.25° → 0.5° 降分辨率重建任务，
实现双三次插值基线与两级残差卷积网络（doubleCNN）对比实验。

## 2. 环境配置
- Python 3.10，CUDA 可选
- 安装依赖：pip install -r requirements.txt

## 3. 数据
- 数据来源：ERA5 单层日平均气温，0.25°
- 区域：北纬 16.25°–40°，东经 100°–123.75°
- 时间：1940-01-01 至 1940-12-31
- 测试集由原始 0.25° 降采样到 0.5°，再用双三次插值上采样回 0.25° 作为输入。

## 4. 运行步骤
1. python src/prepare_data.py
2. python src/bicubic_baseline.py
3. python src/train.py
4. python src/evaluate.py
5. python src/plot_results.py

## 5. 主要结果
- 双三次插值 MAE：约 0.23–0.27℃
- doubleCNN MAE：约 0.20–0.25℃
- 结果图见 results/

## 6. 论文
见 paper.pdf