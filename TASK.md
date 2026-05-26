# TASK.md

## 目标

修改 GAI_COMM 仓库中的 DM 训练与验证流程。

要求：

1. 使用 LEO Rural 数据验证 DM 性能：
   - 训练集：`dataset/LEO_Rural_seed1111.mat`
   - 验证集：`dataset/LEO_Rural_seed2222.mat`
   - 训练场景参数仍使用 `--train Rural`
   - 验证场景参数仍使用 `--test Rural`

2. 按论文可用强度训练：
   - 训练到 `200000` iterations
   - 保存并评估 `checkpoint_20.pth`
   - 不把本次任务改成完整 `600000` iterations 训练

3. 在以下 SNR 范围内进行 NMSE 测试：
   - SNR = `[-15, -10, -5, 0, 5, 10, 15, 20]`

4. 绘制：
   - SNR-NMSE 曲线
   - 尽量保持现有 DM 绘图风格
   - CSV 和最终打印结果中的 NMSE 使用线性值，不换算成 dB

5. 将测试结果保存为 CSV 格式：
   - 保存位置：`results/DM/test_train-Rural_test-Rural/results.csv`
   - 列包括：
     - `SNR`
     - `dm`

6. 最终打印结果格式类似：

```python
[
 ['SNR', 'DM'],
 [-15, ...],
 [-10, ...],
 [-5, ...],
 [0, ...],
 [5, ...],
 [10, ...],
 [15, ...],
 [20, ...]
]
```

## 训练修改

1. 修改 `train_diffusion_model.py`，新增命令行参数：
   - `--n_iters`：覆盖 `config.training.n_iters`
   - `--no_snapshot_sampling`：关闭训练过程中的 snapshot 条件采样，仅保留 checkpoint/eval_loss

2. 默认行为保持兼容：
   - 不传 `--n_iters` 时继续使用配置文件默认值
   - 不传 `--no_snapshot_sampling` 时继续沿用配置中的 `snapshot_sampling`

3. 本次训练命令：

```bash
/root/miniconda3/envs/myDM/bin/python train_diffusion_model.py \
  --gpu_id 0 \
  --train Rural \
  --workdir models/DM/Rural \
  --n_iters 200000 \
  --no_snapshot_sampling
```

## 验证修改

1. 修改 `test_diffusion_model.py`，新增命令行参数：
   - `--snr_values`，默认值为 `-15 -10 -5 0 5 10 15 20`
   - `--num_test_sample`，默认值为 `64`

2. 修正验证归一化：
   - 训练统计量来自 seed1111 训练集
   - seed2222 验证集使用训练集的 `mean/std`
   - 与训练阶段 `run_lib.py` 的验证逻辑保持一致

3. 新增 CSV 保存和最终结果打印：
   - 使用标准库 `csv`
   - CSV 保存线性 NMSE
   - 最终打印 `[['SNR', 'DM'], ...]`
   - 保留 `results.pt`
   - 保存线性 NMSE 曲线为 `results_mse.png`
   - 保留原有 dB 曲线 `results.png` 以兼容已有结果查看流程

4. 本次验证命令：

```bash
/root/miniconda3/envs/myDM/bin/python test_diffusion_model.py \
  --gpu_id 0 \
  --train Rural \
  --test Rural \
  --model_pth checkpoint_20.pth \
  --snr_values -15 -10 -5 0 5 10 15 20
```

## 验收标准

1. 训练完成后存在：
   - `models/DM/Rural/checkpoints/checkpoint_20.pth`

2. 验证完成后存在：
   - `results/DM/test_train-Rural_test-Rural/results.pt`
   - `results/DM/test_train-Rural_test-Rural/results.csv`
   - `results/DM/test_train-Rural_test-Rural/results_mse.png`

3. `results.csv` 内容满足：
   - 第一行为 `SNR,dm`
   - 后续 8 行对应 `[-15, -10, -5, 0, 5, 10, 15, 20]`
   - `dm` 列为线性 NMSE

4. 控制台最终打印格式满足：
   - 第一行为 `['SNR', 'DM']`
   - 后续每行包含一个 SNR 和对应线性 NMSE

## 假设

1. 使用当前可用环境 `/root/miniconda3/envs/myDM/bin/python`。
2. 使用 GPU `0`，当前机器检测到 NVIDIA GeForce RTX 3090。
3. 不新增单独测试脚本，不改动 ML/WGAN baseline。
4. 不改变 LEO 数据文件命名规则和 `loaders.py` 的数据解析入口。
