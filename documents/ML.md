# ML 信道估计说明

本文档概述当前项目中 Maximum Likelihood baseline 的信道估计流程。主要实现入口为 `baseline_utils/test_ml.py`。

## 1. 基本思想

ML baseline 不训练生成模型，而是在每个 SNR 下直接根据导频观测构造线性最小二乘问题，求解信道矩阵。

当前代码使用 Hermitian 信道：

$$
\mathbf H_h = \mathbf H^H \in \mathbb C^{N_t \times N_r}
$$

导频矩阵为：

$$
\mathbf P \in \mathbb C^{N_t \times N_p}
$$

接收导频信号为：

$$
\mathbf Y = \mathbf P^H \mathbf H_h + \mathbf N
$$

其中：

$$
\mathbf Y \in \mathbb C^{N_p \times N_r}
$$

## 2. 估计过程

运行入口：

```bash
python baseline_utils/test_ml.py --gpu 0 --train Rural --test Rural --pilot_alpha 0.6
```

主要流程：

1. 读取训练场景数据，得到训练集归一化统计量。
2. 使用相同统计量归一化测试场景信道。
3. 从 `Channels` 中取得 QPSK 导频矩阵 `P` 和真实信道 `H_herm`。
4. 对每个 SNR 构造带噪声接收信号 `Y`。
5. 对每个样本求解正规方程，得到信道估计。
6. 保存不同 SNR 下的 NMSE 结果。

代码中的求解形式为：

$$
\hat{\mathbf H}_h
=
\arg\min_{\mathbf H}
\left\|\mathbf Y-\mathbf P^H\mathbf H\right\|_F^2
+ \sigma_n^2\|\mathbf H\|_F^2
$$

对应正规方程：

$$
(\mathbf P\mathbf P^H+\sigma_n^2\mathbf I)\hat{\mathbf H}_h
=
\mathbf P\mathbf Y
$$

在 `test_ml.py` 中，实际矩阵变量 `val_P` 已经是 $\mathbf P^H$，所以代码写作：

$$
(\mathbf A^H\mathbf A+\sigma_n^2\mathbf I)\hat{\mathbf H}_h
=
\mathbf A^H\mathbf Y,
\qquad
\mathbf A=\mathbf P^H
$$

并通过 `np.linalg.lstsq` 求解。

## 3. 重要参数

| 参数 | 默认值 | 说明 |
| ---- | ------ | ---- |
| `train_seed` | `1111` | 训练集数据种子 |
| `val_seed` | `2222` | 测试集数据种子 |
| `num_test_sample` | `256` | ML baseline 测试样本数 |
| `pilot_alpha` | `0.6` | 导频数量比例 |
| `spacing` | `[0.5]` | 天线间距配置 |
| `snr_range` | `-10:2.5:30` dB | 测试 SNR 扫描范围 |

导频数量：

$$
N_p = \lfloor N_t \alpha \rfloor
$$

其中 $\alpha$ 对应 `pilot_alpha`。

当前 ML baseline 中噪声功率按下式扫描：

$$
\sigma_n^2 = 10^{-SNR/10}
$$

噪声添加形式为：

$$
\mathbf N =
\frac{\sqrt{\sigma_n^2}}{\sqrt{2}}
(\mathbf N_r + j\mathbf N_i)
$$

其中 $\mathbf N_r$ 和 $\mathbf N_i$ 为标准高斯噪声。

## 4. 评价指标

NMSE 定义为：

$$
NMSE =
\frac{\|\hat{\mathbf H}_h-\mathbf H_h\|_F^2}
{\|\mathbf H_h\|_F^2}
$$

结果保存到：

```text
baseline_utils/results/ml_baseline/train<train>_test<test>/
```

保存字段包括：

| 字段 | 说明 |
| ---- | ---- |
| `snr_range` | SNR 扫描点 |
| `spacing` | 天线间距 |
| `pilot_alpha` | 导频比例 |
| `nmse_all` | 每个 SNR、每个样本的 NMSE |
| `avg_nmse` | 每个 SNR 的平均 NMSE |

