# Diffusion 信道估计说明

本文档概述当前项目中基于 score-based SDE Diffusion Model 的 MIMO 信道估计流程。主要实现入口为 `train_diffusion_model.py`、`test_diffusion_model.py`、`controllable_channel_generation.py` 和 `channel_sampling.py`。

## 1. 基本思想

扩散模型先学习真实信道矩阵的先验分布，再在测试阶段把接收导频信号作为条件，引导反向采样过程生成满足观测约束的信道估计。

原始复数信道为：

$$
\mathbf H \in \mathbb C^{N_r \times N_t}
$$

代码中训练和采样主要使用 Hermitian 后的信道表示：

$$
\mathbf X =
\begin{bmatrix}
\Re\{\mathbf H^H\} \\
\Im\{\mathbf H^H\}
\end{bmatrix}
\in \mathbb R^{2 \times N_t \times N_r}
$$

对应 batch 张量形状为：

```text
(batch_size, 2, N_t, N_r)
```

## 2. 训练过程

训练入口：

```bash
python train_diffusion_model.py --gpu_id 0 --train Rural --workdir models/DM/Rural --n_iters 200000 --no_snapshot_sampling
```

训练流程：

1. 通过 `loaders.Channels` 读取 `.mat` 信道数据。
2. 将复数信道归一化，并拆分为实部、虚部两个通道。
3. 使用 VE SDE 对信道样本加入不同强度的高斯扰动。
4. 训练 NCSN++ score network 估计噪声扰动分布的 score。
5. 周期性保存 checkpoint 到 `models/DM/<scenario>/checkpoints/`。

VE SDE 的边缘扰动形式为：

$$
\mathbf x(t) = \mathbf x(0) + \sigma(t)\mathbf z,
\qquad
\mathbf z \sim \mathcal N(\mathbf 0, \mathbf I)
$$

其中：

$$
\sigma(t)=\sigma_{\min}
\left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)^t
$$

score network 学习：

$$
s_\theta(\mathbf x(t), t)
\approx
\nabla_{\mathbf x(t)} \log p_t(\mathbf x(t))
$$

## 3. 条件信道估计过程

测试入口：

```bash
python test_diffusion_model.py --gpu_id 0 --train Rural --test Rural --model_pth checkpoint_20.pth --snr_values -15 -10 -5 0 5 10 15 20
```

导频观测模型为：

$$
\mathbf Y = \mathbf P^H \mathbf H^H + \mathbf N
$$

其中 `P` 为 QPSK 导频矩阵，`Y` 为接收导频信号。测试时每个 SNR 点的噪声功率为：

$$
\sigma_n^2 = 10^{-SNR/10} N_t
$$

条件扩散采样从 VE SDE 先验开始：

$$
\mathbf X_T \sim \mathcal N(0, \sigma_{\max}^2 \mathbf I)
$$

然后使用 Predictor-Corrector 反向采样。当前代码默认：

```text
Predictor: ReverseDiffusionPredictor
Corrector: LangevinCorrector
corrector snr: 0.16
corrector steps: 2
probability_flow: False
```

条件梯度来自观测似然：

$$
\nabla_{\mathbf H}\log p(\mathbf Y|\mathbf H)
=
\frac{\mathbf P(\mathbf Y-\mathbf P^H\mathbf H)}{\sigma_n^2}
$$

Predictor 中使用扩散先验 score 和条件 score 的组合：

$$
g(\mathbf X,t)
=
s_\theta(\mathbf X,t)
+ \omega \nabla_{\mathbf X}\log p(\mathbf Y|\mathbf X)
$$

其中当前实现会根据两个梯度的平均幅度自适应计算权重 `omega`，使条件梯度与 score 梯度处于相近数量级。

## 4. 重要参数

| 参数 | 默认值 | 说明 |
| ---- | ------ | ---- |
| `sigma_min` | `0.01` | VE SDE 最小噪声尺度 |
| `sigma_max` | `50` | VE SDE 最大噪声尺度 |
| `num_scales` | `2100` | 反向采样离散步数 |
| `training.batch_size` | `32` | 扩散模型训练 batch size |
| `optim.lr` | `2e-4` | Adam 学习率 |
| `model.ema_rate` | `0.999` | EMA 衰减系数 |
| `pilot_alpha` | `0.6` | 导频数量比例 |
| `num_test_sample` | `64` | 默认测试样本数 |
| `snr_range` | `[-15, -10, -5, 0, 5, 10, 15, 20]` dB | 测试 SNR 扫描范围 |

导频数量按下式设置：

$$
N_p = \lfloor N_t \alpha \rfloor
$$

其中 $\alpha$ 对应 `pilot_alpha`。

## 5. 评价指标

每个 SNR 和每个采样步都会记录 NMSE：

$$
NMSE =
\frac{\|\hat{\mathbf H}-\mathbf H\|_F^2}
{\|\mathbf H\|_F^2}
$$

绘图时通常转换为 dB：

$$
NMSE_{dB}=10\log_{10}(NMSE)
$$

测试结果默认保存到：

```text
results/DM/test_train-<train>_test-<test>/results.pt
```

测试脚本同时保存线性 NMSE 曲线和 CSV：

```text
results/DM/test_train-<train>_test-<test>/results_mse.png
results/DM/test_train-<train>_test-<test>/results.csv
```
