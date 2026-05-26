# WGAN 信道估计说明

本文档概述当前项目中 WGAN-GP baseline 的训练与信道估计流程。主要实现入口为 `baseline_utils/wgan_gp.py` 和 `baseline_utils/test_wgan.py`。

## 1. 基本思想

WGAN-GP 先学习信道在角度域中的生成先验。测试阶段不直接输出信道，而是固定训练好的 Generator，通过优化潜变量 `z`，寻找一个既符合导频观测又落在生成模型信道流形上的估计。

Generator 输入：

$$
\mathbf z \sim \mathcal N(\mathbf 0,\mathbf I),
\qquad
\mathbf z \in \mathbb R^{65}
$$

Generator 输出角度域信道的实部和虚部：

$$
G_\theta(\mathbf z)
\in \mathbb R^{2 \times N_t \times N_r}
$$

## 2. WGAN-GP 训练过程

训练入口：

```bash
python baseline_utils/wgan_gp.py --gpu 0 --train Rural
```

主要流程：

1. 读取训练场景 `.mat` 信道数据。
2. 使用训练集标准差归一化信道。
3. 使用 DFT 基把空间域信道转换到角度域。
4. 将角度域复数信道拆成实部和虚部两个通道。
5. 使用 WGAN-GP 训练 Generator 和 Discriminator。
6. 按 `save_freq` 保存 Generator checkpoint。

角度域转换可概括为：

$$
\tilde{\mathbf H}
=
(\mathbf A_R^H \mathbf H \mathbf A_T)^T
$$

其中 $\mathbf A_T$ 和 $\mathbf A_R$ 为发送端和接收端 DFT 基。

WGAN-GP 的判别器损失为：

$$
L_D
=
-
\left(
\mathbb E_{\mathbf x \sim p_{data}}[D(\mathbf x)]
-
\mathbb E_{\mathbf z}[D(G(\mathbf z))]
\right)
+ \lambda_{gp}
\mathbb E_{\hat{\mathbf x}}
\left(\|\nabla_{\hat{\mathbf x}}D(\hat{\mathbf x})\|_2-1\right)^2
$$

生成器损失为：

$$
L_G
=
-
\mathbb E_{\mathbf z}[D(G(\mathbf z))]
$$

## 3. WGAN 信道估计过程

测试入口：

```bash
python baseline_utils/test_wgan.py --gpu 0 --train Rural --test Rural --pilot_alpha 0.6
```

测试时构造混合波束导频观测：

$$
\mathbf Y
=
\mathbf W \mathbf H \mathbf F \mathbf S + \mathbf N
$$

其中：

| 符号 | 含义 |
| ---- | ---- |
| $\mathbf W$ | 接收 combiner |
| $\mathbf F$ | 训练 precoder |
| $\mathbf S$ | QPSK pilot symbols |
| $\mathbf H$ | 待估计空间域信道 |

向量化后可写为：

$$
\mathbf y = \mathbf A \operatorname{vec}(\mathbf H) + \mathbf n
$$

其中代码中：

$$
\mathbf A =
\operatorname{kron}((\mathbf F\mathbf S)^T, \mathbf W)
$$

Generator 输出先位于角度域，再通过 DFT Kronecker 基转换回空间域：

$$
\operatorname{vec}(\mathbf H)
=
(\mathbf A_T^* \otimes \mathbf A_R)
\operatorname{vec}(\tilde{\mathbf H})
$$

测试阶段固定 Generator 参数，优化潜变量：

$$
\mathbf z^\*
=
\arg\min_{\mathbf z}
\left\|
\mathbf y
-
\mathbf A
\operatorname{vec}(G_\theta(\mathbf z))
\right\|_2^2
+ \lambda \|\mathbf z\|_2^2
$$

当前代码使用 Adam 对 `z` 优化 200 步。

## 4. 重要参数

| 参数 | 默认值 | 说明 |
| ---- | ------ | ---- |
| `latent_dim` | `65` | Generator 潜变量维度 |
| `batch_size` | `200` | WGAN 训练 batch size |
| `n_iters` | `60000` | WGAN 训练迭代次数 |
| `critic_steps` | `5` | 每次更新 Generator 前更新 Discriminator 的次数 |
| `lr` | `5e-5` | WGAN RMSprop 学习率 |
| `lambda_gp` | `10.0` | gradient penalty 权重 |
| `save_freq` | `2000` | Generator checkpoint 保存间隔 |
| `pilot_alpha` | `0.6` | 导频数量比例 |
| `lambda_reg` | `1e-3` | 测试时潜变量正则项权重 |
| `learning_rate` | `0.02` | 测试时优化潜变量 `z` 的 Adam 学习率 |
| `latent optimization steps` | `200` | 每个样本估计时优化 `z` 的步数 |
| `SNR_vec` | `-10:2.5:30` dB | 测试 SNR 扫描范围 |

导频数量：

$$
N_p = \lfloor \alpha N_t \rfloor
$$

其中 $\alpha$ 对应 `pilot_alpha`。

## 5. 评价指标

WGAN baseline 使用 NMSE 评价估计结果：

$$
NMSE =
\frac{\|\hat{\mathbf H}-\mathbf H\|_F^2}
{\|\mathbf H\|_F^2}
$$

结果保存到：

```text
results/wgan_gp/train<train>_test<test>/
```

保存字段包括：

| 字段 | 说明 |
| ---- | ---- |
| `nmse_all` | 每个 SNR、checkpoint、repeat 和 test sample 的 NMSE |
| `avg_nmse` | 聚合后的平均 NMSE |
| `pilot_alpha` | 导频比例 |
| `snr_range` | SNR 扫描点 |
| `config` | 测试配置 |

