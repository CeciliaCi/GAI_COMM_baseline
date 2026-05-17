# LEO CE Baseline 技术方案

本文档描述当前仓库已经实现的 MIMO 信道估计 baseline。实现目标是：使用 DeepMIMO-5GNR 或 LEO-NTN `.mat` 复数信道矩阵训练 score-based SDE 扩散模型与 WGAN-GP baseline，在测试阶段把接收导频信号作为条件，通过条件生成方式恢复信道，并以 NMSE 随 SNR 的变化评估信道估计性能。

当前实现的主要入口为：

| 功能 | 入口脚本 | 主要作用 | 默认输出 |
| ---- | -------- | -------- | -------- |
| 扩散模型训练 | `train_diffusion_model.py` | 训练 NCSN++ score model | `models/DM/<scenario>/checkpoints/` |
| 条件信道估计测试 | `test_diffusion_model.py` | 加载 checkpoint 并执行 PC 条件采样 | `results/DM/test_train-<train>_test-<test>/` |
| NMSE 结果转换 | `calMSE.py` | 从 `results.pt` 绘制并导出 NMSE 曲线 | `results_nmse.png`、`results_mat.mat` |
| 传统 baseline | `baseline_utils/` | ML、Lasso/l1CS、WGAN-GP 对比实验 | `results/<baseline>/...` |

公共 SDE、loss、模型结构和训练循环位于 `sde_score/`。DeepMIMO 数据加载位于 `loaders.py`；条件扩散采样位于 `controllable_channel_generation.py`；predictor/corrector 更新器位于 `channel_sampling.py`；实验默认值位于 `configs/`。

---

## 1. 数据集与场景

当前仓库支持两类 `.mat` 信道数据输入。README 中列出的默认场景为：

| 场景 | 类型 | 默认文件名规则 |
| ---- | ---- | -------------- |
| `O1_28` | 室外场景 | `O1_28_path10_seed<seed>.mat` |
| `O1_28B` | 室外场景 | `O1_28B_path10_seed<seed>.mat` |
| `I2_28B` | 室内场景 | `I2_28B_path10_seed<seed>.mat` |
| `mixed` | 混合场景 | 同时使用 `O1_28B`、`O1_28`、`I2_28B` |
| `Rural` | LEO-NTN 场景 | `LEO_Rural_seed<seed>.mat` |
| `Urban` | LEO-NTN 场景 | `LEO_Urban_seed<seed>.mat` |
| `DenseUrban` | LEO-NTN 场景 | `LEO_DenseUrban_seed<seed>.mat` |
| `mixed_leo` | LEO-NTN 混合场景 | 同时使用 `Rural`、`Urban`、`DenseUrban` |

数据文件默认放置在：

```text
DeepMIMO-5GNR/DeepMIMO_dataset/
```

LEO-NTN 数据文件默认放置在：

```text
dataset/
```

每个 `.mat` 文件需要包含 `channels` 字段。`loaders.Channels` 会根据场景名自动解析 `DeepMIMO-5GNR/DeepMIMO_dataset/<scenario>_path<num_paths>_seed<seed>.mat` 或 `dataset/LEO_<Scenario>_seed<seed>.mat`，再按配置中的场景列表逐个读取并拼接为统一的信道样本集合。训练集默认使用随机种子 `1111`，验证/测试集默认使用随机种子 `2222`。

---

## 2. 信道矩阵与张量表示

原始信道为复数 MIMO 矩阵：

$$
\mathbf H \in \mathbb C^{N_r \times N_t}.
$$

默认配置中：

| 参数 | 默认值 | 含义 |
| ---- | ------ | ---- |
| `config.data.image_size` | `[64, 16]` | 扩散模型输入中的 `[N_t, N_r]` |
| `config.data.num_channels` | `2` | 实部和虚部两个通道 |
| `config.data.num_paths` | `10` | DeepMIMO 文件名中的路径数 |
| `config.data.spacing_list` | `[0.5]` | 天线间距配置 |

数据集输出同时保留原信道和 Hermitian 信道。扩散模型训练使用 Hermitian 信道：

$$
\mathbf X =
\begin{bmatrix}
\Re\{\mathbf H^H\} \\
\Im\{\mathbf H^H\}
\end{bmatrix}
\in \mathbb R^{2 \times N_t \times N_r}.
$$

在 PyTorch 中，一个 batch 的训练张量形状为：

```text
(batch_size, 2, 64, 16)
```

`controllable_channel_generation.py` 中提供两类复数映射：

```python
map_complex_to_coeff(X)      # complex -> (batch, 2, n_tx, n_rx)
map_coeff_to_complex(X_coeff) # (batch, 2, n_tx, n_rx) -> complex
```

这些函数决定条件梯度、NMSE 计算和最终信道恢复的维度约定。

---

## 3. 导频观测模型

`Channels` 为每个信道样本生成随机 QPSK 导频矩阵：

$$
\mathbf P \in \mathbb C^{N_t \times N_p}.
$$

导频符号来自：

$$
\frac{1}{\sqrt 2}\{1+j,\ 1-j,\ -1+j,\ -1-j\}.
$$

在测试入口中，导频数量由导频比例给出：

$$
N_p=\left\lfloor N_t \alpha \right\rfloor,
$$

其中 `--pilot_alpha` 默认值为 `0.6`。接收导频信号按以下形式构造：

$$
\mathbf Y = \mathbf P^H\mathbf H^H + \mathbf N.
$$

对每个 SNR 扫描点，噪声功率为：

$$
\sigma_n^2 = 10^{-SNR/10} N_t.
$$

`test_diffusion_model.py` 默认扫描：

```text
-10 dB, -7.5 dB, ..., 30 dB
```

---

## 4. 扩散模型训练

当前主配置使用 VE SDE 与 NCSN++ score network。默认训练入口根据训练场景选择配置：

| 训练场景 | 配置文件 | 归一化 |
| -------- | -------- | ------ |
| `O1_28` | `configs/ve/CE_ncsnpp_deep_continuous.py` | `global` |
| 其他场景 | `configs/ve/CE_ncsnpp_deep_continuous_norm.py` | `zero_mean` |

VE SDE 的正向边缘分布为：

$$
\mathbf x(t)=\mathbf x(0)+\sigma(t)\mathbf z,
\qquad
\mathbf z\sim\mathcal N(\mathbf 0,\mathbf I),
$$

其中：

$$
\sigma(t)=\sigma_{\min}
\left(\frac{\sigma_{\max}}{\sigma_{\min}}\right)^t.
$$

默认参数为：

| 参数 | 默认值 |
| ---- | ------ |
| `sigma_min` | `0.01` |
| `sigma_max` | `50` |
| `num_scales` | `2100` |
| `training.batch_size` | `32` |
| `optim.lr` | `2e-4` |
| `model.ema_rate` | `0.999` |

训练损失通过 `sde_score.losses.get_step_fn()` 构造，在随机时间 $t$ 上训练 score network 近似：

$$
s_\theta(\mathbf x(t),t)\approx \nabla_{\mathbf x(t)}\log p_t(\mathbf x(t)).
$$

训练循环位于 `sde_score/run_lib.py`，主要流程为：

1. 创建 NCSN++ 模型、EMA 和 Adam 优化器。
2. 从 `Channels` 读取 `H_herm` batch。
3. 按 VE SDE 噪声扰动信道张量。
4. 更新 score network。
5. 周期性保存 `checkpoints-meta/checkpoint.pth` 和 `checkpoints/checkpoint_<k>.pth`。

---

## 5. 条件扩散信道估计

测试阶段从 VE SDE 先验噪声开始反向采样：

$$
\mathbf X_T \sim \mathcal N(0,\sigma_{\max}^2\mathbf I).
$$

反向采样使用 predictor-corrector 框架：

| 组件 | 默认实现 |
| ---- | -------- |
| Predictor | `ReverseDiffusionPredictor` |
| Corrector | `LangevinCorrector` |
| corrector SNR | `0.16` |
| corrector steps | `2` |
| probability flow | `False` |

条件项来自导频观测似然。设当前采样信道为 $\mathbf X$，其复数形式为 $\mathbf H_{est}$，则条件梯度使用：

$$
\nabla_{\mathbf H}\log p(\mathbf Y|\mathbf H)
=
\frac{\mathbf P(\mathbf Y-\mathbf P^H\mathbf H)}{\sigma_n^2}.
$$

代码中 `condition_grad_fn()` 将该复数梯度重新映射为实部/虚部双通道张量。Predictor 中使用 score 与条件 score 的加权和：

$$
g(\mathbf X,t)
=s_\theta(\mathbf X,t)+\omega\nabla_{\mathbf X}\log p(\mathbf Y|\mathbf X).
$$

其中 $\omega$ 由 score 与条件梯度的平均幅度自适应缩放，使两类梯度处于相近数量级。Corrector 中直接使用 score 与条件 score 的和进行 Langevin 更新。

---

## 6. NMSE 评估

每个 SNR 点下，PC 采样会在每个反向时间步记录一次 NMSE：

$$
NMSE_i =
\frac{
\|\hat{\mathbf H}_i-\mathbf H\|_F^2
}{
\|\mathbf H\|_F^2
}.
$$

`test_diffusion_model.py` 保存三类结果：

| 字段 | 含义 |
| ---- | ---- |
| `nmse_all` | 每个 SNR、每个扩散步、每个样本的 NMSE |
| `avg_nmse` | 对测试样本求平均后的逐步 NMSE |
| `best_nmse` | 每个 SNR 下沿扩散步取最小平均 NMSE |

绘图时使用：

$$
NMSE_{dB}=10\log_{10}(NMSE).
$$

默认测试样本数为 `64`。README 中说明论文复现实验可使用 `256` 个测试样本获得更平滑曲线。

---

## 7. 输出内容

扩散模型测试输出目录为：

```text
results/DM/test_train-<train>_test-<test>/
```

主要产物为：

| 文件 | 内容 |
| ---- | ---- |
| `results.pt` | SNR 扫描、NMSE 记录、导频比例、天线间距和测试配置 |
| `results.png` | `best_nmse` 随 SNR 变化的曲线 |
| `results_nmse.png` | `calMSE.py` 重新绘制的 NMSE 曲线 |
| `results_mat.mat` | 供 MATLAB 或后续绘图使用的 NMSE/MSE 数据 |

训练输出目录为：

```text
models/DM/<train>/checkpoints/
```

主要产物为：

| 文件 | 内容 |
| ---- | ---- |
| `checkpoint_<k>.pth` | 周期性保存的模型、优化器和 EMA 状态 |
| `checkpoints-meta/checkpoint.pth` | 训练恢复用中间 checkpoint |
| `stdout.txt` | 训练日志 |
| `tensorboard/` | TensorBoard 标量日志 |

---

## 8. 命令行接口

默认训练：

```powershell
.\.venv\Scripts\python.exe train_diffusion_model.py --gpu_id 0 --train mixed --workdir models/DM/mixed/
```

默认测试：

```powershell
.\.venv\Scripts\python.exe test_diffusion_model.py --gpu_id 0 --train mixed --test mixed --model_pth checkpoint_20.pth
```

结果转换：

```powershell
.\.venv\Scripts\python.exe calMSE.py --train mixed --test mixed
```

主要参数为：

| 参数 | 入口 | 说明 |
| ---- | ---- | ---- |
| `--gpu_id` | train/test | CUDA 设备编号 |
| `--train` | train/test/calMSE | 训练场景，支持 `mixed`、`mixed_leo`、`O1_28`、`O1_28B`、`I2_28B`、`Rural`、`Urban`、`DenseUrban` |
| `--test` | test/calMSE | 测试场景 |
| `--model_pth` | test | checkpoint 文件名 |
| `--spacing` | test/baseline | 天线间距列表 |
| `--pilot_alpha` | test/baseline | 导频比例 |
| `--workdir` | train | 模型和日志输出目录 |

---

## 9. Baseline 对比

`baseline_utils/` 提供传统和生成式对比方法：

| 文件 | 方法 | 输出目录 |
| ---- | ---- | -------- |
| `test_ml.py` | 最大似然/最小二乘估计 | `results/ml_baseline/` |
| `test_l1Fourier_lifted.py` | Fourier 字典 lifting + L1 稀疏恢复 | `results/l1CS_lifted<k>/` |
| `wgan_gp.py` | WGAN-GP baseline 训练 | `models/wgan_gp/<train>/` |
| `test_wgan.py` | WGAN-GP 生成先验辅助估计 | `results/wgan_gp/` |

这些 baseline 使用同一套 loader 解析 DeepMIMO 或 LEO-NTN 信道数据，并沿用相同的 SNR 扫描和 NMSE 指标，服务于扩散模型信道估计结果的公平对比。

---

## 10. 当前实现边界

当前文档只描述项目已经实现的 baseline。实现边界如下：

- 当前输入数据源为外部生成的 DeepMIMO-5GNR 或 LEO-NTN `.mat` 信道数据，不包含卫星星历传播、LEO 轨道动力学或 NTN 链路几何生成。
- 当前扩散模型主线使用 VE SDE；VP/subVP 类在 `sde_score/sde_lib.py` 中存在实现，但主测试入口未启用。
- 当前训练和测试默认使用 NCSN++，输入为 Hermitian 信道的实部/虚部双通道张量。
- 当前测试通过 `best_nmse` 沿扩散采样步取最小平均 NMSE，用于评估反向采样过程中的最佳恢复点。
- 当前条件采样使用导频观测似然梯度，不引入显式信道物理参数、轨道状态或近地散射几何约束。
