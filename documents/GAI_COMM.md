# GAI_COMM 项目总览

本文档只用于说明仓库的总体定位、模块组织、数据入口和实验输出。具体方法细节分别维护在：

- `documents/Diffusion.md`
- `documents/ML.md`
- `documents/WGAN.md`

## 1. 项目目标

`GAI_COMM` 面向无线 MIMO 信道估计任务，核心思路是将接收导频信号作为条件，恢复复数信道矩阵，并使用 NMSE 评估不同方法在不同 SNR 下的估计性能。

当前仓库包含两类实现路线：

- 基于 score-based SDE 的条件扩散模型
- 对比 baseline，包括 ML 和 WGAN-GP

## 2. 数据与场景

仓库支持两类 `.mat` 信道数据：

- DeepMIMO 场景：`O1_28`、`O1_28B`、`I2_28B`、`mixed`
- LEO 场景：`Rural`、`Urban`、`DenseUrban`、`mixed_leo`

默认数据路径：

```text
DeepMIMO-5GNR/DeepMIMO_dataset/
dataset/
```

每个数据文件需要包含 `channels` 字段。训练集和测试集默认使用不同随机种子读取不同文件，当前约定为：

- 训练种子：`1111`
- 验证/测试种子：`2222`

## 3. 代码结构

项目当前按职责划分为以下模块：

| 模块 | 主要职责 |
| ---- | -------- |
| `train_diffusion_model.py` | 扩散模型训练入口 |
| `test_diffusion_model.py` | 条件扩散信道估计测试入口 |
| `baseline_utils/` | ML、WGAN-GP 等 baseline 训练与测试 |
| `sde_score/` | SDE、loss、采样器和模型结构 |
| `loaders.py` | 数据集读取、导频生成、张量组织 |
| `controllable_channel_generation.py` | 条件扩散采样逻辑 |
| `channel_sampling.py` | predictor / corrector 更新器 |
| `configs/` | 默认实验配置和场景配置 |
| `documents/` | 项目总览和各方法说明文档 |

## 4. 实验入口

当前常用入口如下：

```bash
python train_diffusion_model.py --gpu_id 0 --train Rural --workdir models/DM/Rural --n_iters 200000 --no_snapshot_sampling
python test_diffusion_model.py --gpu_id 0 --train Rural --test Rural --model_pth checkpoint_20.pth --snr_values -15 -10 -5 0 5 10 15 20
python baseline_utils/test_ml.py --gpu 0 --train Rural --test Rural
python baseline_utils/wgan_gp.py --gpu 0 --train Rural
python baseline_utils/test_wgan.py --gpu 0 --train Rural --test Rural
```

## 5. 输出约定

实验结果默认按方法分别输出：

- 扩散模型测试结果：`results/DM/test_train-<train>_test-<test>/`
- ML baseline 结果：`baseline_utils/results/ml_baseline/train<train>_test<test>/`
- WGAN baseline 结果：`results/wgan_gp/train<train>_test<test>/`

模型权重默认输出：

- 扩散模型 checkpoint：`models/DM/<scenario>/checkpoints/`
- WGAN generator checkpoint：`models/wgan_gp/<scenario>/`

## 6. 文档维护原则

`documents/GAI_COMM.md` 只保留项目总体说明，不展开具体公式推导、方法细节和单个 baseline 的实现流程。涉及具体方法时，应同步维护对应文档：

- 扩散模型相关改动更新 `documents/Diffusion.md`
- ML baseline 相关改动更新 `documents/ML.md`
- WGAN baseline 相关改动更新 `documents/WGAN.md`
