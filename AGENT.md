# Agent Instructions (LEO CE Baseline)

你正在处理的仓库是 **LEO_CE_baseline**。在开始任何代码修改、重构、实验脚本调整或新增功能前，请严格遵循以下工作流程与约束。

## 0. 必读文档（先读再做）

在进行规划与实现前，必须先通读并理解以下文档中的现有设计、术语与实验假设：

1. [README.md](README.md)
2. [GAI_COMM.md](GAI_COMM.md)

> 要求：先阅读上述文档，再进行任务。

## 1. 项目定位与实现边界

本仓库实现的是基于 score-based SDE 扩散模型的 MIMO 信道估计 baseline。当前代码以 DeepMIMO-5GNR 生成的复数信道矩阵为数据源，使用接收导频信号作为条件，通过 classifier-guided predictor-corrector 反向采样恢复信道。

1. 不要把当前仓库描述为完整的 LEO 轨道传播、卫星星历解析或 NTN 物理链路仿真器；当前主线是信道估计与生成式恢复 baseline。
2. 涉及 LEO/NTN 扩展时，需要明确说明新增数据、几何模型、信道模型与当前 DeepMIMO 数据接口之间的关系。
3. 技术文档应直接描述当前实现事实，不使用“旧方案/新方案”“之前/现在”等修改痕迹式表述。
4. 论文复现实验、baseline 对比和模型结构说明应与代码入口、配置文件和输出目录保持一致。

## 2. 模块结构与重构约束

重构时必须保持现有职责边界清晰，避免把训练、采样、数据加载、模型定义和 baseline 对比堆到单个大文件中：

1. 扩散模型训练与通用 SDE 逻辑保留在 `sde_score/` 下，例如 `run_lib.py`、`sde_lib.py`、`losses.py` 和 `models/`。
2. 条件信道生成与 PC 采样逻辑保留在顶层 `controllable_channel_generation.py` 和 `channel_sampling.py`，或在重构时拆入职责明确的小模块。
3. DeepMIMO 数据加载、复数/实数张量转换和导频构造逻辑保留在 `loaders.py` 或专门的数据模块中。
4. 配置继续放在 `configs/` 下，新增实验配置应进入对应 SDE 类型子目录，例如 `configs/ve/` 或 `configs/vp/`。
5. 传统 baseline、GAN baseline、压缩感知 baseline 等对比代码放在 `baseline_utils/`，不要混入扩散模型核心训练代码。
6. `__init__.py` 只允许包说明或少量稳定 re-export，不允许放训练流程、数据读取、文件输出或实验默认值。
7. 单个模块同时承担数据解析、模型推理、绘图和文件输出等多类职责时，需要拆分到同一领域下的独立函数或模块。

## 3. 配置、默认值与实验复现约束

新增、调整或重构运行参数时，必须让默认值来源清晰，避免在多个入口脚本中出现彼此不一致的实验假设。

1. 训练 batch size、SDE 类型、噪声尺度、NCSN++ 网络结构、优化器、数据归一化、天线维度、路径数、导频数和采样步数等默认值应集中在 `configs/default_CE_configs.py` 与具体配置文件中。
2. 命令行参数只负责选择 GPU、训练场景、测试场景、checkpoint、导频比例、天线间距和输出路径等实验入口参数。
3. 新增配置字段时，需要同步更新配置文件、调用入口、README 和 [LEO_CE_BASELINE.md](LEO_CE_BASELINE.md)。
4. 随机种子、数据集文件名规则、归一化方式、SNR 扫描范围、样本数量和输出目录属于可复现实验假设，修改时必须在文档或配置中同步说明。
5. 不要在模型代码中隐藏新的实验默认值；若必须引入公式常数或数值稳定项，应在就近注释中说明其用途。

## 4. 数据与复数张量约束

当前信道数据是复数 MIMO 矩阵，代码通过实部/虚部双通道张量喂入扩散模型。修改数据链路时必须保持形状和共轭转置约定一致。

1. DeepMIMO `.mat` 数据默认位于 `DeepMIMO-5GNR/DeepMIMO_dataset/`，文件名形如 `<scenario>_path<num_paths>_seed<seed>.mat`。
2. `Channels` 数据集读取字段 `channels`，并根据 `config.data.image_size=[64,16]` 使用 `n_tx=64`、`n_rx=16`。
3. 模型训练使用 `H_herm`，张量形状为 `(batch, 2, n_tx, n_rx)`，两个通道分别表示实部和虚部。
4. 条件估计中的导频矩阵 `P` 为 QPSK 随机导频，观测模型与 `Y = P^H H + N` 的实现保持一致。
5. 修改 `map_complex_to_coeff()`、`map_coeff_to_complex()`、`map_complex_to_components()` 或 `map_components_to_complex()` 时，必须同时检查训练、测试和 baseline 脚本中的维度使用。

## 5. 任务收尾

对于会改变代码逻辑、实验入口、配置默认值、数据格式或输出结果的改动，你都需要同步维护以下文档：

1. [README.md](README.md)
2. [GAI_COMM.md](GAI_COMM.md)

维护文档时必须保持口径一致，直接描述项目当前设计和当前模型。需要表达限制时，写成稳定的实现边界；需要表达扩展方向时，写成明确的后续工作，不混入当前已实现能力。
