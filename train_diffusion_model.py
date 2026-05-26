"""
Training diffusion model by score-based SDE
"""
import os, sys
os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:516"
import sde_score.run_lib as run_lib
import logging
import tensorflow as tf
import torch, os, argparse
from loaders import expand_scenarios, infer_channel_image_size

#优化CUDA计算精度与性能
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = True

def main(args):
  # Create the working directory 创建工作目录，用于保存模型、日志
  tf.io.gfile.makedirs(args.workdir)
  # Set logger so that it outputs to both console and file 配置日志：输出到控制台和文件
  # Make logging work for both disk and Google Cloud Storage 放置日志到文件
  gfile_stream = open(os.path.join(args.workdir, 'stdout.txt'), 'w')
  handler = logging.StreamHandler(gfile_stream)
  formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
  handler.setFormatter(formatter)
  logger = logging.getLogger()
  logger.addHandler(handler)
  logger.setLevel('INFO')
  # Run the training pipeline 启动训练：调用run_lib中的train函数（核心训练逻辑）
  run_lib.train(args.config, args.workdir)


if __name__ == "__main__":
  #定义命令行参数
  parser = argparse.ArgumentParser()
  parser.add_argument('--gpu_id', type=int, default=0)  # Assign the gpu device idx 配置GPU设备，指定使用的GPU设备
  parser.add_argument(
      '--train',
      type=str,
      default='mixed',
      help=(
          "Training scenario. Supported values include "
          "'O1_28B', 'O1_28', 'I2_28B', 'mixed', "
          "'Rural', 'Urban', 'DenseUrban', 'mixed_leo'."
      ),
  )
  parser.add_argument('--scenario_list', type=list, default=['O1_28B', 'O1_28', 'I2_28B'])
  parser.add_argument('--workdir', type=str, default='')
  parser.add_argument('--eval_folder', type=str, default='eval')
  parser.add_argument('--n_iters', type=int, default=None,
                      help='Override config.training.n_iters for this run.')
  parser.add_argument('--no_snapshot_sampling', action='store_true',
                      help='Disable conditional sampling at training snapshots.')
  args = parser.parse_args()
  args.scenario_list = expand_scenarios(args.train)
  if args.train == 'O1_28':
    from configs.ve.CE_ncsnpp_deep_continuous import get_config
  else:
    from configs.ve.CE_ncsnpp_deep_continuous_norm import get_config

  config = get_config()

  # Choose GPU 
  torch.cuda.set_device(args.gpu_id)
  config.device = torch.device('cuda', args.gpu_id)

  # Choose channel scenario
  config.data.train_scenario = args.train
  config.data.scenario_list = args.scenario_list
  config.data.image_size = infer_channel_image_size(
      config.data.scenario_list,
      seed=1111,
      num_paths=config.data.num_paths,
  )
  if args.n_iters is not None:
    config.training.n_iters = args.n_iters
  if args.no_snapshot_sampling:
    config.training.snapshot_sampling = False
  if not args.workdir:
    args.workdir = os.path.join('models', 'DM', args.train)
  args.config = config

  main(args)

