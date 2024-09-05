#!/bin/bash
#SBATCH --job-name=frozenseg
#SBATCH --output=output/slurm/%j.run.out
#SBATCH --error=output/slurm/%j.run.err
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --comment=yhx_team
export MODULEPATH="/opt/app/spack/share/spack/modules/linux-centos7-haswell:/opt/app/spack/share/spack/modules/linux-centos7-cascadelake:/usr/share/Modules/modulefiles:/etc/modulefiles:/opt/app/modulefiles"
source /users/cx_xchen/.bashrc_12.1 
export DETECTRON2_DATASETS=/users/cx_xchen/DATASETS/
export TORCH_DISTRIBUTED_DEBUG=DETAIL
conda activate frozenseg

port=$((10000 + RANDOM % 50000))
sam=vit_b
path=output/ConvNext-L_${sam}_1x
python train_net.py --num-gpus 4 --dist-url tcp://127.0.0.1:$port \
  --config-file configs/coco/frozenseg/convnext_large_eval_ade20k.yaml \
  OUTPUT_DIR  $path \
  MODEL.MASK_FORMER.SAM_QUERY_FUSE_LAYER 2 \
  MODEL.MASK_FORMER.SAM_FEATURE_FUSE_LAYER 0 \
  MODEL.SAM_NAME $sam \
  MODEL.FROZEN_SEG.CLIP_PRETRAINED_WEIGHTS pretrained_checkpoint/models--laion--CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/open_clip_pytorch_model.bin
