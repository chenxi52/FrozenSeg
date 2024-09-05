#!/bin/bash
#SBATCH --job-name=save_sam_masks
#SBATCH --output=output/slurm/%j.run.out
#SBATCH --error=output/slurm/%j.run.err
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=128
#SBATCH --comment=yhx_team
export OMP_NUM_THREADS=4
export DETECTRON2_DATASETS=/users/cx_xchen/DATASETS/
export MODULEPATH="/opt/app/spack/share/spack/modules/linux-centos7-haswell:/opt/app/spack/share/spack/modules/linux-centos7-cascadelake:/usr/share/Modules/modulefiles:/etc/modulefiles:/opt/app/modulefiles"
source /users/cx_xchen/.bashrc_12.1
conda activate ovsam

port=$((10000 + RANDOM % 50000))
configs=(
    # "configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_eval_a847.yaml"
    # "configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_eval_ade20k.yaml"
    # "configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_eval_pas21.yaml"
    # "configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_eval_pc59.yaml"
    # "configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_eval_pc459.yaml"
    # "configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_eval_cityscapes.yaml"
    # "configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_eval_coco.yaml"
    # "configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_eval_mapillary_vistas.yaml"
)

python save_sam_masks.py
