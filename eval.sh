#!/bin/bash
#SBATCH --job-name=frozenseg_eval
#SBATCH --output=output/slurm/%j.run.out
#SBATCH --error=output/slurm/%j.run.err
#SBATCH --partition=gpu-a100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --comment=yhx_team
export MODULEPATH="/opt/app/spack/share/spack/modules/linux-centos7-haswell:/opt/app/spack/share/spack/modules/linux-centos7-cascadelake:/usr/share/Modules/modulefiles:/etc/modulefiles:/opt/app/modulefiles"
source /users/cx_xchen/.bashrc_12.1
export DETECTRON2_DATASETS=/users/cx_xchen/DATASETS/
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export OMP_NUM_THREADS=1
export USE_SIMPLE_THREADED_LEVEL3=1
conda activate frozenseg

configs=(
    # "configs/coco/frozenseg/convnext_large_eval_a847.yaml"
    # "configs/coco/frozenseg/convnext_large_eval_ade20k.yaml"
    # "configs/coco/frozenseg/convnext_large_eval_lvis.yaml"
    # "configs/coco/frozenseg/convnext_large_eval_pas21.yaml"
    "configs/coco/frozenseg/r50x64_eval_ade20k.yaml"
    # "configs/coco/frozenseg/convnext_large_eval_cityscapes.yaml"
    # "configs/coco/frozenseg/convnext_large_eval_coco.yaml"
    # "configs/coco/frozenseg/convnext_large_eval_mapillary_vistas.yaml"
    # configs/coco/frozenseg/convnext_large_eval_bdd_panop.yaml
    # configs/coco/frozenseg/convnext_large_eval_bdd_sem.yaml
)
# port=$((10000 + RANDOM % 50000))
# sam=vit_b
# path=output/ConvNext-L_${sam}_1x
# for config in "${configs[@]}"; do
#     python train_net.py --eval-only --num-gpus 1 --dist-url tcp://127.0.0.1:$port \
#         --config-file $config \
#         OUTPUT_DIR $path/$(basename "$config" .yaml) \
#         MODEL.WEIGHTS modified_model.pth \
#         MODEL.SAM_NAME vit_b \
#         MODEL.FROZEN_SEG.CLIP_PRETRAINED_WEIGHTS pretrained_checkpoint/models--laion--CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/open_clip_pytorch_model.bin \
#         TEST.USE_SAM_MASKS False \
#         MODEL.FROZEN_SEG.GEOMETRIC_ENSEMBLE_BETA 0.6
# done

########## with mask ensemble ########
# for config in "${configs[@]}"; do
#     python train_net.py --eval-only --num-gpus 1 --dist-url tcp://127.0.0.1:$port \
#         --config-file $config \
#         OUTPUT_DIR $path/w_maskEnsemble/$(basename "$config" .yaml) \
#         MODEL.WEIGHTS $path/model_final.pth \
#         MODEL.MASK_FORMER.SAM_QUERY_FUSE_LAYER 2 \
#         MODEL.MASK_FORMER.SAM_FEATURE_FUSE_LAYER 0 \
#         MODEL.SAM_NAME vit_b \
#         MODEL.FROZEN_SEG.CLIP_PRETRAINED_WEIGHTS pretrained_checkpoint/models--laion--CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup/open_clip_pytorch_model.bin \
#         TEST.USE_SAM_MASKS True \
#         TEST.PKL_SAM_MODEL_NAME vit_h
# done
port=$((10000 + RANDOM % 50000))

for path in output/test; do
for beta in 0.8; do
for model in final; do
for config in "${configs[@]}"; do
        python train_net.py --eval-only --num-gpus 2 --dist-url tcp://127.0.0.1:$port \
            --config-file $config \
            OUTPUT_DIR "$path/$(basename "$config" .yaml)" \
            MODEL.WEIGHTS "./frozenseg_RN50x64.pth" \
            MODEL.SAM_NAME vit_b 
    done
done
done
done
########### test recall ############
# path=output/Sam_query/ConvNext-L_vit_b_1x
# for config in "${configs[@]}"; do
#     srun python train_net.py --eval-only --num-gpus 4 --dist-url tcp://127.0.0.1:$port \
#         --config-file $config \
#         OUTPUT_DIR "output/Ablation/recall_withEverything/$(basename "$config" .yaml)" \
#         MODEL.WEIGHTS "$path/model_final.pth" \
#         TEST.USE_SAM_MASKS True \
#         MODEL.MASK_FORMER.TEST.RECALL_ON True \
#         MODEL.MASK_FORMER.TEST.SEMANTIC_ON False \
#         MODEL.MASK_FORMER.TEST.INSTANCE_ON False \
#         MODEL.MASK_FORMER.TEST.PANOPTIC_ON False \
# done