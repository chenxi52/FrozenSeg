try:
    from shapely.errors import ShapelyDeprecationWarning
    import warnings
    warnings.filterwarnings('ignore', category=ShapelyDeprecationWarning)
except:
    pass

import copy
import itertools    
import logging
import os

from collections import OrderedDict
from typing import Any, Dict, List, Set
import pycocotools.mask as mask_util

import torch
import numpy as np
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog, build_detection_train_loader, build_detection_test_loader
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.solver.build import maybe_add_gradient_clipping
from detectron2.utils.logger import setup_logger

from frozenseg import (
    COCOInstanceNewBaselineDatasetMapper,
    COCOPanopticNewBaselineDatasetMapper,
    InstanceSegEvaluator,
    MaskFormerInstanceDatasetMapper,
    MaskFormerPanopticDatasetMapper,
    MaskFormerSemanticDatasetMapper,
    SemanticSegmentorWithTTA,
    add_maskformer2_config,
    add_frozenseg_config,
)
from detectron2.solver import build_lr_scheduler
from collections import OrderedDict
from detectron2.utils.file_io import PathManager
from detectron2.utils.comm import all_gather, is_main_process, synchronize
import json
from detectron2.evaluation.sem_seg_evaluation import load_image_into_numpy_array
warnings.filterwarnings("ignore")


def prepare_class_names_from_metadata(metadata, train_metadata):
    def split_labels(x):
        res = []
        for x_ in x:
            x_ = x_.replace(', ', ',')
            x_ = x_.split(',') # there can be multiple synonyms for single class
            res.append(x_)
        return res
    # get text classifier
    try:
        class_names = split_labels(metadata.stuff_classes) # it includes both thing and stuff
        train_class_names = split_labels(train_metadata.stuff_classes)
    except:
        # this could be for insseg, where only thing_classes are available
        class_names = split_labels(metadata.thing_classes)
        train_class_names = split_labels(train_metadata.thing_classes)
    train_class_names = {l for label in train_class_names for l in label}
    category_overlapping_list = []
    for test_class_names in class_names:
        is_overlapping = not set(train_class_names).isdisjoint(set(test_class_names))
        category_overlapping_list.append(is_overlapping)
    category_overlapping_list = np.array(category_overlapping_list)
    
    return category_overlapping_list


class SemSegSeenUnseenRecallEvaluator(SemSegEvaluator):
    def __init__(self,
        dataset_name,
        distributed=True,
        output_dir=None,
        *,
        sem_seg_loading_fn=load_image_into_numpy_array,
        num_classes=None,
        ignore_label=None,
        train_dataset_name = None):
        # recall of the final result
        super().__init__(dataset_name,distributed,output_dir,sem_seg_loading_fn=sem_seg_loading_fn,num_classes=num_classes,ignore_label=ignore_label)
        train_metadata =  MetadataCatalog.get(train_dataset_name)
        test_metadata = MetadataCatalog.get(dataset_name)
        self.category_overlapping_mask = prepare_class_names_from_metadata(test_metadata, train_metadata)
        self.iou_thresholds =  [0.5, 0.75, 0.9]
    
    def reset(self):
        self._conf_matrix = np.zeros((self._num_classes + 1, self._num_classes + 1), dtype=np.int64)
        self._b_conf_matrix = np.zeros(
            (self._num_classes + 1, self._num_classes + 1), dtype=np.int64
        )
        self._predictions = []
        self._unseen_tp_cnt = np.zeros(len(self.iou_thresholds), dtype=np.int64)
        self._seen_tp_cnt = np.zeros(len(self.iou_thresholds), dtype=np.int64)
        self._unseen_labels = np.zeros(1, dtype=np.int64)
        self._seen_labels = np.zeros(1, dtype=np.int64)

    def process(self, inputs, outputs):
        """
         outputs: list of dicts with key "sem_seg" that contains 250 queries semantic
                segmentation prediction.
        """

        for input, output in zip(inputs, outputs):
            output = output["recall_seg"].to(self._cpu_device) # (n,h,w)
            output = output>0
            pred = np.array(output, dtype=int) #(n,h,w)
            gt_filename = self.input_file_to_gt_file[input["file_name"]]
            gt = self.sem_seg_loading_fn(gt_filename, dtype=int)
            gt[gt == self._ignore_label] = self._num_classes
            gt_classes = np.delete(np.unique(gt), np.where(np.unique(gt) == self._num_classes))
            for c in gt_classes:
                if self.category_overlapping_mask[c] == 1:
                    self._seen_labels += 1
                else:
                    self._unseen_labels += 1
            for i, thresh in enumerate(self.iou_thresholds):
                for c in gt_classes:
                    mask_true = gt == c # (h,w)
                    iou = self.calculate_iou(mask_true, pred) # n
                    if self.category_overlapping_mask[c] == 1:
                        self._seen_tp_cnt[i] += np.any(iou>thresh)
                    else:
                        self._unseen_tp_cnt[i] += np.any(iou>thresh)
            # [[tp_0.5, tp_0.75, tp_0.9], [tp_0.5, tp_0.75, tp_0.9]]
            self._predictions.extend(self.encode_json_recall_seg(pred, input["file_name"]))
            
    def calculate_iou(self, mask1, mask2):
        intersection = np.logical_and(mask1, mask2)
        union = np.logical_or(mask1, mask2)
        return np.sum(intersection, axis=(1,2)) / np.sum(union, axis=(1,2))    
    
    def encode_json_recall_seg(self, recall_seg, input_file_name):
        json_list = []
        for mask_pred in recall_seg:
            mask_pred = mask_pred.astype(np.uint8)
            mask_rle = mask_util.encode(np.array(mask_pred[:,:,None], order="F"))[0]
            mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
            json_list.append(
                {"file_name": input_file_name, "segmentation": mask_rle}
            )
        return json_list
    
    def evaluate(self):
        if self._distributed:
            synchronize()
            seen_tp_list = all_gather(self._seen_tp_cnt)
            unseen_tp_list = all_gather(self._unseen_tp_cnt)
            seen_labels = all_gather(self._seen_labels)
            unseen_labels = all_gather(self._unseen_labels)
            if not is_main_process():
                return
            self._seen_tp_cnt = np.zeros_like(self._seen_tp_cnt)
            self._unseen_tp_cnt = np.zeros_like(self._unseen_tp_cnt)
            self._seen_labels = np.zeros_like(self._seen_labels)
            self._unseen_labels = np.zeros_like(self._unseen_labels)
            for seen_tp in seen_tp_list:
                self._seen_tp_cnt += seen_tp
            for unseen_tp in unseen_tp_list:
                self._unseen_tp_cnt += unseen_tp
            for label1 in seen_labels:
                self._seen_labels += label1
            for label2 in unseen_labels:
                self._unseen_labels += label2

        if self._output_dir:
            PathManager.mkdirs(self._output_dir)
            file_path = os.path.join(self._output_dir, "recall_predictions.json")
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(self._predictions))
        # instance-level Recall
        seen_recalls = self._seen_tp_cnt / self._seen_labels
        unseen_recalls = self._unseen_tp_cnt / self._unseen_labels
        assert len(seen_recalls) == 3
        assert len(unseen_recalls) == 3
        res = {}
        for i, iou_threshold in enumerate([0.5, 0.75, 0.9]):
            res[f"S_Recall@IoU={iou_threshold:.2f}"] = 100 * seen_recalls[i]
            res[f"U_Recall@IoU={iou_threshold:.2f}"] = 100 * unseen_recalls[i]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "recall_evaluation.pth")
            with PathManager.open(file_path, "wb") as f:
                torch.save(res, f)
        results = OrderedDict({"recall_seg": res})
        self._logger.info(results)
        return results    
    
class Trainer(DefaultTrainer):
    """
    Extension of the Trainer class adapted to FrozenSeg.
    """

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """
        Create evaluator(s) for a given dataset.
        This uses the special metadata "evaluator_type" associated with each
        builtin dataset. For your own dataset, you can simply create an
        evaluator manually in your script and do not have to worry about the
        hacky if-else logic here.
        """
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        # semantic segmentation
        if evaluator_type in ["sem_seg", "ade20k_panoptic_seg"] and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(
                SemSegEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                )
            )
        if cfg.MODEL.MASK_FORMER.TEST.RECALL_ON:
            evaluator_list.append(
                SemSegSeenUnseenRecallEvaluator(
                    dataset_name,
                    distributed=True,
                    output_dir=output_folder,
                    train_dataset_name=cfg.DATASETS.TRAIN[0]
                )
            )
        # instance segmentation
        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
        # panoptic segmentation
        if evaluator_type in [
            "coco_panoptic_seg",
            "ade20k_panoptic_seg",
            "cityscapes_panoptic_seg",
            "mapillary_vistas_panoptic_seg",
        ]:
            if cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON:
                evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
        # COCO
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "coco_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Mapillary Vistas
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        if evaluator_type == "mapillary_vistas_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
            evaluator_list.append(SemSegEvaluator(dataset_name, distributed=True, output_dir=output_folder))
        # Cityscapes
        if evaluator_type == "cityscapes_instance":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesInstanceEvaluator(dataset_name)
        if evaluator_type == "cityscapes_sem_seg":
            assert (
                torch.cuda.device_count() > comm.get_rank()
            ), "CityscapesEvaluator currently do not work with multiple machines."
            return CityscapesSemSegEvaluator(dataset_name)
        if evaluator_type == "cityscapes_panoptic_seg":
            if cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesSemSegEvaluator(dataset_name)) #!!!
            if cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
                assert (
                    torch.cuda.device_count() > comm.get_rank()
                ), "CityscapesEvaluator currently do not work with multiple machines."
                evaluator_list.append(CityscapesInstanceEvaluator(dataset_name))
        # ADE20K
        if evaluator_type == "ade20k_panoptic_seg" and cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON:
            evaluator_list.append(InstanceSegEvaluator(dataset_name, output_dir=output_folder))
        # LVIS
        if evaluator_type == "lvis":
            return LVISEvaluator(dataset_name, output_dir=output_folder)
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        # Semantic segmentation dataset mapper
        if cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_semantic":
            mapper = MaskFormerSemanticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Panoptic segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_panoptic":
            mapper = MaskFormerPanopticDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # Instance segmentation dataset mapper
        elif cfg.INPUT.DATASET_MAPPER_NAME == "mask_former_instance":
            mapper = MaskFormerInstanceDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco instance segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_instance_lsj":
            mapper = COCOInstanceNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        # coco panoptic segmentation lsj new baseline
        elif cfg.INPUT.DATASET_MAPPER_NAME == "coco_panoptic_lsj":
            mapper = COCOPanopticNewBaselineDatasetMapper(cfg, True)
            return build_detection_train_loader(cfg, mapper=mapper)
        else:
            mapper = None
            return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        """
        It now calls :func:`detectron2.solver.build_lr_scheduler`.
        Overwrite it if you'd like a different scheduler.
        """
        return build_lr_scheduler(cfg, optimizer)
    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)
    
    @classmethod
    def build_optimizer(cls, cfg, model):
        weight_decay_norm = cfg.SOLVER.WEIGHT_DECAY_NORM
        weight_decay_embed = cfg.SOLVER.WEIGHT_DECAY_EMBED

        defaults = {}
        defaults["lr"] = cfg.SOLVER.BASE_LR
        defaults["weight_decay"] = cfg.SOLVER.WEIGHT_DECAY

        norm_module_types = (
            torch.nn.BatchNorm1d,
            torch.nn.BatchNorm2d,
            torch.nn.BatchNorm3d,
            torch.nn.SyncBatchNorm,
            # NaiveSyncBatchNorm inherits from BatchNorm2d
            torch.nn.GroupNorm,
            torch.nn.InstanceNorm1d,
            torch.nn.InstanceNorm2d,
            torch.nn.InstanceNorm3d,
            torch.nn.LayerNorm,
            torch.nn.LocalResponseNorm,
        )

        params: List[Dict[str, Any]] = []
        memo: Set[torch.nn.parameter.Parameter] = set()
        for module_name, module in model.named_modules():
            for module_param_name, value in module.named_parameters(recurse=False):
                if not value.requires_grad:
                    continue
                # Avoid duplicating parameters
                if value in memo:
                    continue
                memo.add(value)

                hyperparams = copy.copy(defaults)
                if "backbone" in module_name:
                    hyperparams["lr"] = hyperparams["lr"] * cfg.SOLVER.BACKBONE_MULTIPLIER
                if (
                    "relative_position_bias_table" in module_param_name
                    or "absolute_pos_embed" in module_param_name
                ):
                    print(module_param_name)
                    hyperparams["weight_decay"] = 0.0
                if isinstance(module, norm_module_types):
                    hyperparams["weight_decay"] = weight_decay_norm
                if isinstance(module, torch.nn.Embedding):
                    hyperparams["weight_decay"] = weight_decay_embed
                params.append({"params": [value], **hyperparams})

        def maybe_add_full_model_gradient_clipping(optim):
            # detectron2 doesn't have full model gradient clipping now
            clip_norm_val = cfg.SOLVER.CLIP_GRADIENTS.CLIP_VALUE
            enable = (
                cfg.SOLVER.CLIP_GRADIENTS.ENABLED
                and cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model"
                and clip_norm_val > 0.0
            )

            class FullModelGradientClippingOptimizer(optim):
                def step(self, closure=None):
                    all_params = itertools.chain(*[x["params"] for x in self.param_groups])
                    torch.nn.utils.clip_grad_norm_(all_params, clip_norm_val)
                    super().step(closure=closure)

            return FullModelGradientClippingOptimizer if enable else optim

        optimizer_type = cfg.SOLVER.OPTIMIZER
        if optimizer_type == "SGD":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.SGD)(
                params, cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.MOMENTUM
            )
        elif optimizer_type == "ADAMW":
            optimizer = maybe_add_full_model_gradient_clipping(torch.optim.AdamW)(
                params, cfg.SOLVER.BASE_LR
            )
        else:
            raise NotImplementedError(f"no optimizer type {optimizer_type}")
        if not cfg.SOLVER.CLIP_GRADIENTS.CLIP_TYPE == "full_model":
            optimizer = maybe_add_gradient_clipping(cfg, optimizer)
        return optimizer

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA.
        logger.info("Running inference with test-time augmentation ...")
        model = SemanticSegmentorWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    add_maskformer2_config(cfg)
    add_frozenseg_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    setup_logger(output=cfg.OUTPUT_DIR, distributed_rank=comm.get_rank(), name="frozenSeg",enable_propagation=True)
    return cfg


def main(args):
    cfg = setup(args)
    if args.eval_only:
        model = Trainer.build_model(cfg)

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        frozen_params_exclude_text = 0
        for n, p in model.named_parameters():
            if p.requires_grad:
                continue
            if 'clip_model.token_embedding' in n or 'clip_model.positional_embedding' in n or 'clip_model.transformer' in n or 'clip_model.ln_final' in n or 'clip_model.text_projection' in n:
                continue
            frozen_params_exclude_text += p.numel()    
        print(f"total_params: {total_params}, trainable_params: {trainable_params}, frozen_params: {frozen_params}, frozen_params_exclude_text: {frozen_params_exclude_text}")

        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
           
        res = Trainer.test(cfg, model)
        if cfg.TEST.AUG.ENABLED:
            res.update(Trainer.test_with_TTA(cfg, model))
        if comm.is_main_process():
            verify_results(cfg, res)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
