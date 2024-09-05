from typing import Tuple
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances
from detectron2.utils.memory import retry_if_cuda_oom
from .modeling.criterion import SetCriterion
from .modeling.matcher import HungarianMatcher
from segment_anything.build_sam import sam_model_registry
from .modeling.transformer_decoder.frozenseg_transformer_decoder import MaskPooling, get_classification_logits
from segment_anything import sam_model_registry
import pickle
VILD_PROMPT = [
    "a photo of a {}.",
    "This is a photo of a {}",
    "There is a {} in the scene",
    "There is the {} in the scene",
    "a photo of a {} in the scene",
    "a photo of a small {}.",
    "a photo of a medium {}.",
    "a photo of a large {}.",
    "This is a photo of a small {}.",
    "This is a photo of a medium {}.",
    "This is a photo of a large {}.",
    "There is a small {} in the scene.",
    "There is a medium {} in the scene.",
    "There is a large {} in the scene.",
]


@META_ARCH_REGISTRY.register()
class FrozenSeg(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        train_metadata,
        test_metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        geometric_ensemble_alpha: float,
        geometric_ensemble_beta: float,
        ensemble_on_valid_mask: bool,
        # FrozenSeg
        sam_name: str,
        mask_pred_alpha: float,
        use_sam_masks: bool,
        recall_on: bool,
        pkl_sam_name: str,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
        """
        super().__init__()
        self.backbone = backbone
        self.sem_seg_head = sem_seg_head
        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.train_metadata = train_metadata
        self.test_metadata = test_metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        self.mask_pooling = MaskPooling()
        self.geometric_ensemble_alpha = geometric_ensemble_alpha
        self.geometric_ensemble_beta = geometric_ensemble_beta
        self.ensemble_on_valid_mask = ensemble_on_valid_mask

        self.train_text_classifier = None
        self.test_text_classifier = None
        self.void_embedding = nn.Embedding(1, backbone.dim_latent) # use this for void

        _, self.train_num_templates, self.train_class_names = self.prepare_class_names_from_metadata(train_metadata, train_metadata)
        self.category_overlapping_mask, self.test_num_templates, self.test_class_names = self.prepare_class_names_from_metadata(test_metadata, train_metadata)
        # sam args
        sam_ckpt_path = {
            'vit_t': './pretrained_checkpoint/mobile_sam.pt',
            'vit_b': './pretrained_checkpoint/sam_vit_b_01ec64.pth',
            'vit_l': './pretrained_checkpoint/sam_vit_l_0b3195.pth',
            'vit_h': './pretrained_checkpoint/sam_vit_h_4b8939.pth',
        }
        self.sam = sam_model_registry[sam_name](checkpoint=sam_ckpt_path[sam_name])
        # freeze SAM
        for name, param in self.sam.named_parameters():
            if 'mask_decoder' in name:
                param.requires_grad = False
            else:
                param.requires_grad = False

        if not self.training:
            sam_mask_for_each_dataset={
                                    "openvocab_ade20k_full_sem_seg_val": 'ade20k_val',
                                    'openvocab_coco_2017_val_panoptic_with_sem_seg':'',
                                    'openvocab_pascal_ctx59_sem_seg_val': 'pc_val',
                                    'openvocab_pascal_ctx459_sem_seg_val': 'pc_val',
                                    'openvocab_pascal21_sem_seg_val':'pas_val',
                                    "bdd10k_val_sem_seg":'bdd_val',
                                    }
            self.sam_mask_dataset_name = sam_mask_for_each_dataset[self.test_metadata.name]
        self.counter = 0
        self.use_sam_masks = use_sam_masks
        self.mask_pred_alpha = mask_pred_alpha
        self.recall_on = recall_on
        self.pkl_sam_name = pkl_sam_name


    def prepare_class_names_from_metadata(self, metadata, train_metadata):
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
        category_overlapping_mask = torch.tensor(
            category_overlapping_list, dtype=torch.long)
        
        def fill_all_templates_ensemble(x_=''):
            res = []
            for x in x_:
                for template in VILD_PROMPT:
                    res.append(template.format(x))
            return res, len(res) // len(VILD_PROMPT)
       
        num_templates = []
        templated_class_names = []
        for x in class_names:
            templated_classes, templated_classes_num = fill_all_templates_ensemble(x)
            templated_class_names += templated_classes
            num_templates.append(templated_classes_num) # how many templates for current classes
        class_names = templated_class_names
        return category_overlapping_mask, num_templates, class_names

    def set_metadata(self, metadata):
        self.test_metadata = metadata
        self.category_overlapping_mask, self.test_num_templates, self.test_class_names = self.prepare_class_names_from_metadata(metadata, self.train_metadata)
        self.test_text_classifier = None
        return

    def get_text_classifier(self):
        if self.training:
            if self.train_text_classifier is None:
                text_classifier = []
                # this is needed to avoid oom, which may happen when num of class is large
                bs = 128
                for idx in range(0, len(self.train_class_names), bs):
                    text_classifier.append(self.backbone.get_text_classifier(self.train_class_names[idx:idx+bs], self.device).detach())
                text_classifier = torch.cat(text_classifier, dim=0)

                # average across templates and normalization.
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(VILD_PROMPT), len(VILD_PROMPT), text_classifier.shape[-1]).mean(1)
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                self.train_text_classifier = text_classifier
            return self.train_text_classifier, self.train_num_templates
        else:
            if self.test_text_classifier is None:
                text_classifier = []
                # this is needed to avoid oom, which may happen when num of class is large
                bs = 128
                for idx in range(0, len(self.test_class_names), bs):
                    text_classifier.append(self.backbone.get_text_classifier(self.test_class_names[idx:idx+bs], self.device).detach())
                text_classifier = torch.cat(text_classifier, dim=0)

                # average across templates and normalization.
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                text_classifier = text_classifier.reshape(text_classifier.shape[0]//len(VILD_PROMPT), len(VILD_PROMPT), text_classifier.shape[-1]).mean(1)
                text_classifier /= text_classifier.norm(dim=-1, keepdim=True)
                self.test_text_classifier = text_classifier
            return self.test_text_classifier, self.test_num_templates

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MASK_FORMER.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MASK_FORMER.CLASS_WEIGHT
        dice_weight = cfg.MODEL.MASK_FORMER.DICE_WEIGHT
        mask_weight = cfg.MODEL.MASK_FORMER.MASK_WEIGHT

        # building criterion
        matcher = HungarianMatcher(
            cost_class=class_weight,
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight, "loss_mask": mask_weight, "loss_dice": dice_weight}

        if deep_supervision:
            dec_layers = cfg.MODEL.MASK_FORMER.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["labels", "masks"]

        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MASK_FORMER.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MASK_FORMER.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MASK_FORMER.IMPORTANCE_SAMPLE_RATIO,
        )
        #sem_seg_postprocess_before_inference: for panoptic and instance 
        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MASK_FORMER.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MASK_FORMER.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MASK_FORMER.TEST.OVERLAP_THRESHOLD,
            "train_metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "test_metadata": MetadataCatalog.get(cfg.DATASETS.TEST[0]),
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MASK_FORMER.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON
                or cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON
                or cfg.MODEL.MASK_FORMER.TEST.RECALL_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON,
            "recall_on": cfg.MODEL.MASK_FORMER.TEST.RECALL_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "geometric_ensemble_alpha": cfg.MODEL.FROZEN_SEG.GEOMETRIC_ENSEMBLE_ALPHA,
            "geometric_ensemble_beta": cfg.MODEL.FROZEN_SEG.GEOMETRIC_ENSEMBLE_BETA,
            "ensemble_on_valid_mask": cfg.MODEL.FROZEN_SEG.ENSEMBLE_ON_VALID_MASK,
            # FrozenSeg
            "sam_name": cfg.MODEL.SAM_NAME,
            "mask_pred_alpha": cfg.TEST.SAM_MASK_PRED_ALPHA,
            'use_sam_masks': cfg.TEST.USE_SAM_MASKS,
            "pkl_sam_name":  cfg.TEST.PKL_SAM_MODEL_NAME
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def preprocess_wo_norm(self, x, resize=(512, 512)):
        x = x.float()
        x = F.interpolate(
              x.unsqueeze(0),
              size=resize,
              mode="bilinear",
              align_corners=False,
            )
        return x[0]

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`.
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, H, W) format.
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:
                * "sem_seg" if semantic_on
                * "panoptic_seg" if panoptic_on
                * "instances" if instance_on
                * "recall_seg" if recall_on
        """
        images = [x["image"].to(self.device) for x in batched_inputs] #raw images 3 1024 1024
        if self.sam is None:
            sam_embedding = None
        else:
            images_sam = [(x-self.sam.pixel_mean)/self.sam.pixel_std for x in images]
        if not self.training:
            images_sam = ImageList.from_tensors(images_sam, self.size_divisibility) 
            images_sam = images_sam.tensor.to(self.device)
            images_sam = torch.stack([self.preprocess_wo_norm(x, resize=(1024,1024)) for x in images_sam], dim=0)
        else:
            images_sam = torch.stack(images_sam, dim=0).to(self.device)
        last_embedding, interm_embeddings = self.sam.image_encoder(images_sam)
        sam_embedding = (last_embedding, interm_embeddings)

        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility) 
        features = self.backbone(images.tensor)

        text_classifier, num_templates = self.get_text_classifier()
        text_classifier = torch.cat([text_classifier, F.normalize(self.void_embedding.weight, dim=-1)], dim=0)
        features['text_classifier'] = text_classifier
        features['num_templates'] = num_templates
        features['sam_embedding'] = sam_embedding
        features['sam'] = self.sam
        outputs = self.sem_seg_head(features) 

        if self.training:
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None
            # bipartite matching-based loss
            losses = self.criterion(outputs, targets)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            clip_feature = features["clip_vis_dense"]
            mask_for_pooling = F.interpolate(mask_pred_results, size=clip_feature.shape[-2:],
                                                mode='bilinear', align_corners=False)

            if "convnext" in self.backbone.model_name.lower():
                pooled_clip_feature = self.mask_pooling(clip_feature, mask_for_pooling) #mask>0
                pooled_clip_feature = self.backbone.visual_prediction_forward(pooled_clip_feature)
            elif "rn" in self.backbone.model_name.lower():
                pooled_clip_feature = self.backbone.visual_prediction_forward(clip_feature, mask_for_pooling)
            else:
                raise NotImplementedError
            
            out_vocab_cls_results = get_classification_logits(pooled_clip_feature, text_classifier, self.backbone.clip_model.logit_scale, num_templates)
            in_vocab_cls_results = mask_cls_results[..., :-1] # remove void
            out_vocab_cls_results = out_vocab_cls_results[..., :-1] # remove void
            out_vocab_cls_probs = out_vocab_cls_results.softmax(-1)
            in_vocab_cls_results = in_vocab_cls_results.softmax(-1)
            category_overlapping_mask = self.category_overlapping_mask.to(self.device)
           
            if self.ensemble_on_valid_mask:
                # Only include out_vocab cls results on masks with valid pixels
                # We empirically find that this is important to obtain reasonable AP/mIOU score with ResNet CLIP models
                valid_masking = (mask_for_pooling > 0).to(mask_for_pooling).sum(-1).sum(-1) > 0
                valid_masking = valid_masking.to(in_vocab_cls_results.dtype).unsqueeze(-1)
                alpha = torch.ones_like(in_vocab_cls_results) * self.geometric_ensemble_alpha
                beta = torch.ones_like(in_vocab_cls_results) * self.geometric_ensemble_beta
                alpha = alpha * valid_masking
                beta = beta * valid_masking
            else:
                alpha = self.geometric_ensemble_alpha
                beta = self.geometric_ensemble_beta

            cls_logits_seen = (
                (in_vocab_cls_results ** (1 - alpha) * out_vocab_cls_probs**alpha).log()
                * category_overlapping_mask
            )
            cls_logits_unseen = (
                (in_vocab_cls_results ** (1 - beta) * out_vocab_cls_probs**beta).log()
                * (1 - category_overlapping_mask)
            )
            cls_results = cls_logits_seen + cls_logits_unseen

            # This is used to filtering void predictions. 
            is_void_prob = F.softmax(mask_cls_results, dim=-1)[..., -1:]
            mask_cls_probs = torch.cat([
                cls_results.softmax(-1) * (1.0 - is_void_prob),
                is_void_prob], dim=-1)
            mask_cls_results = torch.log(mask_cls_probs + 1e-8)
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
          
            del outputs
            processed_results = []
            for mask_cls_result, mask_pred_result, input_per_image, image_size in zip(
                mask_cls_results, mask_pred_results, batched_inputs, images.image_sizes
            ): 
                height = input_per_image.get("height", image_size[0])
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})
                if self.sem_seg_postprocess_before_inference: # panoptic on
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
                    mask_cls_result = mask_cls_result.to(mask_pred_result)

                if self.use_sam_masks:
                    assert not self.training
                    img_id = input_per_image.get("image_id", None)
                    if img_id is None:
                        filename = input_per_image.get('file_name', None)
                        if filename is None:
                            assert NameError, 'No image_id or file_name in input_per_image'
                        elif filename is not None:
                            img_id = filename.split('/')[-1].split('.')[0]
                    with open(f'output/SAM_masks_pred/{self.pkl_sam_name}_{self.sam_mask_dataset_name}/{img_id}.pkl', 'rb') as f:
                        everything_mask = pickle.load(f)
                    
                    sam_mask_pred = [torch.from_numpy(mask['preds']).to(mask_cls_result.device) for mask in everything_mask]
                    if len(sam_mask_pred) == 0:
                        sam_mask_pred = None
                        sam_cls_results = None
                        sam_iou_scores = None
                    else:
                        sam_mask_pred = torch.stack(sam_mask_pred, dim=0) # M, H, W
                        sam_iou_scores = [torch.tensor(mask['predicted_iou']).sigmoid().to(mask_cls_result.device) for mask in everything_mask]
                        sam_iou_scores = torch.stack(sam_iou_scores, dim=0) # M, 1
                        sam_mask_for_pooling_clip = F.interpolate(sam_mask_pred.unsqueeze(0), size=clip_feature.shape[-2:], mode="nearest")
                        sam_mask_for_pooling_clip = ImageList.from_tensors([sam_mask_for_pooling_clip[0]], self.size_divisibility)
                        if 'convnext' in self.backbone.model_name.lower():
                            sam_pooled_clip_feature = self.mask_pooling(clip_feature, sam_mask_for_pooling_clip.tensor.to(torch.float32))
                            sam_pooled_clip_feature = self.backbone.visual_prediction_forward(sam_pooled_clip_feature)
                            sam_cls_results = get_classification_logits(sam_pooled_clip_feature, text_classifier, self.backbone.clip_model.logit_scale, num_templates)
                        elif "rn" in self.backbone.model_name.lower():
                            sam_pooled_clip_feature = self.backbone.visual_prediction_forward(clip_feature, sam_mask_for_pooling_clip.tensor.to(torch.float32))
                            sam_cls_results = get_classification_logits(sam_pooled_clip_feature, text_classifier, self.backbone.clip_model.logit_scale, num_templates) # have nan
                        else:
                            print("not support")
                            raise NotImplementedError
                        sam_mask_pred = sam_mask_pred.to(mask_pred_result)
                        sam_cls_results = sam_cls_results.to(mask_cls_result)
                    if not self.sem_seg_postprocess_before_inference: #### For semantic segmentation and recall inference
                        mask_pred_result = mask_pred_result[:, :image_size[0], :image_size[1]]
                    del everything_mask
                #######################################################

                if self.recall_on and not self.use_sam_masks:
                    res = retry_if_cuda_oom(self.recall_inference)(mask_pred_result)
                    processed_results[-1]["recall_seg"] = res
                elif self.recall_on and self.use_sam_masks:
                    if sam_mask_pred.shape[-2:] != mask_pred_result.shape[-2:]: 
                        sam_mask_pred = F.interpolate(sam_mask_pred.unsqueeze(0), size=mask_pred_result.shape[-2:], mode="bilinear", align_corners=False)[0]
                    res = retry_if_cuda_oom(self.recall_inference_with_everything)(mask_pred_result, sam_mask_pred)
                    res = retry_if_cuda_oom(sem_seg_postprocess)(res, image_size, height, width)
                    processed_results[-1]["recall_seg"] = res

                if self.semantic_on:
                    if self.use_sam_masks:
                        if sam_mask_pred.shape[-2:] != mask_pred_result.shape[-2:]: 
                            sam_mask_pred = F.interpolate(sam_mask_pred.unsqueeze(0), size=mask_pred_result.shape[-2:], mode="bilinear", align_corners=False)[0]
                        res = self.geo_with_sam_inference(mask_cls_result, mask_pred_result, sam_mask_pred, sam_cls_results, category_overlapping_mask)
                    else:
                        res = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference : # for sem seg
                        res = retry_if_cuda_oom(sem_seg_postprocess)(res, image_size, height, width)
                    processed_results[-1]["sem_seg"] = res
                
                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r

                # instance segmentation inference
                if self.instance_on:
                    instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["instances"] = instance_r
            return processed_results

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                }
            )
        return new_targets
    

    def resize_feat(self, x, resize_shape):
        x = F.interpolate(
              x,
              size=(resize_shape[0], resize_shape[1]),
              mode="bilinear",
              align_corners=False,
            )
        return x
    
    def recall_inference(self, mask_pred):
        """
        Return: (q, h, w)
        """
        return mask_pred

    def recall_inference_with_everything(self, mask_pred, sam_mask_pred):
        """
        Return: (q, h, w)
        """
        if sam_mask_pred is None:
            return self.recall_inference(mask_pred)
        return torch.cat([mask_pred, sam_mask_pred], dim=0)

    def semantic_inference(self, mask_cls, mask_pred):
        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
        mask_pred = mask_pred.sigmoid() 
        semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
        return semseg

 
    def panoptic_inference(self, mask_cls, mask_pred):
        scores, labels = F.softmax(mask_cls, dim=-1).max(-1)
        mask_pred = mask_pred.sigmoid()
        num_classes = len(self.test_metadata.stuff_classes)
        keep = labels.ne(num_classes) & (scores > self.object_mask_threshold)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_mask_cls = mask_cls[keep]
        cur_mask_cls = cur_mask_cls[:, :-1]
        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0
        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]): # through all mask queries
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.test_metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= 0.5).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= 0.5)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold: # 0.8 for coco, 0 for else.
                        continue
                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1
                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id
                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )
            return panoptic_seg, segments_info
        


    def geo_with_sam_inference(self, mask_cls, mask_pred, sam_mask_pred, sam_mask_cls, category_overlapping_mask=None):
        if sam_mask_cls is None:
            return self.semantic_inference(mask_cls, mask_pred)

        mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1] 
        sam_mask_cls = F.softmax(sam_mask_cls, dim=-1)[..., :-1].squeeze(0) # M, C
        mask_pred = mask_pred.sigmoid()
        sam_mask_pred = sam_mask_pred.sigmoid()

        alpha = self.mask_pred_alpha
        beta = 0.
        in_mask_cls = mask_cls * category_overlapping_mask.view(1,-1) # (q,c)
        out_mask_cls = mask_cls * (1 - category_overlapping_mask).view(1,-1)  # (q,c) # 0
        out_mask_cls = out_mask_cls * (1-alpha)
        in_mask_cls = in_mask_cls * (1-beta)

        ## MaskEnsemble
        in_sam_mask_cls = sam_mask_cls * category_overlapping_mask.view(1,-1) # (m,c)
        out_sam_mask_cls = sam_mask_cls * (1 - category_overlapping_mask).view(1,-1) # (m,c)
        
        sam_mask_left = out_sam_mask_cls.max(dim=1).values>0.5 # m
        out_sam_mask_cls = out_sam_mask_cls[sam_mask_left] # m', c
        out_sam_mask_pred = sam_mask_pred[sam_mask_left] # m', h, w
        
        out_sam_mask_cls = out_sam_mask_cls * alpha
        in_sam_mask_cls = in_sam_mask_cls * beta

        in_semseg = torch.einsum("qc,qhw->chw", in_mask_cls, mask_pred)
        out_semseg = torch.einsum("qc,qhw->chw", out_mask_cls, mask_pred)
        if not out_sam_mask_cls.shape[0]==0:
            out_sam_semseg = torch.einsum("mc,mhw->chw", out_sam_mask_cls, out_sam_mask_pred)
        else:
            out_sam_semseg = torch.zeros_like(out_semseg)
        out_semseg = out_semseg + out_sam_semseg 
        semseg = in_semseg + out_semseg
        
        return semseg
    

    def instance_inference(self, mask_cls, mask_pred):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]

        # [Q, K]
        scores = F.softmax(mask_cls, dim=-1)[:, :-1]
        # if this is panoptic segmentation
        if self.panoptic_on:
            num_classes = len(self.test_metadata.stuff_classes)
        else:
            num_classes = len(self.test_metadata.thing_classes)
        labels = torch.arange(num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        # scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.num_queries, sorted=False)
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)
        labels_per_image = labels[topk_indices]

        topk_indices = topk_indices // num_classes
        # mask_pred = mask_pred.unsqueeze(1).repeat(1, self.sem_seg_head.num_classes, 1).flatten(0, 1)
        mask_pred = mask_pred[topk_indices]

        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.test_metadata.thing_dataset_id_to_contiguous_id.values()

            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        result.pred_boxes = Boxes(torch.zeros(mask_pred.size(0), 4))
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result
