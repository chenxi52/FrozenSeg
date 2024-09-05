# FrozenSeg: Harmonizing Frozen Foundation Models for Open-Vocabulary Segmentation

>[**FrozenSeg: Harmonizing Frozen Foundation Models for Open-Vocabulary Segmentation**]()
XXXX

## Abstract

>Open-vocabulary segmentation is challenging, with the need of segmenting and recognizing objects for an open set of categories in unconstrained environments. Building on the success of powerful vision-language (ViL) foundation models like CLIP, recent efforts sought to harness their zero-short capabilities to recognize unseen categories. Despite demonstrating strong performances, they still face a fundamental challenge of generating precise mask proposals for unseen categories and scenarios, resulting in inferior segmentation performance eventually. To address this, we introduce a novel approach, FrozenSeg, designed to integrate spatial knowledge from a localization foundation model (e.g., SAM) and semantic knowledge extracted from a ViL model (e.g., CLIP), in a synergistic framework. Taking the ViL model’s visual encoder as the feature backbone, we inject the space-aware feature into learnable query and CLIP feature in the transformer decoder. In addition, we devise a mask proposal ensemble strategy for further improving the recall rate and mask quality. To fully exploit pre-trained knowledge while minimizing training overhead, we freeze both foundation models, focusing optimization efforts solely on a light transformer decoder for mask proposal generation – the performance bottleneck. Extensive experiments show that FrozenSeg advances state-of-the-art results across various segmentation benchmarks, trained exclusively on COCO panoptic data and tested in a zero-shot manner.

## Updates
- **XXX** Code is avaliable.

## Dependencies and Installation
See [installation instructions](INSTALL.md).

## Getting Started
See [Preparing Datasets for FC-CLIP](datasets/README.md).

See [Getting Started with  FC-CLIP](GETTING_STARTED.md).


## Models
<table>
<thead>
  <tr>
    <th align="center"></th>
    <th align="center" style="text-align:center" colspan="3"><a href="logs/testing/ade20k.log">ADE20K(A-150)</th>
    <th align="center" style="text-align:center" colspan="3"><a href="logs/testing/cityscapes.log">Cityscapes</th>
    <th align="center" style="text-align:center" colspan="2"><a href="logs/testing/mapillary_vistas.log">Mapillary Vistas</th>
    <th align="center" style="text-align:center"><a href="logs/testing/a-847.log">ADE20K-Full <br> (A-847)</th>
    <th align="center" style="text-align:center"><a href="logs/testing/pc-59.log">Pascal Context 59 <br> (PC-59)</th>
    <th align="center" style="text-align:center"><a href="logs/testing/pc-459.log">Pascal Context 459 <br> (PC-459)</th>
    <th align="center" style="text-align:center"><a href="logs/testing/pc-21.log">Pascal VOC 21 <br> (PAS-21) </th>
    <th align="center" style="text-align:center"><a href="logs/testing/pc-20.log">Pascal VOC 20 <br> (PAS-20) </th>
    <th align="center" style="text-align:center" colspan="3"><a href="logs/testing/coco.log">COCO <br> (training dataset)</th>
    <th align="center" style="text-align:center">download </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td align="center"></td>
    <td align="center">PQ</td>
    <td align="center">mAP</td>
    <td align="center">mIoU</td>
    <td align="center">PQ</td>
    <td align="center">mAP</td>
    <td align="center">mIoU</td>
    <td align="center">PQ</td>
    <td align="center">mIoU</td>
    <td align="center">mIoU</td>
    <td align="center">mIoU</td>
    <td align="center">mIoU</td>
    <td align="center">mIoU</td>
    <td align="center">mIoU</td>
    <td align="center">PQ</td>
    <td align="center">mAP</td>
    <td align="center">mIoU</td>
  </tr>
    <td align="center"><a href="configs/coco/panoptic-segmentation/fcclip/fcclip_convnext_large_eval_ade20k_r50.yaml"> FC-CLIP (ResNet50) </a></td>
    <td align="center">17.9</td>
    <td align="center">9.5</td>
    <td align="center">23.3</td>
    <td align="center">40.3</td>
    <td align="center">21.6</td>
    <td align="center">53.2</td>
    <td align="center">15.9</td>
    <td align="center">24.4</td>
    <td align="center">7.1</td>
    <td align="center">50.5</td>
    <td align="center">12.9</td>
    <td align="center">75.9</td>
    <td align="center">89.5</td>
    <td align="center">50.7</td>
    <td align="center">40.7</td>
    <td align="center">58.8</td>
    <td align="center"><a href="https://drive.google.com/file/d/1tcB-8FNON-LwckXQbUyKcBA2G7TU65Zh/view?usp=sharing"> checkpoint </a></td>
  </tr>
  <tr>
    <td align="center"><a href="configs/coco/panoptic-segmentation/fcclip/fclip_convnext_large_eval_ade20k_r101.yaml"> FC-CLIP (ResNet101) </a></td>
    <td align="center">19.1</td>
    <td align="center">10.2</td>
    <td align="center">24.0</td>
    <td align="center">40.9</td>
    <td align="center">24.1</td>
    <td align="center">53.9</td>
    <td align="center">16.7</td>
    <td align="center">23.2</td>
    <td align="center">7.7</td>
    <td align="center">48.9</td>
    <td align="center">12.3</td>
    <td align="center">77.6</td>
    <td align="center">91.3</td>
    <td align="center">51.4</td>
    <td align="center">41.6</td>
    <td align="center">58.9</td>
    <td align="center"><a href="https://drive.google.com/file/d/1P0mdgftReWzVbPQ0O0CSBfW3krHhTOj0/view?usp=sharing"> checkpoint </a></td>
  </tr>
</tbody>
</table>


## Model evaluation

## <a name="Citing FrozenSeg"></a>Citing  FrozenSeg

If you use FrozenSeg in your research, please use the following BibTeX entry.

```BibTeX
@inproceedings{yu2023fcclip,
  title={Convolutions Die Hard: Open-Vocabulary Segmentation with Single Frozen Convolutional CLIP},
  author={Qihang Yu and Ju He and Xueqing Deng and Xiaohui Shen and Liang-Chieh Chen},
  booktitle={NeurIPS},
  year={2023}
}
```