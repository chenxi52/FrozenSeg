import os 
import sys

# os.system("pip install gdown")
# os.system("pip install imutils")
# os.system("pip install gradio_client==0.2.7")
# os.system("python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'")
# os.system("pip install git+https://github.com/cocodataset/panopticapi.git")
# os.system("python fcclip/modeling/pixel_decoder/ops/setup.py build install")

import gradio as gr
from detectron2.utils.logger import setup_logger
from contextlib import ExitStack
import numpy as np
import cv2
import torch
import itertools
from detectron2.config import get_cfg
from detectron2.utils.visualizer import ColorMode, random_color
from detectron2.data import MetadataCatalog

from frozenseg import add_maskformer2_config, add_frozenseg_config
from demo.predictor import DefaultPredictor, OpenVocabVisualizer
from PIL import Image
import json

setup_logger()
logger = setup_logger(name="fcclip")
cfg = get_cfg()
cfg.MODEL.DEVICE='cuda'
add_maskformer2_config(cfg)
add_frozenseg_config(cfg)
cfg.merge_from_file("configs/coco/frozenseg/convnext_large_eval_ade20k.yaml")
# os.system("gdown 1-91PIns86vyNaL3CzMmDD39zKGnPMtvj")
cfg.MODEL.WEIGHTS = './modified_model.pth'
cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = False
cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = True
predictor = DefaultPredictor(cfg)


title = "FrozenSeg"
article = "<p style='text-align: center'><a href='' target='_blank'>FrozenSeg</a> | <a href='' target='_blank'>Github Repo</a></p>"

examples = [
    [
        "demo/examples/ADE_val_00000001.jpg",
        "",
        ["ADE (150 categories)"],
    ],
    [
        "demo/examples/frankfurt_000000_005898_leftImg8bit.png",
        "",
        ["Cityscapes (19 categories)"],
    ]
]


coco_metadata = MetadataCatalog.get("openvocab_coco_2017_val_panoptic_with_sem_seg")
ade20k_metadata = MetadataCatalog.get("openvocab_ade20k_panoptic_val")
cityscapes_metadata = MetadataCatalog.get("openvocab_cityscapes_fine_panoptic_val")
lvis_classes = open("./frozenseg/data/datasets/lvis_1203_with_prompt_eng.txt", 'r').read().splitlines()
lvis_classes = [x[x.find(':')+1:] for x in lvis_classes]
lvis_colors = list(
    itertools.islice(itertools.cycle(coco_metadata.stuff_colors), len(lvis_classes))
)
# rerrange to thing_classes, stuff_classes
coco_thing_classes = coco_metadata.thing_classes
coco_stuff_classes = [x for x in coco_metadata.stuff_classes if x not in coco_thing_classes]
coco_thing_colors = coco_metadata.thing_colors
coco_stuff_colors = [x for x in coco_metadata.stuff_colors if x not in coco_thing_colors]
ade20k_thing_classes = ade20k_metadata.thing_classes
ade20k_stuff_classes = [x for x in ade20k_metadata.stuff_classes if x not in ade20k_thing_classes]
ade20k_thing_colors = ade20k_metadata.thing_colors
ade20k_stuff_colors = [x for x in ade20k_metadata.stuff_colors if x not in ade20k_thing_colors]
cityscapes_stuff_classes = cityscapes_metadata.stuff_classes
cityscapes_stuff_color = cityscapes_metadata.stuff_colors
cityscapes_thing_classes = cityscapes_metadata.thing_classes
cityscapes_thing_color = cityscapes_metadata.thing_colors

def build_demo_classes_and_metadata(vocab, label_list):
    extra_classes = []

    if vocab:
        for words in vocab.split(";"):
            extra_classes.append(words)
    extra_colors = [random_color(rgb=True, maximum=1) for _ in range(len(extra_classes))]
    print("extra_classes:", extra_classes)
    demo_thing_classes = extra_classes
    demo_stuff_classes = []
    demo_thing_colors = extra_colors
    demo_stuff_colors = []

    if any("COCO" in label for label in label_list):
        demo_thing_classes += coco_thing_classes
        demo_stuff_classes += coco_stuff_classes
        demo_thing_colors += coco_thing_colors
        demo_stuff_colors += coco_stuff_colors
    if any("ADE" in label for label in label_list):
        demo_thing_classes += ade20k_thing_classes
        demo_stuff_classes += ade20k_stuff_classes
        demo_thing_colors += ade20k_thing_colors
        demo_stuff_colors += ade20k_stuff_colors
    if any("LVIS" in label for label in label_list):
        demo_thing_classes += lvis_classes
        demo_thing_colors += lvis_colors
    if any("Cityscapes" in label for label in label_list):
        demo_thing_classes += cityscapes_thing_classes
        demo_thing_colors += cityscapes_thing_color
        demo_stuff_classes += cityscapes_stuff_classes
        demo_stuff_colors += cityscapes_stuff_color
    

    MetadataCatalog.pop("fcclip_demo_metadata", None)
    demo_metadata = MetadataCatalog.get("fcclip_demo_metadata")
    demo_metadata.thing_classes = demo_thing_classes
    demo_metadata.stuff_classes = demo_thing_classes + demo_stuff_classes
    demo_metadata.thing_colors = demo_thing_colors
    demo_metadata.stuff_colors = demo_thing_colors + demo_stuff_colors
    demo_metadata.stuff_dataset_id_to_contiguous_id = {
        idx: idx for idx in range(len(demo_metadata.stuff_classes))
    }
    demo_metadata.thing_dataset_id_to_contiguous_id = {
        idx: idx for idx in range(len(demo_metadata.thing_classes))
    }
    demo_classes = demo_thing_classes + demo_stuff_classes
    return demo_classes, demo_metadata


def inference(image_path, vocab, label_list):
    logger.info("building class names")
    vocab = vocab.replace(", ", ",").replace("; ", ";")
    demo_classes, demo_metadata = build_demo_classes_and_metadata(vocab, label_list)
    predictor.set_metadata(demo_metadata)
    im = cv2.imread(image_path)
    outputs = predictor(im)
    v = OpenVocabVisualizer(im[:, :, ::-1], demo_metadata, instance_mode=ColorMode.IMAGE)
    panoptic_result = v.draw_panoptic_seg(outputs["panoptic_seg"][0].to("cpu"), outputs["panoptic_seg"][1]).get_image()
    return Image.fromarray(np.uint8(panoptic_result)).convert('RGB')

    
with gr.Blocks(title=title,
                css="""
               #submit {background: #3498db; color: white; border: none; padding: 10px 20px; border-radius: 5px;width: 20%;margin: 0 auto; display: block;}
                """
            ) as demo:
    gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>" + title + "</h1>")
    input_components = []
    output_components = []

    with gr.Row():
        output_image_gr = gr.Image(label="Panoptic Segmentation Output", type="pil")
        output_components.append(output_image_gr)

    with gr.Row().style(equal_height=True):
        with gr.Column(scale=3, variant="panel") as input_component_column:
            input_image_gr = gr.Image(type="filepath", label="Input Image")
            extra_vocab_gr = gr.Textbox(label="Extra Vocabulary (separated by ;)", placeholder="house;sky")
            category_list_gr = gr.CheckboxGroup(
                choices=["COCO (133 categories)", "ADE (150 categories)", "LVIS (1203 categories)", "Cityscapes (19 categories)"],
                label="Category to use",
            )
            input_components.extend([input_image_gr, extra_vocab_gr, category_list_gr])

        with gr.Column(scale=2):
            examples_handler = gr.Examples(
                examples=examples,
                inputs=[c for c in input_components if not isinstance(c, gr.State)],
                outputs=[c for c in output_components if not isinstance(c, gr.State)],
                fn=inference,
                cache_examples=torch.cuda.is_available(),
                examples_per_page=5,
            )
            with gr.Row():
                clear_btn = gr.Button("Clear")
                submit_btn = gr.Button("Submit", variant="primary")

    gr.Markdown(article)

    submit_btn.click(
        inference,
        input_components,
        output_components,
        api_name="predict",
        scroll_to_output=True,
    )

    clear_btn.click(
        None,
        [],
        (input_components + output_components + [input_component_column]),
        _js=f"""() => {json.dumps(
                    [component.cleared_value if hasattr(component, "cleared_value") else None
                     for component in input_components + output_components] + (
                        [gr.Column.update(visible=True)]
                    )
                    + ([gr.Column.update(visible=False)])
                )}
                """,
    )

demo.launch(server_port=7881)
