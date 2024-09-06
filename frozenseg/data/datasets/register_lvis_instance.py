import os
from detectron2.data.datasets.lvis import get_lvis_instances_meta, register_lvis_instances

_PREDEFINED_SPLITS_LVIS_v1 = {
    "openvocab_lvis_v1_train": ("coco/", "lvis/lvis_v1_train.json"),
    "openvocab_lvis_v1_val": ("coco/", "lvis/lvis_v1_val.json"),
    "openvocab_lvis_v1_test_dev": ("coco/", "lvis/lvis_v1_image_info_test_dev.json"),
    "openvocab_lvis_v1_test_challenge": ("coco/", "lvis/lvis_v1_image_info_test_challenge.json"),
}
def register_all_lvis_instance(root):
    for key, (image_root, json_file) in _PREDEFINED_SPLITS_LVIS_v1.items():
        register_lvis_instances(
            key,
            get_lvis_instances_meta(key),
            os.path.join(root, json_file) if "://" not in json_file else json_file,
            os.path.join(root, image_root),
        )

_root = os.getenv("DETECTRON2_DATASETS", "datasets")
register_all_lvis_instance(_root)
