'''
Save SAM mask predictions
'''
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
import torch.multiprocessing as mp
import pickle
from tqdm import tqdm
import torch
import cv2
import os
import json
import argparse
import numpy as np
img_anno = {
            'ade20k_val':['ADEChallengeData2016/images/validation', 'ADEChallengeData2016/ade20k_panoptic_val.json'],
            'pc_val': ['pascal_ctx_d2/images/validation','' ],
            'pas_val':['pascal_voc_d2/images/validation',''],
            }
sam_checkpoint_dict = {
            'vit_b': 'pretrained_checkpoint/sam_vit_b_01ec64.pth',
            'vit_h': 'pretrained_checkpoint/sam_vit_h_4b8939.pth',
            'vit_l': 'pretrained_checkpoint/sam_vit_l_0b3195.pth',
            'vit_t': 'pretrained_checkpoint/mobile_sam.pt'
            }

def process_images(args, gpu, data_chunk, save_path, if_parallel):
    def to_parallel(if_parallel):
        sam_checkpoint = sam_checkpoint_dict[args.sam_model]
        sam = sam_model_registry[args.sam_model](checkpoint=sam_checkpoint)
        if not if_parallel:
            torch.cuda.set_device(gpu)
            sam = sam.cuda()
        else:
            sam = sam.cuda()
            sam =  torch.nn.DataParallel(sam)
            sam = sam.module
        return sam
    
    sam = to_parallel(if_parallel)
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        pred_iou_thresh=0.8,
        stability_score_thresh=0.7,
        crop_n_layers=0,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  
        output_mode='coco_rle'
    )
    # Process each image
    for image_info in tqdm(data_chunk):
        if isinstance(image_info, dict):
            if 'coco_url' in image_info:
                coco_url = image_info['coco_url']
                file_name = coco_url.split('/')[-1].split('.')[0] + '.jpg'
            elif 'file_name' in image_info:
                file_name = image_info['file_name'].split('.')[0] + '.jpg'
            file_path = os.path.join(dataset_path,img_anno[args.data_name][0])
        else: 
            assert isinstance(image_info, str)
            file_name = image_info.split('.')[0] + '.jpg'
            file_path = os.path.join(dataset_path,img_anno[args.data_name][0])
        image_path = f'{file_path}/{file_name}'
        try:
            id =file_name.split('.')[0]
            id = id.replace('/','_')
            savepath = f'{save_path}/{id}.pkl' 
            if not os.path.exists(savepath):
                image = cv2.imread(image_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                everything_mask = mask_generator.generate(image) 
                everything_mask = sorted(everything_mask, key=lambda x: x['area'], reverse=True)
                if len(everything_mask) >50:
                    everything_mask = everything_mask[:50]
                with open(savepath, 'wb') as f:
                    pickle.dump(everything_mask, f)
        except Exception as e:
            print(f"Failed to load or convert image at {image_path}. Error: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default='pas_val')
    parser.add_argument('--sam_model', type=str, default='vit_h')
    argss = parser.parse_args()
    gpus = os.getenv("CUDA_VISIBLE_DEVICES", "")
    dataset_path = os.getenv("DETECTRON2_DATASETS", "/users/cx_xchen/DATASETS/")

    num_gpus = len([x.strip() for x in gpus.split(",") if x.strip().isdigit()])
    print(f"Using {num_gpus} GPUs")
    # File paths
    if img_anno[argss.data_name][1] != '':
        json_file_path = os.path.join(dataset_path, img_anno[argss.data_name][1])
        # Load data
        with open(json_file_path, 'r') as file:
            data = json.load(file)
        # Split data into chunks for each GPU
        data_chunks = np.array_split(data['images'], num_gpus)
    else:
        image_dir = os.path.join(dataset_path, img_anno[argss.data_name][0])
        image_files = os.listdir(image_dir)
        data_chunks = np.array_split(image_files, num_gpus)
    # Create processes
    save_path = f'output/SAM_masks_pred/{argss.sam_model}_{argss.data_name}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    processes = []
    for gpu in range(num_gpus):
        p = mp.Process(target=process_images, args=(argss, gpu, data_chunks[gpu],save_path, False))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
