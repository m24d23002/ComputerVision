import cv2
import numpy as np
import os
from pycocotools.coco import COCO

image_dir = './COCO/val2017'
annotation_file = './COCO/annotations/instances_val2017.json'

def get_category_mappings(coco):
    cat_ids = sorted(coco.getCatIds())
    id_to_index = {cat_id: i for i, cat_id in enumerate(cat_ids)}
    index_to_id = {i: cat_id for i, cat_id in enumerate(cat_ids)}
    return id_to_index, index_to_id

def load_coco_data(max_boxes=10):
    image_size = (224, 224)
    coco = COCO(annotation_file)
    image_ids = coco.getImgIds()[:500]  # Limit for faster training

    id_to_index, _ = get_category_mappings(coco)

    images = []
    class_targets = []
    bbox_targets = []

    for image_id in image_ids:
        image_info = coco.loadImgs(image_id)[0]
        image_path = os.path.join(image_dir, image_info['file_name'])

        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image at {image_path}")
            continue

        image = cv2.resize(image, image_size)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        images.append(image)

        ann_ids = coco.getAnnIds(imgIds=image_id)
        anns = coco.loadAnns(ann_ids)

        label_vec = np.zeros(80)
        bbox_vec = np.zeros((max_boxes, 4))  # Fixed size for bounding boxes

        for i, ann in enumerate(anns[:max_boxes]): # Limit to max_boxes
            cat_id = ann['category_id']
            if cat_id in id_to_index:
                label_vec[id_to_index[cat_id]] = 1
            
            # Normalize bounding boxes relative to resized image dimensions
            bbox = ann['bbox']
            bbox_vec[i] = [bbox[0] / image_info['width'] * image_size[0],
                           bbox[1] / image_info['height'] * image_size[1],
                           bbox[2] / image_info['width'] * image_size[0],
                           bbox[3] / image_info['height'] * image_size[1]]

        class_targets.append(label_vec)
        bbox_targets.append(bbox_vec)

    images = np.array(images, dtype=np.float32) / 255.0
    class_targets = np.array(class_targets, dtype=np.float32)
    bbox_targets = np.array(bbox_targets, dtype=np.float32)

    return images, class_targets, bbox_targets