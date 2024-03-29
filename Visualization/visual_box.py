from PIL import Image, ImageDraw
from collections import defaultdict
import os

def draw_bbox_and_save(img_ids, bbox_list, out_dir, datapath):
    # Check if input lists are of the same length
    if len(img_ids) != len(bbox_list):
        raise ValueError('The length of img_ids must match the length of bbox_list')
    
    # Group bounding boxes by image ID
    bbox_grouped = dict()
    for img_id, bbox in zip(img_ids, bbox_list):
        id=str(img_id.item())
        if id not in bbox_grouped:
            bbox_grouped[id]=[bbox]
        else :
            bbox_grouped[id].append(bbox)
    
    # Ensure the output directory exists
    os.makedirs(out_dir, exist_ok=True)
    # Process each unique image ID
    for img_id, bboxes in bbox_grouped.items():
        img_name = '0' * (12 - len(img_id)) + img_id + '.jpg'
        img_path = f'{datapath}/{img_name}'
        with Image.open(img_path) as img:
            draw = ImageDraw.Draw(img)
            
            for bbox in bboxes:
                bbox = list(bbox)
                # Ensure bbox is in the correct format (as a precaution)
                draw.rectangle(bbox, outline='red', width=2)
            
            output_path = f'{out_dir}/{img_id}_box.jpg'
            img.save(output_path)

