from pathlib import Path
import numpy as np
import torch
import torch.utils.data
from pycocotools import mask as coco_mask
from torchvision.datasets.vision import VisionDataset
from PIL import Image
import os
import os.path
import tqdm
from io import BytesIO
import torch.distributed as dist

# from util.misc import get_local_rank, get_local_size
# import datasets.transforms as T

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')
class TvCocoDetection(VisionDataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.
    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None,
                 cache_mode=False, local_rank=0, local_size=1):
        super(TvCocoDetection, self).__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.cache_mode = cache_mode
        self.local_rank = local_rank
        self.local_size = local_size
        if cache_mode:
            self.cache = {}
            self.cache_images()

    def cache_images(self):
        self.cache = {}
        for index, img_id in zip(tqdm.trange(len(self.ids)), self.ids):
            if index % self.local_size != self.local_rank:
                continue
            path = self.coco.loadImgs(img_id)[0]['file_name']
            
            
            with open(os.path.join(self.root, path), 'rb') as f:
                self.cache[path] = f.read()

    def get_image(self, path):
        if self.cache_mode:
            if path not in self.cache.keys():
                with open(os.path.join(self.root, path), 'rb') as f:
                    self.cache[path] = f.read()
            return Image.open(BytesIO(self.cache[path])).convert('RGB')

        return Image.open(os.path.join(self.root, path)).convert('RGB')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target, original_file_name). target is the object returned by ``coco.loadAnns``.
        """
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        # print(target)
        # print(type(img_id))

        if type(img_id) != str:
            if _isArrayLike(img_id):
                path = [coco.imgs[id] for id in img_id]
            elif type(img_id) == int:
                path = [coco.imgs[img_id]]
        else:
            path = [coco.imgs[img_id]]
            

        path = path[0]['file_name']
        # print(path)
        # path = coco.loadImgs(img_id)[0]['file_name']

        img = self.get_image(path)
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target, path

    def __len__(self):
        return len(self.ids)
class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        # self.return_masks = return_masks
        self.image_id_temp = 0

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        # print(type(image_id) == str)
        if type(image_id) == str:
            image_id = self.image_id_temp
            self.image_id_temp += 1
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]
        
        id_corresponding=[obj["image_id"] for obj in anno]
      
        boxes = [obj["bbox"] for obj in anno]
        
        # guard against no boxes via resizing
        boxes, keep = preprocess_xywh_boxes(boxes, h, w)
        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        # if self.return_masks:
        #     segmentations = [obj["segmentation"] for obj in anno]
        #     masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        classes = classes[keep]
        id_corresponding = torch.tensor(id_corresponding)
        id_corresponding=id_corresponding[keep]
        # if self.return_masks:
        #     masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        # if self.return_masks:
        #     target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints


        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        target["id_corresponding"]=id_corresponding
        return image, target
class CocoDetection(TvCocoDetection):
    def __init__(self, img_folder, ann_file, return_masks, cache_mode=False, local_rank=0, local_size=1, no_cats=False, filter_pct=-1, bdd=False):
        super(CocoDetection, self).__init__(img_folder, ann_file,
                                            cache_mode=cache_mode, local_rank=local_rank, local_size=local_size)
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.no_cats = no_cats
        self.bdd = bdd
        if filter_pct > 0:
            num_keep = float(len(self.ids))*filter_pct
            self.ids = np.random.choice(self.ids, size=round(num_keep), replace=False).tolist()

    def __getitem__(self, idx):
        img, target, original_file_name = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self.no_cats:
            target['labels'][:] = 1
        if self.bdd:
            target['labels'] -= 1
        return img, target, original_file_name

def preprocess_xywh_boxes(boxes, h, w):
    boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4) # -w allow for any number of boxes and 4 corresponding to [x,y,]
    boxes[:, 2:] += boxes[:, :2]
    boxes[:, 0::2].clamp_(min=0, max=w)
    boxes[:, 1::2].clamp_(min=0, max=h)
    keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
    boxes = boxes[keep]
    return boxes, keep
def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True
def get_local_size():
    if not is_dist_avail_and_initialized():
        return 1
    return int(os.environ['LOCAL_SIZE'])


def get_local_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return int(os.environ['LOCAL_RANK'])

def build(image_set, args):
    root = Path(args.coco_path)
    assert root.exists(), f'provided COCO path {root} does not exist'
    mode = 'instances'
    bdd=False
    if args.dataset == 'coco_base':
        PATHS = {
            "train": (root / "COCO/train2017", root / "COCO/annotations" / f'{mode}_train2017.json'),
            "val": (root / "COCO/val2017", root / "COCO/annotations" / f'{mode}_val2017_ood_rm_overlap.json'),
        }
    # elif args.dataset == 'coco_ood_val_bdd':
    #     PATHS = {
    #         "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
    #         "val": (root / "val2017", root / "annotations" / f'{mode}_val2017_ood_wrt_bdd_rm_overlap.json'),
    #     }
    # elif args.dataset == 'openimages_ood_val':
    #     PATHS = {
    #         "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
    #         "val": (args.open_root, args.open_ann_root),
    #     }
    # elif args.dataset == 'bdd':
    #     PATHS = {
    #         "train": (Path(args.bdd_root) / "train", args.bdd_ann_root_train),
    #         # "train" :(Path(args.bdd_root) / "train",
    #         #           '/nobackup-slow/dataset/my_xfdu/bdd-100k/bdd100k/labels/det_20/train_converted.json' ),
    #         "val": (Path(args.bdd_root) /  "val", args.bdd_ann_root_test),
    #     }
    #     bdd = True
    # else:
    #     PATHS = {
    #         "train": (root / "train2017", root / "annotations" / f'{mode}_train2017.json'),
    #         "val": (root / "val2017", root / "annotations" / f'{mode}_val2017.json'),
    #     }
    # if args.dataset == 'openimages_ood_val'or "coco_ood" :
    #     image_set="val"
    img_folder, ann_file = PATHS[image_set]

    no_cats = False
    if 'coco' not in args.dataset and 'bdd' not in args.dataset and 'open' not in args.dataset:
        no_cats = True
    filter_pct = -1
    if image_set == 'train' and args.filter_pct > 0:
        filter_pct = args.filter_pct
    dataset = CocoDetection(img_folder, ann_file, return_masks=args.masks,
                            cache_mode=args.cache_mode, local_rank=get_local_rank(), local_size=get_local_size(), no_cats=no_cats, filter_pct=filter_pct, bdd=bdd)
    return dataset , img_folder, ann_file