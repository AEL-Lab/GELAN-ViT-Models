import os
import numpy as np
import cv2
from yolox.evaluators.voc_eval import voc_eval
from .datasets_wrapper import CacheDataset, cache_read_img
import pickle
class SODDAnnotationTransform(object):
    """Transforms an annotation into a Tensor of bbox coords and label index"""
    def __init__(self, class_to_ind=None):
        self.class_to_ind = class_to_ind or dict(zip(range(1), range(1)))

    def __call__(self, target):
        # Assume the images had already been converted to 640 x 640 
        res = np.empty((0, 5))
        for obj in target:
            class_id = int(obj[0])
            x_center, y_center, width, height = [float(x) for x in obj[1:]]
            
            # Convert normalized values to absolute coordinates
            x_center *= 640
            y_center *= 640
            width *= 640
            height *= 640
            
            xmin = x_center - width / 2
            ymin = y_center - height / 2
            xmax = x_center + width / 2
            ymax = y_center + height / 2
            
            bndbox = [xmin, ymin, xmax, ymax, class_id]
            res = np.vstack((res, bndbox))
        return res, (640, 640)

class SODDDetection(CacheDataset):

    def __init__(
        self,
        data_dir,
        img_size=(640, 640),
        preproc=None,
        target_transform=SODDAnnotationTransform(),
        cache=False,
        cache_type="ram",
        split='train'
    ):
        self.root = data_dir
        self.img_size = img_size
        self.preproc = preproc
        self.target_transform = target_transform
        self._classes = [0]  # Assuming only one class
        self.class_ids = list(range(1))
        self.split = split
        self.cache = cache  # Ensure cache is set
        self.cache_type = cache_type  # Ensure cache_type is set
        self.ids = self._load_image_ids()
        self.num_imgs = len(self.ids)

        self.annotations = self._load_coco_annotations()

        path_filename = [img_id for img_id in self.ids]
        super().__init__(
            input_dimension=img_size,
            num_imgs=self.num_imgs,
            data_dir=self.root,
            cache_dir_name=f"cache_sodd_{self.split}",
            path_filename=path_filename,
            cache=cache,
            cache_type=cache_type
        )

    def __len__(self):
        return self.num_imgs

    def _load_image_ids(self):
        file_path = os.path.join(self.root, f'{self.split}.txt')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"{file_path} does not exist.")
        with open(file_path, 'r') as f:
            ids = [line.strip() for line in f.readlines()]
        return ids

    def _load_coco_annotations(self):
        return [self.load_anno_from_ids(_ids) for _ids in range(self.num_imgs)]

    def load_anno_from_ids(self, index):
        img_id = self.ids[index]
        label_file = os.path.join(self.root, 'labels', self.split, img_id.split('/')[-1].replace('.jpg', '.txt'))
        if not os.path.exists(label_file):
            raise FileNotFoundError(f"{label_file} does not exist.")
        
        with open(label_file, 'r') as f:
            target = [line.strip().split() for line in f.readlines()]

        assert self.target_transform is not None
        res, img_info = self.target_transform(target)

        return (res, img_info, img_info)

    def load_anno(self, index):
        return self.annotations[index][0]

    def load_resized_img(self, index):
        img = self.load_image(index)
        r = min(self.img_size[0] / img.shape[0], self.img_size[1] / img.shape[1])
        resized_img = cv2.resize(
            img,
            (int(img.shape[1] * r), int(img.shape[0] * r)),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.uint8)
        return resized_img

    def load_image(self, index):
        img_id = self.ids[index]
        img_path = os.path.join(self.root, 'images', self.split, img_id.split('/')[-1])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        assert img is not None, f"file named {img_path} not found"
        return img

    @cache_read_img(use_cache=True)
    def read_img(self, index):
        return self.load_resized_img(index)

    def pull_item(self, index):
        """Returns the original image and target at an index for mixup
        
        Note: not using self.__getitem__(), as any transformations passed in
        could mess up this functionality.

        Argument:
            index (int): index of img to show
        Return:
            img, target, img_info, index, img_id
        """
        target, img_info, _ = self.annotations[index]
        img = self.read_img(index)
        img_id = self.ids[index]
        return img, target, img_info, index, img_id.split('/')[-1]

    @CacheDataset.mosaic_getitem
    def __getitem__(self, index):
        img, target, img_info, img_id, _ = self.pull_item(index)
        if self.preproc is not None:
            img, target = self.preproc(img, target, self.input_dim)
        return img, target, img_info, img_id

    def evaluate_detections(self, all_boxes, output_dir=None):
        self._write_voc_results_file(all_boxes)
        IouTh = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
        mAPs = []
        for iou in IouTh:
            mAP = self._do_python_eval(output_dir, iou)
            mAPs.append(mAP)
        print("--------------------------------------------------------------")
        print("map_5095:", np.mean(mAPs))
        print("map_50:", mAPs[0])
        print("--------------------------------------------------------------")
        return np.mean(mAPs), mAPs[0]

    def _get_voc_results_file_template(self):
        filename = "comp4_det_test" + "_{:s}.txt"
        filedir = os.path.join(self.root, "results", "sodd", "Main")
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path

    def _write_voc_results_file(self, all_boxes):
        for cls_ind, cls in enumerate([0]):
            cls_ind = cls_ind
            print("Writing {} results file".format(cls))
            filename = self._get_voc_results_file_template().format(str(cls))  # Convert cls to string
            with open(filename, "wt") as f:
                for im_ind, index in enumerate(self.ids):
                    index = index.split('/')[-1].replace('.jpg', '')
                    dets = all_boxes[cls_ind][im_ind]
                    if dets.shape[0] == 0:
                        continue
                    for k in range(dets.shape[0]):
                        f.write(
                            "{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n".format(
                                index,
                                dets[k, -1],
                                dets[k, 0] + 1,
                                dets[k, 1] + 1,
                                dets[k, 2] + 1,
                                dets[k, 3] + 1,
                            )
                        )

    def _do_python_eval(self, output_dir="output", iou=0.5):
        rootpath = self.root
        name = self.split
        annopath = os.path.join(rootpath, "labels", name, "{:s}.txt")
        imagesetfile = os.path.join(rootpath, f'{name}.txt')
        cachedir = os.path.join(self.root, "annotations_cache", "sodd", name)
        if not os.path.exists(cachedir):
            os.makedirs(cachedir)
        aps = []
        use_07_metric = False
        print("Eval IoU : {:.2f}".format(iou))
        if output_dir is not None and not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate([0]):
            filename = self._get_voc_results_file_template().format(str(cls))  # Convert cls to string
            rec, prec, ap = voc_eval(
                filename,
                annopath,
                imagesetfile,
                str(cls),  # Convert cls to string
                cachedir,
                ovthresh=iou,
                use_07_metric=use_07_metric,
            )
            aps += [ap]
            if iou == 0.5:
                print("AP for {} = {:.4f}".format(cls, ap))
            if output_dir is not None:
                with open(os.path.join(output_dir, str(cls) + "_pr.pkl"), "wb") as f:
                    pickle.dump({"rec": rec, "prec": prec, "ap": ap}, f)
        if iou == 0.5:
            print("Mean AP = {:.4f}".format(np.mean(aps)))
            print("~~~~~~~~")
            print("Results:")
            for ap in aps:
                print("{:.3f}".format(ap))
            print("{:.3f}".format(np.mean(aps)))
            print("~~~~~~~~")
            print("")
            print("--------------------------------------------------------------")
            print("Results computed with the **unofficial** Python eval code.")
            print("Results should be very close to the official MATLAB eval code.")
            print("Recompute with `./tools/reval.py --matlab ...` for your paper.")
            print("-- Thanks, The Management")
            print("--------------------------------------------------------------")

        return np.mean(aps)