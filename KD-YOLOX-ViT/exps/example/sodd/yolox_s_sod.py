#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp
from loguru import logger

import random
import cv2
import matplotlib.pyplot as plt
class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.375
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

        # ---------------- YOLOX with ViT Layer ---------------- #
        
        self.vit = False
        
        # ---------------- Knowledge Distillation config ---------------- #
        
        #KD set to True activate add the KD loss to the ground truth loss
        self.KD = False
        
        #KD_Online set to False recquires the teacher FPN logits saved to the "folder_KD_directory" folder
        #Then the student training will use the teacher FPN logits
        #Otherwise, if KD_Online set to True the student use the online data augmentation and does not recquire saved teacher FPN logits
        self.KD_online = False
        
        #KD_Teacher_Inference set to True save the FPN logits before using offline KD
        
        #folder_KD_directory is the folder where the teacher FPN logits are saved
        
        if self.KD and not self.KD_online:
            # ---------------- dataloader config ---------------- #

            # To disable multiscale training, set the value to 0.
            self.multiscale_range = 0

            # --------------- transform config ----------------- #
            # prob of applying mosaic aug
            self.mosaic_prob = 0
            # prob of applying mixup aug
            self.mixup_prob = 0
            # prob of applying hsv aug
            self.hsv_prob = 0
            # prob of applying flip aug
            self.flip_prob = 0.0
            # rotation angle range, for example, if set to 2, the true range is (-2, 2)
            self.degrees = 0.0
            # translate range, for example, if set to 0.1, the true range is (-0.1, 0.1)
            self.translate = 0
            self.mosaic_scale = (1, 1)
            # apply mixup aug or not
            self.enable_mixup = False
            self.mixup_scale = (1, 1)
            # shear angle range, for example, if set to 2, the true range is (-2, 2)
            self.shear = 0
            
        self.data_dir = "path/to/dataset.yaml"

        
        self.num_classes = 1

        self.max_epoch = 1000
        self.data_num_workers = 4
        self.eval_interval = 20

    def show_random_image_with_annotation(self, ds):
      # Get the image and annotations
      img, target, img_info, img_id, name = ds.pull_item(1)

      # Convert the image from BGR to RGB (for matplotlib display)
      img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
      actual_height, actual_width, _ = img.shape
      # Draw bounding boxes on the image
      for bbox in target:
          class_id = int(bbox[4])
          center_x = bbox[0] * actual_width
          center_y = bbox[1] * actual_height
          width = bbox[2] * actual_width
          height = bbox[3] * actual_height

          xmin = int(center_x - width / 2)
          ymin = int(center_y - height / 2)
          xmax = int(center_x + width / 2)
          ymax = int(center_y + height / 2)

          # Draw rectangle and label
          cv2.rectangle(img_rgb, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
          label = f"Class {class_id}"
          cv2.putText(img_rgb, label, (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


      # Save the image to file
      output_path = f"annotated_image_1.jpg"
      cv2.imwrite(output_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
      print(f"Image saved to {output_path}")

      # Display the image
      plt.figure(figsize=(10, 10))
      plt.imshow(img_rgb)
      plt.title(f"Image ID: {img_id}")
      plt.axis('off')
      plt.show()
      print("Image should be displayed now!")

    def get_dataset(self, cache: bool, cache_type: str = "ram"):
        from yolox.data import CustomDetection, TrainTransform
        ds = CustomDetection(
            data_dir='path/to/train/dataset',
            img_size=self.input_size,
            preproc=TrainTransform(
                max_labels=50,
                flip_prob=self.flip_prob,
                hsv_prob=self.hsv_prob
            ),
            cache=cache,
            cache_type=cache_type,
            split='train'
        )
        return ds

    def get_eval_dataset(self, **kwargs):
        from yolox.data import CustomDetection, ValTransform
        legacy = kwargs.get("legacy", False)
        ds = CustomDetection(
            data_dir='path/to/validation/dataset',
            img_size=self.test_size,
            preproc=ValTransform(legacy=legacy),
            cache=False,
            split='val'
        )
        self.show_random_image_with_annotation(ds)
        return ds

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import VOCEvaluator

        return VOCEvaluator(
            dataloader=self.get_eval_loader(batch_size, is_distributed,
                                            testdev=testdev, legacy=legacy),
            img_size=self.test_size,
            confthre=self.test_conf,
            nmsthre=self.nmsthre,
            num_classes=self.num_classes,
        )