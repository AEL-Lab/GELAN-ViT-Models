# GELAN-ViT-Models

GELAN-ViT-Models is a repository for GELAN-ViT models tailored for satellite object detection tasks. The base of the code for GELAN-ViT comes from the [YOLOv9 repository](https://github.com/WongKinYiu/yolov9). Additionally, this repository includes the dataset handler for running KD-YOLOX-ViT on the Satellite Object Detection (SOD) dataset.

**Research Paper Citation:**

The models are discussed in our research paper:
* Wenxuan Zhang and Peng Hu, "Sensing for Space Safety and Sustainability: A Deep Learning Approach With Vision Transformers," in the 12th Annual IEEE International Conference on Wireless for Space and Extreme Environments (WiSEE 2024), 16-18 December 2024, Daytona Beach, FL, USA.

# GELAN-ViT

## Installation
Follow the steps below to install the required dependencies:

```shell
git clone git@github.com:AEL-Lab/GELAN-ViT-Models.git
cd GELAN-ViT-Models/GELAN-ViT
pip install -r requirements.txt
```


## Training

To train a GELAN-ViT model on your machine or custom dataset, use the following command:

``` shell
python train.py --workers 8 --device 0 --batch 32 --data path/to/data.yaml --img 640 --cfg models/detect/GELAN-ViT.yaml --weights '' --name gelan-vit --hyp hyp.scratch-adj.yaml --epochs 500
```

## Inference
To perform object detection using a trained GELAN-ViT model, use the following command:

``` shell
# inference converted yolov9 models
python detect.py --source './data/images/horses.jpg' --img 640 --device 0 --weights path/to/weights.pt --name gelan_vit_640_detect

```


### Validation
To evaluate the performance of your trained model, use the following command:

```shell
python val.py --batch 8 --weights path/to/weights.pt --data path/to/data.yaml  --workers 6 --save-json
```

### Model Structures
- Primary Models: Located in `GELAN-ViT/models/detect`, these models integrate ViT into the head of GELAN.
- Alternate Models: Located in `GELAN-ViT/models/detect/alternate`, providing an alternative implementation where ViT is appended to GELAN's head. Both versions offer identical performance.


# Dataset 
The SOD dataset used is available in the separate repo at [https://github.com/AEL-Lab/satellite-object-detection-dataset.git](https://github.com/AEL-Lab/satellite-object-detection-dataset.git).

# YOLOX

We also provide dataset handlers for running KD-YOLOX-ViT on the SOD dataset.

### Install
Follow the steps below to install the required dependencies:

```shell
cd GELAN-ViT-Models/KD-YOLOX-ViT
pip install -r requirements.txt
pip install -v -e .
```

Ensure you update the following paths in the experiment configuration files before running the training:

- Replace `path/to/dataset.yaml` with the actual path to your dataset configuration file.
- Replace `path/to/train/dataset` with the actual path to your training dataset.
- Replace `path/to/validation/dataset` with the actual path to your validation dataset.

### Train
To train a KD-YOLOX-ViT model with SOD dataset, use the following command:

```shell
python tools/train.py -f exps/example/sodd/yolox_s_vit_sod.py -b 8 --fp16
```
### Validation
To evaluate the performance of your trained model, use the following command:

```shell
python3 tools/eval.py --speed -f exps/example/sodd/yolox_s_vit_sod.py -b 8 --fp16
```

## Licensing

This repository contains code under two different licenses:

- **GELAN-ViT**: Licensed under the [GNU General Public License v3.0 (GPL-3.0)](./GELAN-ViT/LICENSE.md).
- **KD-YOLOX-ViT**: Licensed under the [Apache License 2.0](./KD-YOLOX-ViT/LICENSE).

Please refer to the `LICENSE` file in each folder for detailed terms.

## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/remaro-network/KD-YOLOX-ViT](https://github.com/remaro-network/KD-YOLOX-ViT)
* [https://github.com/WongKinYiu/yolov9](https://github.com/WongKinYiu/yolov9)
* [https://github.com/AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)
* [https://github.com/WongKinYiu/yolor](https://github.com/WongKinYiu/yolor)
* [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
* [https://github.com/VDIGPKU/DynamicDet](https://github.com/VDIGPKU/DynamicDet)
* [https://github.com/DingXiaoH/RepVGG](https://github.com/DingXiaoH/RepVGG)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://github.com/meituan/YOLOv6](https://github.com/meituan/YOLOv6)

</details>
