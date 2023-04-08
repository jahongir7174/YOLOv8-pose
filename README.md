YOLOv8-pose re-implementation using PyTorch

### Installation

```
conda create -n YOLO python=3.8
conda activate YOLO
conda install pytorch torchvision torchaudio cudatoolkit=10.2 -c pytorch-lts
pip install opencv-python==4.5.5.64
pip install PyYAML
pip install tqdm
```

### Train

* Configure your pose dataset path in `main.py` for training
* Run `bash main.sh $ --train` for training, `$` is number of GPUs

### Test

* Configure your dataset path in `main.py` for testing
* Run `python main.py --test` for testing

### Results

|  Version   | Epochs | Pose mAP |                                                                                       Download |
|:----------:|:------:|---------:|-----------------------------------------------------------------------------------------------:|
| v8_n_pose  |  1000  |     50.2 |                                                                     [model](./weights/best.pt) |
| v8_n_pose* |  1000  |     50.5 | [model](https://github.com/jahongir7174/YOLOv8-pt/releases/download/v0.0.1-alpha/v8_n_pose.pt) |
| v8_s_pose* |  1000  |     59.5 | [model](https://github.com/jahongir7174/YOLOv8-pt/releases/download/v0.0.1-alpha/v8_s_pose.pt) |
| v8_m_pose* |  1000  |     63.8 | [model](https://github.com/jahongir7174/YOLOv8-pt/releases/download/v0.0.1-alpha/v8_m_pose.pt) |
| v8_l_pose* |  1000  |     67.4 | [model](https://github.com/jahongir7174/YOLOv8-pt/releases/download/v0.0.1-alpha/v8_l_pose.pt) |
| v8_x_pose* |  1000  |     69.4 | [model](https://github.com/jahongir7174/YOLOv8-pt/releases/download/v0.0.1-alpha/v8_x_pose.pt) |

* `*` means that weights are ported from original repo, see reference

### Dataset structure

    ├── COCO 
        ├── images
            ├── train2017
                ├── 1111.jpg
                ├── 2222.jpg
            ├── val2017
                ├── 1111.jpg
                ├── 2222.jpg
        ├── labels
            ├── train2017
                ├── 1111.txt
                ├── 2222.txt
            ├── val2017
                ├── 1111.txt
                ├── 2222.txt

### Results

![Alt Text](./demo/demo.gif)

#### Reference

* https://github.com/ultralytics/yolov5
* https://github.com/ultralytics/ultralytics
