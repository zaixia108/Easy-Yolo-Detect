# EasYoloD

Easy Yolo Detection

A tool for users to quickly deploy YOLO detection programs, supporting onnxruntime, opencv(dnn), and openvino.

With just a few lines of code, you can implement YOLO object detection.

## Provider Introduction

1. **onnxruntime**:
    + **cpu**: The most compatible version, does not require a GPU to run.
    + **gpu**: onnxruntime-gpu requires an NVIDIA GPU and the corresponding version of CUDA and cuDNN to be installed. It offers fast performance.
    + **onnxdml**: onnxruntime-directml does not require a specific GPU and can run on integrated graphics. No additional programs need to be installed. It offers moderate performance and is only available on Windows systems.

2. **openvino**:
    + **cpu**: Similar to onnxruntime's CPU version.
    + **gpu**: Only compatible with Intel GPUs, other GPUs are not supported.

3. **opencv**:
    + **cpu**: Similar to the above.
    + **gpu**: Requires a separately compiled OpenCV package with CUDA support, correctly configured paths, and CUDA and cuDNN installed. It offers fast performance.

## Installation and Usage

```bash
pip install EasYoloD
```

### Requirements
+ Python 3.8-3.12
+ opencv-python <= 4.10.0.84
+ numpy <= 1.26

### Usage:

```python
import EasYoloD

EasYoloD.init(provider='onnxruntime', gpu=False) # Use 'onnxdml' for onnxruntime-directml, 'openvino' for openvino
model = EasYoloD.Model()
# conf: confidence threshold
# ious: IoU threshold
# names: can be a file or a list
model.load('modelpath', conf, ious, names)
# Or if you are using OpenCV DNN YOLOv4 weight model
# model.load('config path', 'weight path', inputsize, names, conf, nms)

result = model.detect(img=image)
# Or if you want to handle the output yourself
# result = model.detect_only(img=image)
```

### Output Example:

**detect**:
```
{
  1: [
    {'confidence': 0.89, 'box': [(614, 202), (732, 242)], 'center': (673, 222)}, 
    {'confidence': 0.87, 'box': [(975, 227), (1105, 268)], 'center': (1040, 247)}, 
    {'confidence': 0.87, 'box': [(845, 241), (962, 284)], 'center': (903, 262)}, 
    {'confidence': 0.86, 'box': [(418, 203), (495, 243)], 'center': (456, 223)}, 
    {'confidence': 0.85, 'box': [(713, 233), (822, 273)], 'center': (767, 253)}, 
    {'confidence': 0.83, 'box': [(776, 222), (888, 261)], 'center': (832, 241)}
  ], 
  2: [], 
  3: [
    {'confidence': 0.8, 'box': [(664, 265), (687, 289)], 'center': (675, 277)}
  ], 
  4: [
    {'confidence': 0.86, 'box': [(846, 195), (955, 236)], 'center': (900, 215)}, 
    {'confidence': 0.84, 'box': [(1108, 227), (1208, 273)], 'center': (1158, 250)}
  ], 
  5: [], 
  6: [], 
  7: []
}
```

**detect_only**:
```
(array([[ 614.5011 ,  202.27354,  732.4082 ,  242.74388],
       [ 975.4805 ,  227.59409, 1105.0723 ,  268.69995],
       [ 845.77277,  241.3953 ,  962.0877 ,  284.1887 ],
       [ 418.44012,  203.71834,  495.6739 ,  243.37538],
       [ 846.04956,  195.53143,  955.15515,  236.9972 ],
       [ 713.3884 ,  233.3027 ,  822.95776,  273.27628],
       [1108.0188 ,  227.39557, 1208.6423 ,  273.43536],
       [ 776.30786,  222.16605,  888.85815,  261.70145],
       [ 664.80615,  265.0358 ,  687.7573 ,  289.32138]], dtype=float32), array([0.88843024, 0.86892086, 0.8652373 , 0.8610253 , 0.858262  ,
       0.84596515, 0.8361889 , 0.83084583, 0.8002863 ], dtype=float32), array([0, 0, 0, 0, 3, 0, 3, 0, 2], dtype=int64))
```