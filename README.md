# EasYoloD

Easy Yolo Detect

用户快速部署yolo的识别程序，支持onnxruntime, opencv(dnn), openvino

仅需简短几行代码即可实现yolo目标检测

## Provider 介绍

1. onnxruntime:
    + cpu: 适配性最高的版本，不需要GPU即可执行
    + gpu: onnxruntime-gpu 需要英伟达GPU，并且安装对应版本cuda，cudnn之后才能使用，速度快
    + onnxdml: onnxruntime-directml 不需要使用特定GPU，核显也可以允许，而且不需要安装任何额外程序，速度一般，而且仅适用与windos系统
1. openvino: 
    + cpu: 同onnx的cpu一样
    + gpu: 仅适用于intel的GPU，其他GPU不可用
1. opencv: 
    + cpu: 同上
    + gpu: 需要单独编译带有cuda的opencv包，并正确配置路径，并且安装好cuda和cudnn，速度快

## 安装和使用

```bash
pip install easyolod
```

Requirements
+ Python 3.8-3.12
+ opencv-python <= 4.10.0.84
+ numpy <= 1.26

使用: 

```python
import easyolod

easyolod.init(provider='onnxruntime',gpu=False) # onnxruntime-directml 则使用onnxdml，openvino使用 openvino
model = easyolod.Model()
# conf 置信度
# ious
# namse 可以是文件，也可以是一个list
model.load('modelpath', conf, ious, names)
# or 你使用的是opencv dnn yolov4的weight模型
# model.load('config path', 'weight path', inputsize, names, conf, nms)

result = model.detect(img=image)
# or 你希望自己处理输出
# result = model.detect_only(img=image)
```
