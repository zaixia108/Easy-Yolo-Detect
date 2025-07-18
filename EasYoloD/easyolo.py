import time
import cv2
import numpy as np


class Index:
    USE_GPU = False
    PROVIDER = None
    MODULE = None


def init(provider, gpu=False):
    if provider == 'onnxruntime':
        try:
            import onnxruntime
            Index.MODULE = onnxruntime
        except ImportError:
            raise ImportError('Please install onnxruntime')
    elif provider == 'opencv':
        try:
            import cv2
            Index.MODULE = cv2
        except ImportError:
            raise ImportError('Please install opencv-python')
    elif provider == 'openvino':
        try:
            import openvino.runtime
            Index.MODULE = openvino.runtime
        except ImportError:
            raise ImportError('Please install openvino')
    elif provider == 'onnxdml':
        try:
            import onnxruntime
            Index.MODULE = onnxruntime
            providers = onnxruntime.get_available_providers()
            if 'DmlExecutionProvider' not in providers:
                raise ImportError('DirectMLExecutionProvider is not available\nPlease make sure you have installed onnxruntime-directml')
        except ImportError:
            raise ImportError('Please install onnxruntime-directml')
    else:
        raise ValueError('Unknown provider')
    Index.PROVIDER = provider
    if gpu:
        if provider == 'onnxruntime':
            providers = Index.MODULE.get_available_providers()
            if 'CUDAExecutionProvider' not in providers:
                print('CUDAExecutionProvider is not available')
                print('Available providers:', providers)
                print('Using CPUExecutionProvider')
                Index.USE_GPU = False
            else:
                print('Using CUDAExecutionProvider')
                Index.USE_GPU = True
        elif provider == 'opencv':
            device = Index.MODULE.cuda.getCudaEnabledDeviceCount()
            if device == 0:
                print('CUDA is not available')
                print('Using CPU')
                Index.USE_GPU = False
            else:
                Index.USE_GPU = True
        elif provider == 'openvino':
            core = Index.MODULE.Core()
            device = core.get_available_devices()
            if 'GPU' not in device:
                print('GPU is not available')
                print('Using CPU')
                Index.USE_GPU = False
            else:
                Index.USE_GPU = True
        elif provider == 'onnxdml':
            try:
                import onnxruntime
                Index.MODULE = onnxruntime
                providers = onnxruntime.get_available_providers()
                if 'DmlExecutionProvider' not in providers:
                    Index.USE_GPU = False
                    raise ImportError(
                        'DirectMLExecutionProvider is not available\nPlease make sure you have installed onnxruntime-directml')
                Index.USE_GPU = True
            except ImportError:
                Index.USE_GPU = False
                raise ImportError('Please install onnxruntime-directml')
        else:
            Index.USE_GPU = False
            raise ValueError('Unknown provider')


class OnxDet:
    def __init__(self, path, conf_thres=0.3, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

        # Initialize model
        self.initialize_model(path)

    def __call__(self, image):
        return self.detect_objects(image)

    def initialize_model(self, path):
        if Index.USE_GPU:
            if Index.PROVIDER == 'onnxruntime':
                self.session = Index.MODULE.InferenceSession(path, providers=[
                    'CUDAExecutionProvider'])
            elif Index.PROVIDER == 'onnxdml':
                self.session = Index.MODULE.InferenceSession(path, providers=[
                    'DmlExecutionProvider'])
        else:
            self.session = Index.MODULE.InferenceSession(path, providers=['CPUExecutionProvider'])
        # Get model info
        self.get_input_details()
        self.get_output_details()

    def detect_objects(self, image):
        input_tensor, ratio = self.prepare_input(image)

        # Perform inference on the image
        outputs = self.inference(input_tensor)

        self.boxes, self.scores, self.class_ids = self.process_output(outputs, ratio)

        return self.boxes, self.scores, self.class_ids

    def prepare_input(self, image):
        self.img_height, self.img_width = image.shape[:2]

        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize图片不要直接使用resize，需要按比例缩放，空白区域填空纯色即可
        input_img, ratio = self.ratioresize(input_img)

        # Scale input pixel values to 0 to 1.txt
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor, ratio

    def inference(self, input_tensor):
        start = time.perf_counter()
        outputs = self.session.run(self.output_names, {self.input_names[0]: input_tensor})

        return outputs

    def process_output(self, output, ratio):
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], []

        class_ids = np.argmax(predictions[:, 4:], axis=1)

        boxes = self.extract_boxes(predictions, ratio)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = self.nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def extract_boxes(self, predictions, ratio):
        boxes = predictions[:, :4]

        boxes *= ratio

        boxes = self.xywh2xyxy(boxes)

        return boxes

    def rescale_boxes(self, boxes):

        input_shape = np.array([self.input_width, self.input_height, self.input_width, self.input_height])
        boxes = np.divide(boxes, input_shape, dtype=np.float32)
        boxes *= np.array([self.img_width, self.img_height, self.img_width, self.img_height])

        return boxes

    def get_input_details(self):
        model_inputs = self.session.get_inputs()
        self.input_names = [model_inputs[i].name for i in range(len(model_inputs))]

        self.input_shape = model_inputs[0].shape
        self.input_height = self.input_shape[2]
        self.input_width = self.input_shape[3]

    def get_output_details(self):
        model_outputs = self.session.get_outputs()
        self.output_names = [model_outputs[i].name for i in range(len(model_outputs))]

    # 等比例缩放图片
    def ratioresize(self, im, color=114):
        shape = im.shape[:2]
        new_h, new_w = self.input_height, self.input_width
        padded_img = np.ones((new_h, new_w, 3), dtype=np.uint8) * color

        r = min(new_h / shape[0], new_w / shape[1])

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

        if shape[::-1] != new_unpad:
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)

        padded_img[: new_unpad[1], : new_unpad[0]] = im
        padded_img = np.ascontiguousarray(padded_img)
        return padded_img, 1 / r

    def nms(self, boxes, scores, iou_threshold):
        sorted_indices = np.argsort(scores)[::-1]

        keep_boxes = []
        while sorted_indices.size > 0:
            box_id = sorted_indices[0]
            keep_boxes.append(box_id)

            ious = self.compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

            keep_indices = np.where(ious < iou_threshold)[0]

            sorted_indices = sorted_indices[keep_indices + 1]

        return keep_boxes

    def compute_iou(self, box, boxes):
        # Compute xmin, ymin, xmax, ymax for both boxes
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])

        intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - intersection_area

        iou = intersection_area / union_area

        return iou

    def xywh2xyxy(self, x):
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y


class DnnDet:

    def __init__(self, cfg_path ,weight_path ,input_size: tuple = (416, 416), conf=0.3, nms=0.5):
        self.input_size = input_size
        self.conf = conf
        self.nms = nms
        net = cv2.dnn.DetectionModel(cfg_path, weight_path)
        if Index.USE_GPU:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        net.setInputSize(self.input_size[0], self.input_size[1])
        net.setInputScale(1.0 / 255)
        net.setInputSwapRB(True)
        self.net = net

    def __call__(self, *args, **kwargs):
        return self.net.detect(*args, **kwargs)


class OpenVinoDet:
    def __init__(self, model_path, conf_thres=0.3, iou_thres=0.5):
        self.conf_threshold = conf_thres
        self.iou_threshold = iou_thres

        # Initialize OpenVINO model
        self.core = Index.MODULE.Core()
        self.model = self.core.read_model(model=model_path)
        if Index.USE_GPU:
            self.compiled_model = self.core.compile_model(self.model, "GPU")
        else:
            self.compiled_model = self.core.compile_model(self.model, "CPU")  # 可以根据需要选择其他设备

        inputs = self.model.inputs
        self.input_names = [input_tensor.get_any_name() for input_tensor in inputs]

        outputs = self.model.outputs
        self.output_names = [output_tensor.get_any_name() for output_tensor in outputs]

        self.input_shape = self.model.input(0).shape[-2:]  # H, W

    def __call__(self, image):
        return self.inference(image)

    def inference(self, image):
        input_tensor, ratio = self.prepare_input(image)

        # Perform inference
        outputs = self.compiled_model([input_tensor])

        boxes, scores, class_ids = self.process_output(outputs[0], ratio)

        return boxes, scores, class_ids

    def prepare_input(self, image):
        input_img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Resize image and retain aspect ratio
        input_img, ratio = self.ratio_resize(input_img)

        input_img = input_img / 255.0
        input_img = np.transpose(input_img, (2, 0, 1))
        input_tensor = input_img[np.newaxis, :, :, :].astype(np.float32)

        return input_tensor, ratio

    def ratio_resize(self, img):
        shape = img.shape[:2]
        new_h, new_w = self.input_shape[0], self.input_shape[1]
        padded_img = np.ones((new_h, new_w, 3), dtype=np.uint8) * 114

        r = min(new_h / shape[0], new_w / shape[1])

        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        padded_img[: new_unpad[1], : new_unpad[0]] = img
        padded_img = np.ascontiguousarray(padded_img)
        return padded_img, 1 / r

    def process_output(self, output, ratio):
        predictions = np.squeeze(output[0]).T

        # Filter out object confidence scores below threshold
        scores = np.max(predictions[:, 4:], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]
        # print(scores)

        if len(scores) == 0:
            return [], [], []

        class_ids = np.argmax(predictions[:, 4:], axis=1)

        boxes = self.extract_boxes(predictions, ratio)

        indices = self.nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices]

    def nms(self, boxes, scores, iou_threshold):
        sorted_indices = np.argsort(scores)[::-1]

        keep_boxes = []
        while sorted_indices.size > 0:
            box_id = sorted_indices[0]
            keep_boxes.append(box_id)

            ious = self.compute_iou(boxes[box_id, :], boxes[sorted_indices[1:], :])

            keep_indices = np.where(ious < iou_threshold)[0]

            sorted_indices = sorted_indices[keep_indices + 1]

        return keep_boxes

    def compute_iou(self, box, boxes):
        xmin = np.maximum(box[0], boxes[:, 0])
        ymin = np.maximum(box[1], boxes[:, 1])
        xmax = np.minimum(box[2], boxes[:, 2])
        ymax = np.minimum(box[3], boxes[:, 3])

        intersection_area = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)

        box_area = (box[2] - box[0]) * (box[3] - box[1])
        boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union_area = box_area + boxes_area - intersection_area

        iou = intersection_area / union_area

        return iou

    def extract_boxes(self, predictions, ratio):
        boxes = predictions[:, :4]  # Assuming the first four are box coordinates
        boxes *= ratio  # Rescale boxes

        # Convert boxes from xywh to xyxy format if needed
        boxes = self.xywh_to_xyxy(boxes)

        return boxes

    def xywh_to_xyxy(self, x):
        y = np.copy(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y  # Return in xyxy format


class Model:
    def __init__(self):
        if not Index.PROVIDER:
            raise AttributeError('Please init first')
        self.detector = None
        self.names = None
        self.accesable = False

    def load(self, model_path, names, conf, **kwargs):
        if Index.PROVIDER == 'onnxruntime' or Index.PROVIDER == 'onnxdml':
            ious = kwargs.get('iou')
            if ious is None:
                ious = 0.5
                print('use default iou threshold: 0.5')
            self.detector = OnxDet(model_path, conf, ious)
            if type(names) == str:
                with open(names, 'r', encoding='utf-8') as f:
                    self.names = f.readlines()
                    for i in range(len(self.names)):
                        self.names[i] = self.names[i].strip()
            elif type(names) == list:
                self.names = names
            else:
                raise ValueError('Unknown type of names')
        elif Index.PROVIDER == 'opencv':
            cfg_path = kwargs.get('cfg_path', None)
            if cfg_path is None:
                raise ValueError('cfg_path is required for opencv provider')
            input_size = kwargs.get('input_size')
            if input_size is None:
                input_size = (416, 416)
                print('use default input size: (416, 416)')
            nms = kwargs.get('nms')
            if nms is None:
                nms = 0.5
                print('use default nms threshold: 0.5')
            self.detector = DnnDet(cfg_path, model_path, input_size, conf, nms)
            if type(names) == str:
                with open(names, 'r', encoding='utf-8') as f:
                    self.names = f.readlines()
                    for i in range(len(self.names)):
                        self.names[i] = self.names[i].strip()
            elif type(names) == list:
                self.names = names
            else:
                raise ValueError('Unknown type of names')
        elif Index.PROVIDER == 'openvino':
            ious = kwargs.get('iou')
            if ious is None:
                ious = 0.5
                print('use default iou threshold: 0.5')
            self.detector = OpenVinoDet(model_path, conf, ious)
            if type(names) == str:
                with open(names, 'r', encoding='utf-8') as f:
                    self.names = f.readlines()
                    for i in range(len(self.names)):
                        self.names[i] = self.names[i].strip()
            elif type(names) == list:
                self.names = names
            else:
                raise ValueError('Unknown type of names')
        self.accesable = True

    def detect(self, img, clean=True):
        if not self.accesable:
            raise AttributeError('Model not loaded')
        if Index.PROVIDER == 'onnxruntime' or Index.PROVIDER == 'openvino' or Index.PROVIDER == 'onnxdml':
            result = {}
            for i in self.names:
                result[i] = []
            boxes, scores, class_ids = self.detector(img)
            for i in range(len(boxes)):
                p1 = int(boxes[i][0])
                p2 = int(boxes[i][1])
                p3 = int(boxes[i][2])
                p4 = int(boxes[i][3])
                lt = (p1, p2)
                rb = (p3, p4)
                data = {
                    'confidence': round(scores[i], 2),
                    'box': [lt, rb],
                    'center': (int((lt[0] + rb[0]) // 2), int((lt[1] + rb[1]) // 2)),
                }
                result[self.names[class_ids[i]]].append(data)
            return result
        elif Index.PROVIDER == 'opencv':
            classes, confidences, boxes = self.detector(img, confThreshold=self.detector.conf, nmsThreshold=self.detector.nms)
            result = {}
            for i in self.names:
                result[i] = []
            for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
                left, top, width, height = box
                lt = (int(left), int(top))
                rb = (int(left + width), int(top + height))
                center = (int((lt[0] + rb[0]) // 2 ), int(((lt[1] + rb[1]) // 2)))
                data = {
                    'confidence': confidence,
                    'box': [lt, rb],
                    'center': center,
                }
                result[self.names[classId]].append(data)
            if clean:
                for item in result.copy():
                    if not result[item]:
                        result.pop(item)
            return result

    def detect_only(self, img):
        if not self.accesable:
            raise AttributeError('Model not loaded')
        if Index.PROVIDER == 'onnxruntime' or Index.PROVIDER == 'openvino' or Index.PROVIDER == 'onnxdml':
            boxes, scores, class_ids = self.detector(img)
            return boxes, scores, class_ids
        elif Index.PROVIDER == 'opencv':
            classes, confidences, boxes = self.detector(img, confThreshold=self.detector.conf, nmsThreshold=self.detector.nms)
            return classes, confidences, boxes


class TemplateMatch:
    def __init__(self):
        pass

    @staticmethod
    def input_type(background, template):
        if type(background) == str:
            bg = cv2.imread(background)
        elif type(background) == np.ndarray:
            bg = background
        else:
            raise ValueError('Unknown type of background')
        if type(template) == str:
            tp = cv2.imread(template)
        elif type(template) == np.ndarray:
            tp = template
        else:
            raise ValueError('Unknown type of template')
        return bg, tp

    def single(self, background, template, match_threshold, return_extra=False, show=False, save=False, draw=False):
        bg, tp = self.input_type(background, template)

        bg_pic = cv2.cvtColor(bg, cv2.COLOR_BGR2BGRA)
        tp_pic = cv2.cvtColor(tp, cv2.COLOR_BGR2BGRA)
        res = cv2.matchTemplate(bg_pic, tp_pic, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)  # 寻找最优匹配

        if max_val < match_threshold:
            if return_extra:
                return -1, -1, -1
            else:
                return -1
        # 绘制方框
        th, tw = tp_pic.shape[:2]
        tl = max_loc  # 左上角点的坐标
        br = (tl[0] + tw, tl[1] + th)  # 右下角点的坐标

        if draw:
            cv2.rectangle(bg, tl, br, (255, 255, 255), 2)  # 绘制矩形
            cv2.putText(
                bg,
                str(round(max_val, 3)),
                (tl[0], tl[1] - 10),
                cv2.FONT_ITALIC,
                0.5,
                (0, 0, 255),
                2,
            )

        if show:
            cv2.imshow('result', bg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if save:
            cv2.imwrite('result.png', bg)

        center = (tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2
        box = (tl, br)
        if return_extra:
            return center, max_val, box
        else:
            return center

    def multi(self, background, template, match_threshold, return_extra=False, show=False, save=False, draw=False):
        bg, tp = self.input_type(background, template)

        bg_pic = cv2.cvtColor(bg, cv2.COLOR_BGR2BGRA)
        tp_pic = cv2.cvtColor(tp, cv2.COLOR_BGR2BGRA)

        ret_list = []

        res = cv2.matchTemplate(bg_pic, tp_pic, cv2.TM_CCOEFF_NORMED)
        # 寻找全部符合要求的匹配
        loc = np.where(res >= match_threshold)
        if len(loc[0]) == 0:
            if return_extra:
                return -1, -1, -1
            else:
                return -1
        # 绘制方框
        th, tw = tp_pic.shape[:2]
        for pt in zip(*loc[::-1]):
            tl = pt
            br = (tl[0] + tw, tl[1] + th)  # 右下角点的坐标
            center = (tl[0] + br[0]) // 2, (tl[1] + br[1]) // 2
            threshold = round(res[pt[1], pt[0]], 2)
            box = (tl, br)

            if return_extra:
                ret_list.append((center, threshold, box))
            else:
                ret_list.append(center)
            if draw:
                cv2.rectangle(bg, tl, br, (255, 255, 255), 2)
                cv2.putText(
                    bg,
                    str(round(res[pt[1], pt[0]], 3)),
                    (tl[0], tl[1] - 10),
                    cv2.FONT_ITALIC,
                    0.5,
                    (0, 0, 255),
                    2,
                )

        if show:
            cv2.imshow('result', bg)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        if save:
            cv2.imwrite('result.png', bg)

        return ret_list


if __name__ == '__main__':
    """init('onnxruntime', True)
    model = Model()
    model.load('../model/atkfp16.onnx', 0.3, 0.5, r'../model/names_test.names')
    img = cv2.imread('../img_5.png')

    time_list = []

    for i in range(50):
        t1 = time.time()
        result = model.detect(img)
        t2 = time.time()
        print("耗时：", round((t2 - t1), 2) * 1000, "ms")
        time_list.append(round((t2 - t1), 2) * 1000)

    time_list.pop(0)
    print("平均耗时：", sum(time_list) / len(time_list), "ms")"""

    """init('opencv', False)
    model = Model()
    model.load('..\model\\50k.cfg', '..\model\\50k.weights', (416, 416), r'..\model\\names_test.names', 0.3, 0.5)
    img = cv2.imread('..\img_5.png')
    time_list = []
    for i in range(50):
        t1 = time.time()
        result = model.detect(img)
        t2 = time.time()
        print("耗时：", round((t2 - t1), 2) * 1000, "ms")
        time_list.append(round((t2 - t1), 2) * 1000)

    time_list.pop(0)
    print("平均耗时：", sum(time_list) / len(time_list), "ms")"""
    pass
    init('openvino', False)
    model = Model()
    model.load('../model/best.onnx', 0.3, 0.5, r'../model/names_test.names')
    img = cv2.imread('../img_5.png')
    result = model.detect(img)
    time_list = []
    for i in range(50):
        t1 = time.time()
        result = model.detect(img)
        t2 = time.time()
        time_list.append(round((t2 - t1), 2) * 1000)
    time_list.pop(0)
    print(" Openvino 平均耗时：", sum(time_list) / len(time_list), "ms")

    init('opencv', False)
    model = Model()
    model.load('..\model\\50k.cfg', '..\model\\50k.weights', (416, 416), r'..\model\\names_test.names', 0.3, 0.5)
    img = cv2.imread('..\img_5.png')
    time_list = []
    for i in range(50):
        t1 = time.time()
        result = model.detect(img)
        t2 = time.time()
        time_list.append(round((t2 - t1), 2) * 1000)
    time_list.pop(0)
    print("opencv dnn 平均耗时：", sum(time_list) / len(time_list), "ms")

    init('onnxruntime', False)
    model = Model()
    model.load('../model/atkfp16.onnx', 0.3, 0.5, r'../model/names_test.names')
    img = cv2.imread('../img_5.png')

    time_list = []

    for i in range(50):
        t1 = time.time()
        result = model.detect(img)
        t2 = time.time()
        time_list.append(round((t2 - t1), 2) * 1000)

    time_list.pop(0)
    print("onnxruntime 平均耗时：", sum(time_list) / len(time_list), "ms")