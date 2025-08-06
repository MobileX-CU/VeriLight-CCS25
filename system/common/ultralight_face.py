"""
Ultralight face detector. 
Note the UltralightFace GitHub must be cloned into this directory as well
"""
import argparse
import sys
import cv2
import numpy as np
import os 
sys.path.insert(0, f"{os.path.expanduser('~')}/deepfake_detection/system/e2e/common/Ultra-Light-Fast-Generic-Face-Detector-1MB")
from vision.ssd.config.fd_config import define_img_size

class UltraLightFaceDetector(object):
    def __init__(self, network, device, threshold):
        input_img_size = 480
        define_img_size(input_img_size)  # must put define_img_size() before 'import create_mb_tiny_fd, create_mb_tiny_fd_predictor'

        from vision.ssd.mb_tiny_fd import create_mb_tiny_fd, create_mb_tiny_fd_predictor
        from vision.ssd.mb_tiny_RFB_fd import create_Mb_Tiny_RFB_fd, create_Mb_Tiny_RFB_fd_predictor
        from vision.utils.misc import Timer

     
        label_path = f"{os.path.expanduser('~')}/deepfake_detection/system/e2e/common/Ultra-Light-Fast-Generic-Face-Detector-1MB/models/voc-model-labels.txt"

        class_names = [name.strip() for name in open(label_path).readlines()]
        num_classes = len(class_names)
        self.device = device
        self.candidate_size = 1000
        self.threshold = threshold

        if network == 'slim':
            model_path = f"{os.path.expanduser('~')}/deepfake_detection/system/e2e/common/Ultra-Light-Fast-Generic-Face-Detector-1MB/models/pretrained/version-slim-320.pth"
            # model_path = "models/pretrained/version-slim-640.pth"
            net = create_mb_tiny_fd(len(class_names), is_test=True, device = self.device)
            self.predictor = create_mb_tiny_fd_predictor(net, candidate_size = self.candidate_size, device = self.device)
        elif network == 'RFB':
            model_path = f"{os.path.expanduser('~')}/deepfake_detection/system/e2e/common/Ultra-Light-Fast-Generic-Face-Detector-1MB/models/pretrained/version-RFB-320.pth"
            # model_path = "models/pretrained/version-RFB-640.pth"
            net = create_Mb_Tiny_RFB_fd(len(class_names), is_test=True, device = self.device)
            self.predictor = create_Mb_Tiny_RFB_fd_predictor(net, candidate_size = self.candidate_size, device = self.device)
        else:
            print("The net type is wrong!")
            sys.exit(1)
        net.load(model_path)
    
    def detect(self, orig_image):
        """
        Return face bounding box of highest confidence given an input frame
        """
      
        image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
        boxes, labels, probs = self.predictor.predict(image, self.candidate_size / 2, self.threshold)
        boxes = np.array(boxes.cpu().tolist()).astype(int)
        if boxes.shape[0] == 0:
            return []
        probs = np.array(probs.cpu().tolist())
        boxes = boxes[np.argsort(probs)[::-1]]
        top_box = boxes[0,:].tolist()
        return top_box