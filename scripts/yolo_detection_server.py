#! /usr/bin/env python

## @package yolo_bounding_box_detection
# The ROS node for detecting bounding boxes with the YOLO model
#
# Defines a ROS action server for detecting bounding boxes with the YOLO model. The yolov5 repository from https://github.com/ultralytics/yolov5 has to be pulled next to this package

import rospy
import actionlib
import rospkg
from cv_bridge import CvBridge
from vision_msgs.msg import Detection2D
from bark_msgs.msg import BoundingBoxDetectionAction, BoundingBoxDetectionResult

import numpy as np
import os
import sys
import torch


## BoundingBoxDetectionServer class
#
# Defines a ROS action server for predicting bounding boxes with the YOLO model
class BoundingBoxDetectionServer:
  '''
    Class for YOLO predictions
  '''

  ## Constructor of BoundingBoxDetectionServer class
  # @param model_name string type name of the file containing the trained model's weights (pt file in the 'models' directory of this repository)
  def __init__(self, model_name="best"):

    ## @var rospack
    #  rospkg.RosPack() object for getting ROS package paths
    self.rospack = rospkg.RosPack()
    rospackage_root = self.rospack.get_path("yolo_bounding_box_detection")

    sys.path.insert(0, os.path.join(os.path.split(rospackage_root)[0], 'yolov5'))
    from models.common import DetectMultiBackend
    from utils.torch_utils import select_device
    from utils.augmentations import letterbox
    from utils.general import non_max_suppression, scale_coords, xyxy2xywh

    ## @var letterbox
    #  letterbox function from yolov5/utils/augmentations
    self.letterbox = letterbox

    ## @var non_max_suppression
    #  non_max_suppression function from yolov5/utils/general
    self.non_max_suppression = non_max_suppression

    ## @var scale_coords
    #  scale_coords function from yolov5/utils/general
    self.scale_coords = scale_coords

    ## @var xyxy2xywh
    #  xyxy2xywh function from yolov5/utils/general
    self.xyxy2xywh = xyxy2xywh

    ## @var device
    #  Hardware acceleration (CPU)
    self.device = select_device('')

    weights_file_path = os.path.join(rospackage_root, "models", model_name + ".pt")

    ## @var model
    #  the YOLO model
    self.model = DetectMultiBackend(weights_file_path, device=self.device, dnn=False)
    rospy.loginfo("YOLO model loaded: " + model_name + ".pt")

    ## @var bridge
    #  CvBridge() object for conversions between numpy arrays and ROS Image message types
    self.bridge = CvBridge()

    ## @var server
    #  The ROS action server for the BoundingBoxDetectionAction action. The name of the action is "bounding_box_detection"
    self.server = actionlib.SimpleActionServer('bounding_box_detection', BoundingBoxDetectionAction, self.execute, False)
    self.server.start()


  ## BoundingBoxDetectionAction callback
  # This function gets called whenever the ROS action server receives a goal from a client
  # @param goal bark_msgs/BoundingBoxDetectionGoal type action goal, it contains an image for bounding box detection (see action definition for further details)
  def execute(self, goal):
    '''
    BoundingBoxDetectionAction callback

    goal: bark_msgs/BoundingBoxDetectionGoal, action goal, it contains an image for bounding box detection (see action definition for further details)
    '''
    img = self.bridge.imgmsg_to_cv2(goal.image, desired_encoding='bgr8')
    # Get greater dimension of image
    img_larger_shape = img.shape[1] if img.shape[1] >= img.shape[0] else img.shape[0]

    # Pad the image
    padded = self.letterbox(img, img_larger_shape, self.model.stride, self.model.pt and not self.model.jit)[0]
    padded = padded.transpose((2, 0, 1))[::-1]
    padded = np.ascontiguousarray(padded)

    if len(padded.shape) == 3:
      # Add batch dimension if there is none
      padded = padded[None]

    # Create Torch tensor from padded image
    img_tensor = torch.from_numpy(padded).to(self.device)
    img_tensor = img_tensor.float()
    img_tensor /= 255

    # Predict with the model
    pred = self.model(img_tensor)
    pred = self.non_max_suppression(pred)

    detection_result = []

    # Get x,y,w,h values from predictions
    for i, det in enumerate(pred):
      gn = torch.tensor(img.shape)[[1, 0, 1, 0]]  # normalization gain whwh

      if len(det):
        det[:, :4] = self.scale_coords(img_tensor.shape[2:], det[:, :4], img.shape).round()

        for *xyxy, conf, classif in reversed(det):
          xywh = (self.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()
          x_min = (xywh[0]-xywh[2]/2)*img.shape[1]
          y_min = (xywh[1]-xywh[3]/2)*img.shape[0]
          w = xywh[2]*img.shape[1]
          h = xywh[3]*img.shape[0]

          detection_result.append((conf.item(), int(classif.item()), (x_min,y_min,w,h)))

    detection_result.sort()
    detection_result = list(reversed(detection_result))
    result = BoundingBoxDetectionResult()

    for detected_box in detection_result:
      detection = Detection2D()
      detection.bbox.center.x = detected_box[2][0] + detected_box[2][2]/2
      detection.bbox.center.y = detected_box[2][1] + detected_box[2][3]/2
      detection.bbox.size_x = detected_box[2][2]
      detection.bbox.size_y = detected_box[2][3]
      result.detections.detections.append(detection)
    
    self.server.set_succeeded(result)

if __name__ == '__main__':
  rospy.init_node('yolo_detection_server')
  node_name = rospy.get_name()
  model_name = rospy.get_param("/" + node_name + "/yolo_model_name")
  server = BoundingBoxDetectionServer(model_name)
  rospy.spin()