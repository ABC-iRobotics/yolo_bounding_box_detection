#! /usr/bin/env python

import rospy
import actionlib
from cv_bridge import CvBridge
import rospkg

import numpy as np
import cv2
import os

from bark_msgs.msg import BoundingBoxDetectionAction, BoundingBoxDetectionGoal

if __name__ == '__main__':
    rospy.init_node('yolo_detection_client')

    rospack = rospkg.RosPack()
    rospackage_root = rospack.get_path("yolo_bounding_box_detection")

    client = actionlib.SimpleActionClient('bounding_box_detection', BoundingBoxDetectionAction)
    client.wait_for_server()
    bridge = CvBridge()

    goal = BoundingBoxDetectionGoal()
    img = cv2.imread(os.path.join(rospackage_root, '..', '4_Color.png'))
    goal.image = bridge.cv2_to_imgmsg(img, encoding='bgr8')
    # Fill in the goal here
    client.send_goal(goal)
    client.wait_for_result(rospy.Duration.from_sec(5.0))
    print(client.get_result())