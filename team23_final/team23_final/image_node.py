#!/usr/bin/env python

from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from rclpy.qos import QoSProfile, QoSDurabilityPolicy, QoSReliabilityPolicy, QoSHistoryPolicy
import rclpy
from classify_service.srv import Classify
import cv2
import csv
import sys
from std_msgs.msg import Float64MultiArray
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from rclpy.node import Node

# Setting parameter values
t_lower = 90  # Lower Threshold
t_upper = 100  # Upper threshold

# ROS 2


class ImageNode(Node):

    def __init__(self):
        super().__init__('imagenode')

        # Set Parameters

        self.declare_parameter('show_image_bool', True)
        self.declare_parameter('window_name', "Raw Image")

        # Determine Window Showing Based on Input
        self._display_image = bool(self.get_parameter('show_image_bool').value)

        # Declare some variables
        self._titleOriginal = self.get_parameter(
            'window_name').value  # Image Window Title

        # Only create image frames if we are not running headless (_display_image sets this)
        if (self._display_image):
            # Set Up Image Viewing
            cv2.namedWindow(self._titleOriginal,
                            cv2.WINDOW_AUTOSIZE)  # Viewing Window
            # Viewing Window Original Location
            cv2.moveWindow(self._titleOriginal, 50, 50)

        """Initialize image subscriber and classifier client."""
        # Set up QoS Profiles for passing images over WiFi

        image_qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            durability=QoSDurabilityPolicy.RMW_QOS_POLICY_DURABILITY_VOLATILE,
            depth=1
        )
        self.image_subscriber = self.create_subscription(
            CompressedImage, '/camera/image/compressed', self.imageCallback, image_qos_profile)
        self.image_subscriber
        print('Waiting for classifier service to come up...')

        self.cli = self.create_client(Classify, 'classify')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = Classify.Request()

    def classify_client(self, vector):

        self.req.feature_vector = vector
        self.future = self.cli.call_async(self.req)
        print(self.future)
        response = self.future.result()
        print(response.results)
        print('back2')
        return self.future

        # rclpy.wait_for_service('/classifier_node/classify')
        # self.classify_client = rclpy.ServiceProxy('/classifier_node/classify', Classify)

    def imageCallback(self, image):
        """Process an image and return the class of any sign in the image."""
        print('here')
        # The "CompressedImage" is transformed to a color image in BGR space and is store in "_imgBGR"
        self._imgGray = CvBridge().compressed_imgmsg_to_cv2(image, "mono8")
        if (self._display_image):
            # Display the image in a window
            cv2.imshow('frame', self._imgGray)

        ############################################################################################
        # Begin image processing code (You write this!)
        test = np.array([np.array(self._imgGray)])
        print(test.shape)

        # blur image
        blur = cv2.medianBlur(test, 5)

        # apply threshold
        ret, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY)

        # applying canny edge detection
        edged = cv2.Canny(thresh, t_lower, t_upper)

        test_new = cv2.resize(edged, (33, 25))

        # finding contours
        (cnts, _) = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        idx = 0

        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if w > 40 and h > 40:
                idx += 1
                new_img = blur[y:y+h, x:x+w]
                test_new = cv2.resize(new_img, (33, 25))

        test_data = test_new.flatten().reshape(1,33*25)

        # normalize data
        test_data = StandardScaler().fit(test_data).transform(test_data.astype(float))
        test_data = test_data.reshape(test_data.shape[1])
        # TODO: Fill this in with the features you extracted from the image
        feature_vector = list(test_data)
      

        # End image processing code
        ############################################################################################
        
        classification = self.classify_client(feature_vector)
        print('Classified image as: ' + str(classification.result))


def main(args=None):
    rclpy.init(args=args)
    print("I'm here")
    image_node = ImageNode()

    rclpy.spin(image_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    print("Done!")
    image_node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    try:
        main()
    except rclpy.ROSInterruptException:
        pass
