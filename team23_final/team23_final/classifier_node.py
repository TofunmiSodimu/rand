#!/usr/bin/env python

# ROS2
from classify_service.srv import Classify
from rclpy.node import Node
import numpy as np
import rclpy
import sys
import rospkg

# scikit-learn
#from sklearn.externals import joblib
import joblib



class ClassifierNode(Node):

    def __init__(self):
        """Initialize classifier service in a ROS node."""
        super().__init__('classifiernode')
        self.srv = self.create_service(Classify, 'classify', self.classify_callback)

        # Load the previously-trained classifier model
        filepath = '/home/michelangelo/ros2_ws_tofunmi/src/team23_final/data/classifier/team23_classifier.pkl'
        self.model = joblib.load(filepath)

    def classify_callback(self, req, response):
        """Return binary classification of an ordered grasp pair feature vector."""
        response.results = int(self.model.predict(np.asarray(req.feature_vector).reshape(1, -1)))
        print(response.results)
        return response

def main(args=None):
    rclpy.init(args=args)
    print("I'm here")
    classifier_node = ClassifierNode()

    rclpy.spin(classifier_node)

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    print("Done!")
    classifier_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

