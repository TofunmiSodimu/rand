#!/usr/bin/env python

# ROS2
import rclpy
from rclpy.node import Node
import rospkg
import os

# numpy
import numpy

# scikit-learn (some common classifiers are included as examples, feel free to add your own)
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.externals import joblib
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle

class train_classifier(Node):
    def __init__(self):
        super().__init__('Lab6_Final')
        self.training()

    def training(self):
        """Train a classifier and save it as a .pkl for later use."""

        filepath = '/home/michelangelo/ros2_ws_tofunmi/src/team23_final/data/training/training_data.csv'
        # if len(filepath) > 0 and filepath[0] != '/':
        #     rospack = rospkg.RosPack()
        #     filepath = rospack.get_path('image_classifier') + '/data/training/' + filepath

        split = 0.4

        data, label = self.load_data(filepath)
        print('\nImported', data.shape[0], 'training instances')

        data, label = shuffle(data, label)
        data_train, data_test, label_train, label_test = train_test_split(data, label, test_size=split)

        print('Training classifier...')

        ##########################################################################
        # Begin classifier initialization code (You write this!)

        # Initialize the classifier you want to train with the parameters you want here:
        # Install scikit-learn with these instructions: http://scikit-learn.org/stable/install.html
        # Models, documentation, instructions, and examples can be found here:
        #   http://scikit-learn.org/stable/supervised_learning.html#supervised-learning

        classifier = KNeighborsClassifier(n_neighbors = 2,p=2) # TODO: Replace this with the classifier you want

        # End image processing code (You write this!)
        ############################################################################################

        classifier.fit(data_train, label_train)

        print('Detailed results on a %0.0f/%0.0f train/test split:' % ((1 - split)*100, split*100))
        predicted = classifier.predict(data_test)
        print(metrics.classification_report(label_test, predicted))
        print(metrics.confusion_matrix(label_test, predicted))

        print('Training and saving a model on the full dataset...')
        classifier.fit(data, label)

        joblib.dump(classifier, '/home/michelangelo/ros2_ws_tofunmi/src/team23_final/data/classifier/team23_classifier.pkl')
        print('Saved model classifier.pkl to given directory.')


    def load_data(self,filepath):
        """Parse training data and labels from a .csv file."""
        data = numpy.loadtxt(filepath, delimiter=',')
        x = data[:, :data.shape[1] - 1]
        y = data[:, data.shape[1] - 1]
        return x, y

def main(args=None):
    rclpy.init(args=args)
    print("I'm here")
    train = train_classifier()

    # Destroy the node explicitly
    # (optional - otherwise it will be done automatically
    # when the garbage collector destroys the node object)
    print("Done!")
    train.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    try:
        main()
    except rclpy.ROSInterruptException:
        pass
