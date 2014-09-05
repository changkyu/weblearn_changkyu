Object Detection using Convolutional Neural Network with Patches

Author: Changkyu Song
Contact: changkyusong86@gmail.com (https://sites.google.com/site/changkyusong86)

This project is an simple implementation of object detection using Convolutional Neural Network with an unsupervised or semi-supervised way.
The project is based on the Yangqing's DeCAF project and employs his code published on http://www.eecs.berkeley.edu/~jiayq/.
(please refer software/3rdparty/decaf)

****************
* Requirements *

To run this code, you should install python, ipython and related libraries.
(You may be able to find those libraries on the web very easily, or some tools allow you install them automatically)

And also it uses a libsvm tool, liblinear-weights-1.94 (http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/)

*****************************************
* Simple Description of the source code *

changkyu.py: main script to train/test. You should correct some pathes in this file. (ex> the path of dataset)
ObjectDetector.py: detect object
Classifier.py: execute libsvm and return the results. You should correct the path of libsvm in this file.
download_images.py: a tool to download and rename images from web
objdet_mil.py: not implemented yet

*******
* Run *

1) Run ipython
ipython --matplotlib

2) Import changkyu.py
ipython>> import changkyu

Then the main function will be excuted automatically, and it will run training/testing code.
If you want to edit the training/testing process, you can edit the main() function in changkyu.py

***********
* Results *

The detection results would be stored at software/results/
