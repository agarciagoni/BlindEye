# BlindEye

The aim of this project is to develop an activity recognition algorithm with different applications ranging from elderly care, anomaly and alarm detection to architectural robotics or learning gamification.

In this repository we include the code that start by estimating the users position and detecting the objects and their interactions. The codes are based on the forked repo (https://github.com/ildoonet/tf-pose-estimation) and the YOLO python implementation from https://github.com/madhawav/YOLO3-4-Py

<p align="center">
  <img src="data/general.gif" width="300" height="225" align="center">
</p>

## Own work

The team working in the project has added several features to the original code.

   1. User position: standing / sitting / laying / falling based on pose estimation.
   2. Body angles and distances.
   3. Object detection using YOLOv3
   4. Measuring interaction with objects based on user distance and object movement.
   5. Video output
   6. UI with activity recognition (work in progress).

## Install & Downloads

#### Pose Estimation
**Requires TensorFlow 1.14.0 and Python 3.7.7 or below.** If you have Python 3.8, run this project on a virtual environment.
Install through requirements:
```bash
$ pip3 install -r requirements.txt
```
And build c++ library for post processing
```bash
$ cd tf_pose/pafprocess
$ swig -python -c++ pafprocess.i && python3 setup.py build_ext --inplace
```

Or install it as a package
```bash
$ python setup.py install  # Or, `pip install -e .`
```
#### Object detection
**For Linux/Unix/Ubuntu**
If not installed with the requirements
```bash
$ pip3 install yolo34py #for CPU only
$ pip3 install yolo34py-gpu #GPU Accelared version
```

**For Mac**
pydarknet is incompatible with MacOS. Install ______ with these commands:

To download different models and weights go to the original darknet source https://pjreddie.com/darknet/yolo/. Recommended the YOLOV3 or YOLOV3-tiny.

## To run

```bash
$ python3 blindeye.py --resize 432x368
```
The resize argument is optional but recommended with a good ratio fps/accuracy.

#### Interesting arguments

You can find an explanation of the main customizable arguments here https://github.com/agarciagoni/BlindEye/blob/master/arguments.txt

An example on how to run the code with several arguments:
```bash
$ python3 blindeye.py --device jetson --resize 432x368 --input_type cam --camera 1 --width 960 --height 720 --demo total --cfg cfg/yolov3.cfg --weights weights/yolov3.weights --thresh 0.7 --tensorrt True --save_video True --server True
```
