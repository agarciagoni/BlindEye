#--------------------------------- SERVER LAUNCH --------------------------------
#!/usr/bin/env python3
from __future__ import print_function   # For Py2/3 compatibility

#import atexit
#import http.server
#import inspect
#import json
#import os
#import socketserver
#import traceback
#import importlib
#import wrapper
#import http.server
#import socketserver
import eel

#--------------------------------- POSE DETECTION --------------------------------

import argparse
import logging
import time
import sys
import subprocess
sys.path.insert(0,'/usr/local/lib/python3.7/site-packages')
print(sys.path)
import cv2
#import numpy as np
import math
from math import sqrt
from datetime import datetime
file_name='Video'+datetime.now().strftime('_%d_%m_%H_%M_%S')+'.avi'

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh
#import os

logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

movement_tracker_x = {}
movement_tracker_y = {}

sitting_counter = 0
standing_counter = 0
laying_counter = 0
nothing_counter = 0
cooking_counter = 0

status_tracker = {"SITTING" : [], "STANDING": [], "LAYING": [], "COOKING": [], "NOTHING": []}
status = "NOTHING"
factor = None
pose = ' '
number_assignments = ["nose", "neck", "r_shoulder", "l_shoulder", "r_hip", "l_hip", "r_knee", "l_knee", "l_ankle", "r_ankle", "r_eye", "l_eye", "r_ear", "l_ear", "r_elbow", "l_elbow", "l_wrist", "r_wrist"]
detection_tracker = {}
for part in number_assignments:
    detection_tracker[part] = 0

timer = 0

#RECORDING COORDINATE

def record_coordinate(point):
    
    """
    Records a new coordinate to keep track of where the person is
    throughout the room as the day goes on. 
    """
    
    
    x_key = int(round(point[0]))
    y_key = int(round(point[1]))

    if x_key in movement_tracker_x:
        movement_tracker_x[x_key] += 1
    else:
        movement_tracker_x[x_key] = 1
    if y_key in movement_tracker_y:
        movement_tracker_y[y_key] += 1
    else:
        movement_tracker_y[y_key] = 1
    
    #print(movement_tracker_x)
    #print(movement_tracker_y)
def generate_output(arm_angle, spine_angle, wrist_head_dist, rounded_center):
                if arm_angle:
                    arm_angle = "       Arm Angle: " + str(round(arm_angle))
                else:
                    arm_angle = "       Calculating... "
                if spine_angle: 
                    spine_angle = "       Spine Angle: " + str(round(spine_angle))
                else:
                    spine_angle = "       Calculating... "
                if wrist_head_dist:
                    wrist_head_dist = "       Wrist-Head Distance: " + str(round(wrist_head_dist))
                else:
                    wrist_head_dist = "       Calculating... "
                if rounded_center:
                    rounded_center = "       Center: " + str(rounded_center)
                else:
                    rounded_center = "       Calculating... "
                return arm_angle + spine_angle + wrist_head_dist + rounded_center

#------------------ OBJECT DETECTION & CLASSIFICATION ------------------------------

saved=[]
count=0
measure_objects=['cup','laptop','keyboard','cellphone','bottle']
for obj in measure_objects:
    
    vars()[obj+'s']=[]
    vars()[obj+'_mov']=False
    vars()['dist_r_'+obj]=1000
    vars()['dist_l_'+obj]=1000
    
objects_recognised='The camera recognised: '    
#Fron Object Detection.
#import time
from datetime import datetime
#import pydarknet
from pydarknet import Detector, Image
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def is_moving(obj,thresh):
    obj_mov=False
    if (len(obj)>1):
       x,y,w,h = obj[-1][2]
       x_old,y_old,w_old,h_old = obj[-2][2]
       dist=sqrt((x-x_old)**2 + (y-y_old)**2)
    #   print(dist)
       if dist > thresh:
    #       print(obj[-1][0].decode('utf-8'),' is moving')
           obj_mov=True
    return obj_mov
  
def object_interaction(obj,wrist_x,wrist_y):
    x=obj[0]
    y=obj[1]
    dist=sqrt((x-wrist_x)**2 + (y-wrist_y)**2)
    return dist

def save_object():
    global saved
    if count==0:
        saved.append((str(cat.decode("utf-8")),round(score,3),bounds))
    return saved


#------------------------- JETSON CAMERA OPENING ------------------------------

def open_cam_rtsp(uri, width, height, latency):
    gst_str = ('rtspsrc location={} latency={} ! '
               'rtph264depay ! h264parse ! omxh264dec ! '
               'nvvidconv ! '
               'video/x-raw, width=(int){}, height=(int){}, '
               'format=(string)BGRx ! '
               'videoconvert ! appsink').format(uri, latency, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
def open_cam_usb(dev, width, height):
    # We want to set width and height here, otherwise we could just do:
    #     return cv2.VideoCapture(dev)
    gst_str = ('v4l2src device=/dev/video{} ! '

               'video/x-raw, width=(int){}, height=(int){} ! '
               'videoconvert ! appsink').format(dev, width, height)
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)


def open_cam_onboard(width, height):
    gst_elements = str(subprocess.check_output('gst-inspect-1.0'))
    if 'nvcamerasrc' in gst_elements:
        # On versions of L4T prior to 28.1, add 'flip-method=2' into gst_str
        gst_str = ('nvcamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)2592, height=(int)1458, '
                   'format=(string)I420, framerate=(fraction)30/1 ! '
                   'nvvidconv ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
    elif 'nvarguscamerasrc' in gst_elements:
        gst_str = ('nvarguscamerasrc ! '
                   'video/x-raw(memory:NVMM), '
                   'width=(int)1920, height=(int)1080, '
                   'format=(string)NV12, framerate=(fraction)30/1 ! '
                   'nvvidconv flip-method=2 ! '
                   'video/x-raw, width=(int){}, height=(int){}, '
                   'format=(string)BGRx ! '
                   'videoconvert ! appsink').format(width, height)
    else:
        raise RuntimeError('onboard camera source not found!')
    return cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)

#------------------------- ARGUMENT DEFINITION ------------------------------

def parse_args():
    global file_name  #CHECK IF GLOBAL IS REALLY NEEDED HERE
    # Parse input arguments
    desc = 'Capture and display live camera video on Jetson TX2/TX1'
    parser = argparse.ArgumentParser(description=desc)
    
    parser.add_argument('--device',type=str,default='laptop',
                        help='if using a laptop or a jetson')
##Created for Object detection
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='*.cfg path for object detectin model')
    parser.add_argument('--weights', type=str, default='weights/yolov3.weights', help='path to weights file for object detection models')
    parser.add_argument('--thresh', dest='thresh',
                        help='Object Detection Threshold',
                        default=0.5, type=float)
    parser.add_argument('--show', dest='show_img',
                        help='Show image display 0/1',
                        default=1, type=int)
##From Pose Detection
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--input_type', type=str, default='cam', 
                        help='Wheather to use camera or video')

    parser.add_argument('--resize', type=str, default='0x0',
                        help='if provided, resize images before they are processed. default=0x0, Recommends : 432x368 or 656x368 or 1312x736 ')

    parser.add_argument('--resize-out-ratio', type=float, default=4.0,
                        help='if provided, resize heatmaps before they are post-processed. default=1.0')

    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')

    
    parser.add_argument('--tensorrt', type=str, default="False",
                        help='for tensorrt process.')
    
    parser.add_argument('--rtsp', dest='use_rtsp',
                        help='use IP CAM (remember to also set --uri)',
                        action='store_true')
    parser.add_argument('--uri', dest='rtsp_uri',
                        help='RTSP URI, e.g. rtsp://192.168.1.64:554',
                        default=None, type=str)
    parser.add_argument('--latency', dest='rtsp_latency',
                        help='latency in ms for RTSP [200]',
                        default=200, type=int)
    parser.add_argument('--usb', dest='use_usb',
                        help='use USB webcam (remember to also set --vid)',
                        action='store_true')
    parser.add_argument('--vid', dest='video_dev',
                        help='device # of USB webcam (/dev/video?) [1]',
                        default=1, type=int)
    parser.add_argument('--width', dest='image_width',
                        help='image width [960]',
                        default=1920, type=int)
    parser.add_argument('--height', dest='image_height',
                        help='image height [720]',
                        default=1080, type=int)
    
    parser.add_argument('--save_video', type=bool, default=False,
                        help= 'To write output video.')
    parser.add_argument('--video_input', type=str, 
                        help= 'File of the video to analyze')
    parser.add_argument('--video_file',type=str,default=file_name,
                        help='File to store the video, by default is todays date')
    parser.add_argument('--demo',dest='demo',
                        help='type of video demo we are running: total, objects, persons',
                        default='total', type=str)
    
    parser.add_argument('--server', type=bool, default=False,
                        help= 'Option to launch a html with data')
    parser.add_argument('--black', type=bool, default=False,
                        help= 'Option to only show detected image black the rest')
#    parser.add_argument()
    args = parser.parse_args()
    return args

# -----------------------------  VIDEO WRITE ---------------------------------------

    
# -----------------------------  RUNING CODE ---------------------------------------
    
if __name__ == "__main__":
    result = ""
    objects_detected=' ' 
    objects="Objects: "
    center = None
    arm_angle = None
    spine=None
    spine_angle=None
    angle=None
    args = parse_args()
    print('Called with args:')
    print(args)

    # Optional statement to configure preferred GPU. Available only in GPU version.
    #pydarknet.set_cuda_device(0)
#Object
    if (args.demo == 'total' or args.demo == 'objects'):
        net = Detector(bytes(args.cfg, encoding="utf-8"), bytes(args.weights, encoding="utf-8"), 0,
                       bytes("cfg/coco.data", encoding="utf-8"))
##Pose

 #   logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resize)
    if (args.demo == 'total' or args.demo == 'persons'):   
        if w > 0 and h > 0:
            e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h), trt_bool=str2bool(args.tensorrt))
        else:
            e = TfPoseEstimator(get_graph_path(args.model), target_size=(432, 368), trt_bool=str2bool(args.tensorrt))
    logger.debug('cam read+')

#Input type:
    
    if (args.input_type == 'cam' and args.device=='jetson'):
        cap = open_cam_usb(args.camera,args.image_width,args.image_height)
    elif (args.input_type == 'cam' and args.device=='laptop'):
        cap = cv2.VideoCapture(args.camera)
    elif(args.input_type == 'video'):
        cap = cv2.VideoCapture(args.video_input)

#Video write

    if (args.save_video==True):
        frame_width=int(cap.get(3))
        frame_height=int(cap.get(4))
        print(cap.get(3))
        print(frame_width)
        out= cv2.VideoWriter(args.video_file,cv2.VideoWriter_fourcc('M','J','P','G'),10,(frame_width,frame_height))
    count = 0
    y1 = [0, 0]
    frame = 0
    while True:
        for obj in measure_objects:   
            #vars()[obj+'s']=[]
            vars()[obj+'_detect']=False
            vars()[obj+'_mov']=False
            vars()['dist_r_'+obj]=args.image_width
            vars()['dist_l_'+obj]=args.image_width
            
        r, image = cap.read()
       # r = True
       # image = cv2.imread('cooking2.jpg')
      #  image=cv2.resize(image,(frame_width,frame_height))
        if r:
            start_time = time.time()
          

            # Only measure the time taken by YOLO and API Call overhead
##Object
            if (args.demo == 'total' or args.demo == 'objects'):
                dark_frame = Image(image)
                results = net.detect(dark_frame,args.thresh) #LOOK INTO THIS FUNCTION.
                del dark_frame
            else: 
                results=[]
##Pose
            #logger.debug('image process+')
            if (args.demo == 'total' or args.demo == 'persons'):
                humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
            else:
                humans=[]

            #print(humans) 

           # print("\n")
           # print(datetime.now())
           #print("FPS: ", fps)

#--------------------- OBJECT ANALYSIS ----------------------------------------
            start_time_obj = time.time()
            if results:
                for obj in results:
                    for measure in measure_objects:
                        if (measure in str(obj)):
                           vars()[measure+'_detect']=True
                           vars()[measure+'s'].append(obj)
                           vars()[measure+'_mov']=is_moving(vars()[measure+'s'],20)
                           vars()[measure+'_index']=results.index(obj)
     
                    # save_object()


                for cat, score, bounds in results:
                    x, y, w, h = bounds
                    objects+=str((str(cat.decode("utf-8")),round(score,3)))
                    #print(objects)
                    if cat.decode("utf-8")=='cup':
                        
                        if ('cup' not in saved):  
                            count=0
                            save_object()
                        else:
                            count+=1
                        print(saved)
                    if count >= 5:
                        saved=[]
                    
    
    
    
    #Distance from hand to object
                for human in humans:
                    r_wrist = human.body_parts.get(4, None)		
                    l_wrist = human.body_parts.get(7, None)
                    
                    for obj in measure_objects:
                         if (vars()[obj+'_detect']==True):
                               if r_wrist != None :
                                   vars()['dist_r_'+obj]=object_interaction(results[vars()[obj+'_index']][2],r_wrist.x*args.image_width,r_wrist.y*args.image_height)
         #                           print( vars()['dist_r_'+obj])
                            
                               if l_wrist != None :
                                   vars()['dist_l_'+obj]=object_interaction(results[vars()[obj+'_index']][2],l_wrist.x*args.image_width,l_wrist.y*args.image_height)
                                #    print( vars()['dist_r_'+obj])                 
    
                
                
    #Stablishing interaction
                dist_thresh=150
                for obj in measure_objects:
                    if vars()[obj+'_detect']==True:
                         if (vars()[obj+'_mov']==True or vars()['dist_r_'+obj]<dist_thresh or vars()['dist_l_'+obj]<dist_thresh):
                             print('Interaction with a ',obj,vars()[obj+'_mov'],round(vars()['dist_r_'+obj],2),round(vars()['dist_l_'+obj],2))
                             cv2.putText(image,
                                      "INTERACTION with a: %s" %obj , #(1.0 / (time.time() - fps_time)),
                                      (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                      (255, 255, 255),2)
            end_time_obj = time.time()              
            print('TIME OBJECT: ',end_time_obj-start_time_obj,' + ',(1/(end_time_obj-start_time_obj)))
               
#--------------------- POSE ANALYSIS ----------------------------------------
                             
            start_time_pos = time.time()                
            if  humans:
                #for human in humans:
                    #print(human.body_parts)

                main_body_detected = True
                other_parts_detected = True
                """
                TRACKING ALL THE TIME
                """
                
                for human in humans:
                    neck = human.body_parts.get(1, None)
                    r_shoulder = human.body_parts.get(2, None)
                    l_shoulder = human.body_parts.get(5, None)
                    r_hip = human.body_parts.get(8, None)
                    l_hip = human.body_parts.get(11, None)
                    r_knee = human.body_parts.get(9, None)
                    l_knee = human.body_parts.get(12, None)
                
                    main_body_detected = all([neck, r_shoulder, l_shoulder, r_hip, l_hip, r_knee, l_knee])
        
                """
                OPTIONAL TO TRACK
                """
                for human in humans:
                    nose = human.body_parts.get(0, None) 
                    l_ankle = human.body_parts.get(13, None)
                    r_ankle = human.body_parts.get(10, None)
                    r_eye = human.body_parts.get(14, None)
                    l_eye = human.body_parts.get(15, None)
                    r_ear = human.body_parts.get(16, None)
                    l_ear = human.body_parts.get(17, None)                
                    r_elbow = human.body_parts.get(3, None)
                    l_elbow = human.body_parts.get(6, None)
                    l_wrist = human.body_parts.get(7, None)
                    r_wrist = human.body_parts.get(4, None)
        
                    other_parts_detected = all([nose, r_knee, l_knee, l_ankle, r_ankle, r_eye, l_eye, r_ear, l_ear, r_elbow, l_elbow, l_wrist, r_wrist])   

                    all_parts_detected = [nose, neck, r_shoulder, l_shoulder, r_hip, l_hip, r_knee, l_knee, l_ankle, r_ankle, r_eye, l_eye, r_ear, l_ear, r_elbow, l_elbow, l_wrist, r_wrist]
        
                    for part_index in range(len(all_parts_detected)):
                        if all_parts_detected[part_index]:
                            part_name = number_assignments[part_index]
                            detection_tracker[part_name] += 1
         
                #CALCULATING SPINE
    
                if nose:
                    nose_x = nose.x*image.shape[1] 
                    nose_y = nose.y*image.shape[0]
     
                if r_eye and l_eye:
                    r_eye_x = r_eye.x*image.shape[1]
                    r_eye_y = r_eye.y*image.shape[0]
                    l_eye_x = l_eye.x*image.shape[1]
                    l_eye_y = l_eye.y*image.shape[0]

                if r_shoulder:
                    r_shoulder_x = r_shoulder.x*image.shape[1]
                    r_shoulder_y = r_shoulder.y*image.shape[0] 
                if l_shoulder:
                    l_shoulder_x = l_shoulder.x*image.shape[1]
                    l_shoulder_y = l_shoulder.y*image.shape[0] 
                if r_shoulder and l_shoulder:
                    #spine = ((top x, top y), (bottom x, bottom y))
                   # r_shoulder_x = r_shoulder.x*image.shape[1]
                   # r_shoulder_y = r_shoulder.y*image.shape[0] 
                  #  l_shoulder_x = l_shoulder.x*image.shape[1]
                  #  l_shoulder_y = l_shoulder.y*image.shape[0]
                    shoulder_center_x = (r_shoulder_x + l_shoulder_x) / 2
                    shoulder_center_y = (r_shoulder_y + l_shoulder_y) /2
    
                if r_knee and l_knee:
                    #KNEES
                    r_knee_x = r_knee.x*image.shape[1]
                    r_knee_y = r_knee.y*image.shape[0]
                    l_knee_x = r_knee.x*image.shape[1]
                    l_knee_y = r_knee.y*image.shape[0]
                    knee_center_x = (r_knee_x + l_knee_x) / 2
                    knee_center_y = (r_knee_y + l_knee_y) /2
    
                if r_ankle and l_ankle:
                    #ANKLES
                    r_ankle_x = r_ankle.x*image.shape[1]
                    r_ankle_y = r_ankle.y*image.shape[0]
                    l_ankle_x = r_ankle.x*image.shape[1]
                    l_ankle_y = r_ankle.y*image.shape[0]
                    ankle_center_x = (r_ankle_x + l_ankle_x) /2
                    ankle_center_y = (r_ankle_y + l_ankle_y) /2
    
                try:
                   top_pairs = [(shoulder_center_x, shoulder_center_y), (nose_x, nose_y)]
                   bottom_pairs = [(knee_center_x, knee_center_y), (ankle_center_x, ankle_center_y)]
            
                   top_pair = None
                   bottom_pair = None
                   for x, y in top_pairs:
                      if x and y:
                          top = (x, y)
                   for x, y in bottom_pairs:
                      if x and y:
                          bottom = (x, y)
                   if top and bottom:
                      spine = (top, bottom)
                except:
                   pass
                def dotproduct(v1, v2):
                    return sum((a*b) for a, b in zip(v1, v2))
                def length(v):
                    return math.sqrt(abs(dotproduct(v, v)))
                def find_angle(v1, v2):
                    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))
        
        
                #CALCULATING ANGLE BETWEEN UPPER LEGS AND SPINE
                if r_hip and l_hip:
                    #HIPS
                    hips = True
                    r_hip_x = r_hip.x*image.shape[1]
                    r_hip_y = r_hip.y*image.shape[0]
                    l_hip_x = r_hip.x*image.shape[1]
                    l_hip_y = r_hip.y*image.shape[0]
                    hip_center_x = (r_hip_x - l_hip_x) /2
                    hip_center_y = (r_hip_y - l_hip_y) /2   
        
                    
        
                    if all([r_shoulder, l_shoulder, r_hip, l_hip, l_knee, r_knee]):
                        v1= [abs(shoulder_center_x - hip_center_x), abs(shoulder_center_y - hip_center_y)]
                        v2= [abs(hip_center_x - knee_center_x), abs(hip_center_y - knee_center_y)]
                        angle = find_angle(v1, v2)
        
                #CALCULATING ANGLE BETWEEN UPPER AND LOWER ARMS
                vectors_r = ""
                vectors_l = ""
                if all([r_shoulder, r_elbow, r_wrist]):
                        r_elbow_x = r_elbow.x*image.shape[1]
                        r_elbow_y = r_elbow.y*image.shape[0]
                        r_wrist_x = r_wrist.x*image.shape[1]
                        r_wrist_y = r_wrist.y*image.shape[0]
                        v1= [(r_shoulder_x - r_elbow_x), (r_shoulder_y - r_elbow_y)]
                        v2= [(r_wrist_x - r_elbow_x), (r_wrist_y - r_elbow_y)]
                        try:
                            arm_angle1 = (find_angle(v1, v2)/3.14)*180 
                            vectors_r = (v1, v2)
                        except:
                            arm_angle1 = None
                if all([l_shoulder, l_elbow, l_wrist]):
                        l_elbow_x = l_elbow.x*image.shape[1]
                        l_elbow_y = l_elbow.y*image.shape[0]
                        l_wrist_x = l_wrist.x*image.shape[1]
                        l_wrist_y = l_wrist.y*image.shape[0]
                        v1= [(l_shoulder_x - l_elbow_x), (l_shoulder_y - l_elbow_y)]
                        v2= [(l_wrist_x - l_elbow_x), (l_wrist_y - l_elbow_y)]
                        try:
                            arm_angle2 = (find_angle(v1, v2)/3.14)*180 
                            vectors_l = (v1, v2)
                        except:
                            arm_angle2 = None

                arm_angle1=None
                arm_angle2=None
                if arm_angle1 and arm_angle2:
                    arm_angle = (arm_angle1 + arm_angle2)/2
                elif arm_angle1:
                    arm_angle = arm_angle1
                else:
                    arm_angle = arm_angle2
                            
                #CALCULATING SPINE ANGLE

                if spine:
                    v1 = [abs(spine[0][0] - spine[1][0]), abs(spine[0][1] - spine[1][1])]
                    v2 = [1, 0]
                    try:
                        spine_angle = (find_angle(v1, v2)/3.14)*180
                    except:
                        spine_angle = None


                #FACTOR CALCULATION - to calculate things relative to this factor
        
                if r_eye and l_eye:
                    r_eye_x = r_eye.x*image.shape[1]
                    r_eye_y = r_eye.y*image.shape[0]
                    l_eye_x = l_eye.x*image.shape[1]
                    l_eye_y = l_eye.y*image.shape[0]
        
                if r_shoulder and l_shoulder:
                    if r_shoulder_x - l_shoulder_x != 0:
                        factor = r_shoulder_x - l_shoulder_x
                elif r_knee and l_knee:
                    if r_knee_x - l_knee_x != 0:
                        factor = r_knee_x - l_knee_x
                if all([r_hip, l_hip, r_knee, l_knee]):
                    factor = abs(r_hip_y - r_knee_y)
                elif all([r_hip, l_hip, r_shoulder, l_shoulder]):
                    factor = abs(r_shoulder_y - r_hip_y)
                elif all([r_eye, l_eye]):
                    
                    factor = abs(r_eye_x - l_eye_x)*2
                    
                    
        
        
                #CALCULATING CENTER POINT FOR MOVEMENT TRACKING
                top = None
                bottom = None  
                wrist_head_dist = None
                if nose:
                    nose_x = nose.x*image.shape[1] 
                    nose_y = nose.y*image.shape[0]
                if neck and factor:
                    neck_x = neck.x*image.shape[1]
                    neck_y = neck.y*image.shape[0] + factor/2
                    top = (neck_x, neck_y)
                elif nose and factor:
                    top = (nose_x, nose_y + factor/4)
                    """
                    if r_eye and l_eye:
                        eye_center_x = (r_eye_x + l_eye_x)/2
                        eye_center_y = (r_eye_y + l_eye_y)/2 + factor/5
                        top = (eye_center_x, eye_center_y)
                    """
                    
                    
        
                #DISTANCE FROM WRISTS TO HEAD
                
                if nose and factor:
                    if r_wrist:
                        if nose:
                            wrist_head_dist = ((nose_x - r_wrist_x)**2 + (nose_y - r_wrist_y)**2)**.5
                        elif l_eye:
                            wrist_head_dist = ((l_eye_x - r_wrist_x)**2 + (l_eye_y - r_wrist_y)**2)**.5
                        elif r_eye:
                            wrist_head_dist = ((r_eye_x - r_wrist_x)**2 + (r_eye_y - r_wrist_y)**2)**.5
                    if l_wrist:
                        if nose:
                            wrist_head_dist = ((nose_x - l_wrist_x)**2 + (nose_y - l_wrist_y)**2)**.5
                        elif l_eye:
                            wrist_head_dist = ((l_eye_x - l_wrist_x)**2 + (l_eye_y - l_wrist_y)**2)**.5
                        elif r_eye:
                            wrist_head_dist = ((r_eye_x - l_wrist_x)**2 + (r_eye_y - l_wrist_y)**2)**.5
        
                
        
                #BOTTOM CENTER
                if r_ankle and l_ankle:
                    bottom = (ankle_center_x, ankle_center_y)
                elif r_knee and l_knee:
                    bottom = (knee_center_x, knee_center_y - factor)
     
                if top and bottom:
                    center = ((top[0] + bottom[0]) / 2, (top[1] + bottom[1]) / 2)
                
                def reset(counters, exception=float('inf')):
                    for i in range(len(counters)):
                        if i != exception:
                            counters[i] = 0
    
                counters = [sitting_counter, standing_counter, laying_counter, cooking_counter, nothing_counter] 

                #DECIDING POSITION 
                if spine and factor:
                    previous_status = status
                    
                   # print("SPINE", spine)
                   # print("ANGLE", angle)
                   # print("FACTOR", factor)
                  #  if spine_angle and spine_angle < 50:
                   #     status = "FALL DETECTED"
                   #     reset(counters)
                    if abs(spine[0][1] - spine[1][1])/factor < 1:
                     #   print("LAYING")
                        status = "LAYING"
                        laying_counter += 1
                        if laying_counter == 3:
                             reset(counters, 2)
                    elif angle and angle < 0.1 :
                          #  print("SITTING") 
                            status = "SITTING"
                            sitting_counter += 1
                            if sitting_counter == 3:
                                reset(counters, 0)
                    elif arm_angle and (abs(spine[0][0] - spine[1][0])/factor) < 1 and abs(arm_angle - 60) < 20:
                       # print("COOKING")
                        cooking_counter += 1
                        if cooking_counter == 3:
                            reset(counters, 3)
                    elif (abs(spine[0][0] - spine[1][0])/factor) < 1:
                      #  print("STANDING")
                        status = "STANDING"
                        standing_counter += 1
                        if standing_counter == 3:
                            reset(counters, 1)
                    else:
                       # print("NOTHING")
                        status = "NOTHING"
                        if nothing_counter == 3:
                            standing_counter = 0
                            reset(counters, 4)
                
        

        
                rounded_center = None
                if center:
                    record_coordinate(center)
                    rounded_center = (round(center[0]), round(center[1]))
        
                    #RECORDING STATUS AT TIME OF DAY
                    for counter in [sitting_counter, standing_counter, laying_counter]:
                        if counter == 3:
                            status_tracker[status].append({"start": time.time(), "start_pos":center})
                            if previous_status != status:
                                last_index = len(status_tracker[previous_status]) - 1
                                status_tracker[previous_status][last_index]["end"] = time.time()
                                status_tracker[previous_status][last_index]["end_pos"] = center
                    if nothing_counter == 3 and previous_status != "NOTHING":
                        last_index = len(status_tracker[previous_status]) - 1
                        status_tracker[previous_status][last_index]["end"] = time.time()
                        status_tracker[previous_status][last_index]["end_pos"] = center
        
                    #print(status_tracker)
                  #  print(detection_tracker)
            
            
            
            
                pose = generate_output(arm_angle, spine_angle, wrist_head_dist, rounded_center)
            end_time_pos = time.time()              
            print('TIME POSE: ',end_time_pos-start_time_pos,' + ',(1/(end_time_pos-start_time_pos)))
 
#--------------------- OUTPUT ----------------------------------------
            start_time_out=time.time()        
#Showing     
            if (args.show_img ==1 or args.save_video == True):
##Object
              if (args.black == True):
                image=image-image

              for cat, score, bounds in results:
                x, y, w, h = bounds
                cv2.rectangle(image, (int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(55,0,0))
                cv2.putText(image, str(cat.decode("utf-8")), (int(x-w/2), int(y-h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

##Saved Object
              if (saved):
                for cat, score, bounds in saved:
                   x, y, w, h = bounds
                   cv2.rectangle(image, (int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(255,0,0))
                   cv2.putText(image,(cat), (int(x-w/2), int(y-h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

              #cv2.namedWindow('preview',cv2.WINDOW_NORMAL)
              #cv2.resizeWindow('preview', 960, 720)
              #cv2.imshow("preview", frame)
##Pose

#              test=image-image
              if (args.demo == 'total' or args.demo == 'persons'):
                  body_color=(128,0,0)
                  image = TfPoseEstimator.draw_humans(image, humans, body_color, imgcopy=False)
            
              #if (args.save_video == True):
              #    out.write(image)
                  
              end_time = time.time()
              fps = 1 / (end_time - start_time)
              cv2.putText(image,
                    "FPS: %f " % (fps) + status + pose,
                    (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 2)

              if (args.show_img == True):
                  cv2.imshow('tf-pose-estimation result', image)

              if (args.save_video == True):
                #  image=cv2.resize(image,(frame_width,frame_height))
                  out.write(image)   
                  #cv2.imwrite('cooking_camera2.jpg',image)            
        
       
# print(results) Seems like results is a dict with the name,probability, bounds(4 puntos).
        else:
            print("Video complete")
            break    
        
        end_time_out = time.time()              
        print('TIME OUTPUT: ',end_time_out-start_time_out,' + ',(1/(end_time_out-start_time_out)))                
        
        k = cv2.waitKey(1)
        if k == 0xFF & ord("q"):
            break


out.release()
cap.release()
print(status_tracker)
print(detection_tracker)
try:
   final_end_pos = rounded_center
except:
   final_end_pos=(args.image_width/2, args.image_height/2)

final_end = time.time()

cv2.destroyAllWindows()
previous_json_string = None

status_tracker = {'SITTING': [{'start': 1, 'start_pos': 10, 'end': 2, 'end_pos': 12}], 
'STANDING': [{'start': 1, 'start_pos': 10, 'end': 2, 'end_pos': 12}], 
'LAYING': [{'start': 1, 'start_pos': 10, 'end': 2, 'end_pos': 12}], 
'COOKING': [{'start': 1, 'start_pos': 10, 'end': 2, 'end_pos': 12}], 
'NOTHING': []}
objects_print="<br> <p> The user used:"+ objects_detected +"</p>"
def construct_output(log, activity):
    output = ""
    for session in log:
        start = str(session['start'])
        start_pos = str(session['start_pos'])
        try:
            end = str(session['end'])
            end_pos = str(session['end_pos'])
        except:
            session['end'] = str(final_end)
            session['end_pos'] = str(final_end_pos)
            end = final_end
            end_pos = final_end_pos
        output += "<br> <p> User was " + activity + " from " + start + " to " + end + ". </p>"
    return output

if (args.server == True):
    eel.init('ui')
    @eel.expose   # Expose this function to Javascript
    def get_sitting(): return construct_output(status_tracker['SITTING'], "sitting")
    @eel.expose   # Expose this function to Javascript
    def get_standing(): return construct_output(status_tracker['STANDING'], "standing")
    @eel.expose
    def get_laying(): return construct_output(status_tracker['LAYING'], "laying")
    @eel.expose
    def get_cooking(): return construct_output(status_tracker['COOKING'], "cooking")
    @eel.expose
    def get_objects(): return objects_print
    eel.start('index.html')


################################################################################
################################################################################




#say_hello_py('Python World!')
#eel.say_hello_js('Python World!')   # Call a Javascript function

"""

# Code to list and serve files.
def ls_path(path):
    return [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]

def cat_file(path):
    with open(path, "r") as f:
        file = f.read()
    return file

def load_json_file(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def cleanup():
    # Free the socket.
    print("CLEANING UP!")
    httpd.shutdown()
    print("CLEANED UP")

# ----------------------------------
# STATIC FILES: GET any path relative to PWD
# ----------------------------------
# redirect "/" to "static/index.html"
RPCServerHandler.register_redirect("", "/ui/index.html")

# ----------------------------------
# RPC API (POST)
# ----------------------------------
# restart: reload student code
# returns None
RPCServerHandler.register_function(lambda d: RPCServerHandler.reload_modules(), 'restart')

# ls: list directory contents
# returns a dictionary { directories: ["abc",...], files: ["abc",..] }
RPCServerHandler.register_function(lambda d: ls_path(d['path']), 'ls')

# cat: read contents of a file
# returns string contents of file
RPCServerHandler.register_function(lambda d: cat_file(d['path']), 'cat')

# load_json: read json object from a file
# returns json object encoded by a file
RPCServerHandler.register_function(lambda d: load_json_file(d['path']), 'load_json')

# call: call student code
# returns return value
RPCServerHandler.register_module("wrapper")
# ----------------------------------

atexit.register(cleanup)

# Start the server.
print("serving files and RPCs at port", PORT)
httpd.serve_forever()
"""
