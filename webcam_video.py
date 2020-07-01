#--------------------------------- SERVER LAUNCH --------------------------------
#!/usr/bin/env python3
from __future__ import print_function   # For Py2/3 compatibility

#import atexit
#import http.server
#import inspect
#import json
import os
#import socketserver
#import traceback
#import importlib
#import wrapper
#import http.server
#import socketserver
import eel

#--------------------------------- POSE DETECTION --------------------------------
from utils import analyze_pose, generate_output
import argparse
import logging
import time
import subprocess
# sys.path.insert(0,'/usr/local/lib/python3.7/site-packages')
# print(sys.path)
import cv2
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

status_tracker = {"SITTING" : [], "STANDING": [], "LAYING": [], "COOKING": [], "NOTHING": []}
status = "NOTHING"
factor = None
pose = ' '

timer = 0

#------------------ OBJECT DETECTION & CLASSIFICATION ------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
from PIL import Image as Image_pil
from pydarknet import Detector, Image
from utils import is_moving, object_interaction, save_object, str2bool, draw_bounding_box

saved=[]
count=0

general_objects=['Person','Cell phone','Book','Clock','Chair']

objects_recognised='The camera recognised: '

#------------------------- JETSON CAMERA OPENING ------------------------------

from camera_controller import open_cam_usb, open_cam_rtsp, open_cam_onboard

#------------------------- ACTIVITY CLASSIFICATION ------------------------------
from activity_classification import locate_activity,describe_activity
#------------------------- IMPROVED VISUALS ------------------------------

from PIL import ImageFont, ImageDraw
from PIL import Image as Image_pil

#------------------------- ARGUMENT DEFINITION ------------------------------

def parse_args():
    global file_name  #CHECK IF GLOBAL IS REALLY NEEDED HERE
    # Parse input arguments
    desc = 'Capture and display live camera video on Jetson TX2/TX1'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--objects', type=list, default=['person','cell phone','cup','keyboard','mouse'], 
                        help='list of objects to detect')
    parser.add_argument('--device',type=str,default='laptop',
                        help='if using a laptop or a jetson')
    parser.add_argument('--system', type=str, required=False, help='Enter \'Mac\' or \'Other\'')
    ## For Object detection
    parser.add_argument('--cfg', type=str, default='cfg/yolov3.cfg', help='*.cfg path for object detection model')
    parser.add_argument('--weights', type=str, default='weights/yolov3.weights', help='path to weights file for object detection models')
    parser.add_argument('--labels', type=str, default='cfg/coco.names', help='path to coco name file')
    parser.add_argument('--thresh', dest='thresh',
                        help='Object Detection Threshold',
                        default=0.5, type=float)
    parser.add_argument('--show', dest='show_img',
                        help='Show image display 0/1',
                        default=1, type=int)
    parser.add_argument('--confidence', type=float, default=0.5,
	                    help="minimum probability to filter weak detections")
    parser.add_argument("-t", "--threshold", type=float, default=0.3,
	                    help="threshold when applying non-maxima suppression")
    ## For Pose Detection
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--input_type', type=str, default='cam',
                        help='Camera or video')

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

    parser.add_argument('--save_video', type=bool, default=True,
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

# -----------------------------  RUNNING CODE ---------------------------------------

if __name__ == "__main__":
    result = ""
    objects="Objects: "
    objects_detected=' '
    args = parse_args()
    print('Called with args:')
    print(args)

    for obj in args.objects:
        vars()[obj+'s']=[]
        vars()[obj+'_mov']=False
        vars()['dist_r_'+obj]=1000
        vars()['dist_l_'+obj]=1000
        vars()[obj+'_track_time']=[]
        vars()[obj+'_track_x']=[]
        vars()[obj+'_track_y']=[]

    if args.system == 'Other':
        from Linux_Unix_Ubuntu import load_pydarknet
        net = load_pydarknet(args, logger)
    else:
        from Mac import load_YOLO
        LABELS, COLORS, ln, net = load_YOLO(args.labels, args.weights, args.cfg)

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
    
    frame_width=int(cap.get(3))
    frame_height=int(cap.get(4))
    #Video write
    if (args.save_video==True):   
        if frame_width > args.image_width:
                frame_width=args.image_width
                frame_height=args.image_height
        out= cv2.VideoWriter(args.video_file,cv2.VideoWriter_fourcc('M','J','P','G'),25,(frame_width,frame_height))
    
    count = 0
    count_f=0
    frames=0
    
    #Imrpoved visuals fonts
    font_title=ImageFont.truetype('helvetica-bold.ttf',round(frame_width*0.025))
    font_subtitle=ImageFont.truetype('helvetica-bold.ttf',round(frame_width*0.022))
    font_objects=ImageFont.truetype('helvetica-bold.ttf',round(frame_width*0.019))
    font_inter=ImageFont.truetype('helvetica-bold.ttf',round(frame_width*0.021))
    font_count=ImageFont.truetype('helvetica-bold.ttf',round(frame_width*0.018))
    
    while True:
        objects_tolist=''
        for obj in args.objects:
            #vars()[obj+'s']=[]
            vars()[obj+'_detect']=False
            vars()[obj+'_mov']=False
            vars()['dist_r_'+obj]=args.image_width
            vars()['dist_l_'+obj]=args.image_width

        r, image = cap.read()
        # r = True
        # image = cv2.imread('cooking2.jpg')
        # image=cv2.resize(image,(frame_width,frame_height))
        if r:
            if  frame_width >= args.image_width:
                image=cv2.resize(image,(args.image_width,args.image_height))
                frame_width=args.image_width
                frame_height=args.image_height
            start_time = time.time()

            # Only measure the time taken by YOLO and API Call overhead
            
            ## Object ##

            ################################
            ## ONLY FOR LINUX/UNIX/UBUNTU ##
            ################################

            if (args.demo == 'total' or args.demo == 'objects') and args.system == 'Other':
                results = net.detect(Image(image),args.thresh) #LOOK INTO THIS FUNCTION.
            else:
                results=[]
            #Draw boxes, no names yet.
            if args.system=='Other':
                for cat, score, bounds in results:
                    x, y, w, h = bounds
                    cv2.rectangle(image, (int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(55,0,0))
            ####################################################

            ################################
            ######### ONLY FOR MAC #########
            ################################

            if (args.demo == 'total' or args.demo == 'objects') and args.system == 'Mac':
                from Mac import run_YOLO
                idxs, boxes, classIDs, confidences = run_YOLO(image, ln, args.confidence, args.threshold, net)


            if args.system == 'Mac':
                from Mac import annotate_image_box
                annotate_image_box(image, idxs, boxes, COLORS, LABELS, confidences, classIDs)
            ####################################################

            ## Pose ##
            if (args.demo == 'total' or args.demo == 'persons'):
                humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=args.resize_out_ratio)
            else:
                humans=[]
            #Draw humans now so they are in the background
            if (args.demo == 'total' or args.demo == 'persons'):
                body_color=(128,0,0)
                image = TfPoseEstimator.draw_humans(image, humans, body_color, imgcopy=False)
                
    
    
    
    
            #Change image to draw better text   
    
            im_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            pil_img=Image_pil.fromarray(im_rgb)
            draw=ImageDraw.Draw(pil_img)   
#--------------------- OBJECT ANALYSIS ----------------------------------------

            start_time_obj = time.time()

            ################################
            ## ONLY FOR LINUX/UNIX/UBUNTU ##
            ################################

            if args.system == 'Other':

                if results:
                   # print('objects detected')
                    
                    for obj in results:
                        for measure in args.objects:
                            if (measure in str(obj)):
                                vars()[measure+'_detect']=True
                                vars()[measure+'s'].append(obj)
                                vars()[measure+'_mov']=is_moving(vars()[measure+'s'],20)
                                vars()[measure+'_index']=results.index(obj)
                                vars()[measure+'_track_time'].append((datetime.now().strftime('%d/%m/%H:%M:%S')))
                                vars()[measure+'_track_x'].append(obj[-1][0])
                                vars()[measure+'_track_y'].append(args.image_height-obj[-1][1])

                   #Count number of objects
                    for cat, score, bounds in results:
                        objects_tolist+=str(cat.decode("utf-8"))+','
                    objects_list=objects_tolist.split(',')
                    objects_list.remove('')
                    objects_uniq=list(set(objects_list))
                    #draw.text((frame_width*0.83,5),'Detected objects:',font=font_count)
                    for i in range(len(objects_uniq)):
                         if objects_uniq[i] in objects_list: 
                             obj_count=objects_list.count(objects_uniq[i])
                         else:   
                             obj_count=0
                       #  draw.text((frame_width*0.83,round(frame_width*0.021)*(i+1.69)),objects_uniq[i]+' : '+str(obj_count),font=font_count)
                       
                    for cat, score, bounds in results:
                        x, y, w, h = bounds
                        objects+=str((str(cat.decode("utf-8")),round(score,3)))
                       # print(objects)
                        if cat.decode("utf-8")=='cup':
                            if saved:
                            #    print(saved)
                                for obj in saved:
                                    #print(obj[0])
                                    if ('cup' not in obj[0]):
                                        count=0
                                        save_object(cat, score, bounds)
                                    else:
                                        count+=1
                                        # print(saved)
                                if count >=5:
                                    saved.remove(obj)
                            else:
                                save_object()
                        else:
                            if saved:
                                count_f+=1
                                if count_f >= 3:
                                    saved.clear()
                                    count_f=0
                    # print(saved)

                    #Distance from hand to object
                    for human in humans:
                        r_wrist = human.body_parts.get(4, None)
                        l_wrist = human.body_parts.get(7, None)

                        for obj in args.objects:
                            if (vars()[obj+'_detect']==True):
                                if r_wrist != None :
                                    vars()['dist_r_'+obj]=object_interaction(results[vars()[obj+'_index']][2],r_wrist.x*args.image_width,r_wrist.y*args.image_height)

                                if l_wrist != None :
                                    vars()['dist_l_'+obj]=object_interaction(results[vars()[obj+'_index']][2],l_wrist.x*args.image_width,l_wrist.y*args.image_height)
                                    #    print( vars()['dist_r_'+obj])

                    #Establishing interaction
                        dist_thresh=400
                        for obj in args.objects:
                            if vars()[obj+'_detect']==True:
                                 if (#vars()[obj+'_mov']==True# or
                                     vars()['dist_r_'+obj]<dist_thresh or vars()['dist_l_'+obj]<dist_thresh):
                                     pass
                             #print('Interaction with a ',obj,vars()[obj+'_mov'],round(vars()['dist_r_'+obj],2),round(vars()['dist_l_'+obj],2))
                             #draw.text((10,frame_width*0.023),"Interaction with a: %s" %obj,font=font_inter)
                             #cv2.putText(image,
                             #        "INTERACTION with a: %s" %obj , #(1.0 / (time.time() - fps_time)),
                             #        (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                             #        (255, 255, 255),2)
                            # for cat, score, bounds in results:
                            #    x, y, w, h = bounds
                            #    if cat.decode("utf-8") == obj:
                             #      cv2.rectangle(image, (int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(55,155,100))
            ####################################################
            
            ################################
            ######### ONLY FOR MAC #########
            ################################

     #       if args.system == 'Mac':
      #          from Mac import annotate_image
       #         annotate_image_box(image, idxs, boxes, COLORS, LABELS, confidences, classIDs)

            ####################################################

            end_time_obj = time.time()
           # print('TIME OBJECT: ',end_time_obj-start_time_obj,' + ',(1/(end_time_obj-start_time_obj)))

#--------------------- POSE ANALYSIS ----------------------------------------

            start_time_pos = time.time()
            if  humans:
                #for human in humans:
                    #print(human.body_parts)
                poses, updated_status_tracker = analyze_pose(humans, image, movement_tracker_x, movement_tracker_y, status_tracker)
            end_time_pos = time.time()

            ## Delete when actually testing ##
     #       try:
     #           print('TIME POSE: ',end_time_pos-start_time_pos,' + ',(1/(end_time_pos-start_time_pos)))
     #       except ZeroDivisionError:
     #           print('No pose detected')
     #           break

#--------------------- activity output ----------------------------------------
            initial_x=10
            initial_y=10
            label_step=round(frame_width*0.023)
            color_font=(255,255,255)
            video_title='Home Care - Kitchen'
            draw.text((initial_x,initial_y),video_title,font=font_title,fill=color_font)
      #      if frames <80: video_subtitle='Studying scene..'
     #       elif frames<190: video_subtitle='Kids Playing'
           # video_subtitle='Wheelchair user in the kitchen'
            #draw.text((initial_x,initial_y+label_step),video_subtitle,font=font_subtitle,fill=color_font)
          #  if frames <50: video_subtitle2='Performance'
          # elif frames <100: video_subtitle2='Performance: stable movement with oil bottle'
          #  elif frames<375: video_subtitle2='Performance: good accuracy'
          
            #draw.text((initial_x,initial_y+1*label_step),video_subtitle2,font=font_subtitle,fill=color_font)
 #           color_font1=(255,255,255)
  #          if frames <75: video_subtitle3='Level of independence: 3 / 5'
   #         elif frames<200: 
    #            color_font1=(255,255,255)
     #           video_subtitle3='Level of independence: 4 / 5'
      #      elif frames<=375:
       #         video_subtitle3='Level of independence: 4 / 5 - Check fire temperature'
        #        color_font1=(255,0,0)
     
            #draw.text((initial_x,initial_y+2*label_step),video_subtitle3,font=font_subtitle,fill=color_font1)
            
            
            initial_x2=10
            initial_y2=10+round(frame_width*0.024)
            label_step=-round(frame_width*0.024)
            try:
                rooms,main_room=locate_activity(objects_list)
                room_label='In the: ' + main_room+' with : '+str(rooms[main_room])
                draw.text((initial_x2,initial_y2),room_label,font=font_objects)
            except:
                room_label='In the: ' + 'Estimating room...'
                draw.text((initial_x2,initial_y2),room_label,font=font_objects)

            label_count=1
            try:
                activities,main_activity=describe_activity(objects_list)
                act_obj=[(obj,objects_list.count(obj)) for obj in objects_uniq if obj in activities[main_activity]]                 
                activity_label=(main_activity + ' using '+ str(act_obj))
                draw.text((initial_x2,initial_y2-label_step),activity_label,font=font_objects)
                for key_act in activities:
                    if key_act not in main_activity:
                        label_count+=1
                        act_obj2=[(obj,objects_list.count(obj)) for obj in objects_uniq if obj in activities[key_act]]
                        label_extra=key_act+' using: '+str(act_obj2)
                        draw.text((initial_x2,initial_y2-label_step*label_count),label_extra,font=font_objects)
            except:
                #label_count=0
                activity_label='Estimating activity...'
                draw.text((initial_x2,initial_y2-label_step),activity_label,font=font_objects)

            try:
                    other_objects=[(obj,objects_list.count(obj)) for obj in objects_uniq if obj in general_objects]
                    label_general='General objects detected: '+str(other_objects)
                    draw.text((initial_x2,initial_y2-label_step*(label_count+1)),label_general,font=font_objects)
            except: 
                pass
            
#--------------------- OUTPUT ----------------------------------------
            start_time_out=time.time()
            #Showing
            if (args.show_img ==1 or args.save_video == True):
                ##Object
              if (args.black == True):
                image=image-image

              for cat, score, bounds in results:
                x, y, w, h = bounds
                   
                if (y-h/2)<(initial_y+4*label_step) and (x-w/2) < (frame_width/4):                    
                    print(str(cat.decode("utf-8")),(y-h/2))
           #     if str(cat.decode("utf-8")) == 'Person' : draw.text((int(x-w/2), int(y-h/2)-round(frame_width*0.021)), 'Child',font=font_objects,fill=color_font)       
                else: draw.text((int(x-w/2), int(y-h/2)-round(frame_width*0.021)), str(cat.decode("utf-8")),font=font_objects,fill=color_font)
              
              if args.system == 'Mac':
                from Mac import annotate_image_label
                annotate_image_label(draw, idxs, boxes, COLORS, LABELS, confidences, classIDs,frame_width,font_objects,color_font)
#                ##Saved Object
#              if (saved):
#                for cat, score, bounds in saved:
#                   x, y, w, h = bounds
#                   cv2.rectangle(image, (int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(255,0,0))
#                   cv2.putText(image,(cat), (int(x-w/2), int(y-h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))


##Pose


   



              end_time = time.time()
              fps = 1 / (end_time - start_time)
             # print(fps)
#              for pose in poses:  
      #            draw.text((10,5),"Fps: %.2f " % (round(fps,2)) + status + pose,font=font_title)
#                  cv2.putText(image,
#                        "FPS: %f " % (fps) + status + pose,
#                        (10, 10),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
#                        (255, 255, 255), 2)

              image=cv2.cvtColor(np.array(pil_img),cv2.COLOR_RGB2BGR)
              if (args.show_img == True):
                  cv2.imshow('tf-pose-estimation result', image)

              if (args.save_video == True):
                # image=cv2.resize(image,(frame_width,frame_height))
                out.write(image)
                #cv2.imwrite('cooking_camera2.jpg',image)


# print(results) Seems like results is a dict with the name,probability, bounds(4 puntos).
        else:
            print("Video complete or not camera detected")
            break

        end_time_out = time.time()
       # print('TIME OUTPUT: ',end_time_out-start_time_out,' + ',(1/(end_time_out-start_time_out)))
        frames+=1
        k = cv2.waitKey(1)
        if k == 0xFF & ord("q"):
            break
print(frames) 
out.release()
cap.release()


#------------------------- Data Frame Save  -------------------------------------
for obj in args.objects:
    vars()[obj+'_data']=pd.DataFrame({'time':vars()[obj+'_track_time'],'x':vars()[obj+'_track_x'],'y':vars()[obj+'_track_y']})
    vars()[obj+'_data'].to_csv('output/'+obj+'_data.csv')


#------------------------- Heatmap save  -------------------------------------

colors = [  'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
cont=1
for obj in args.objects:
    plt.figure()
    plt.hist2d(vars()[obj+'_data'].iloc[:,1],vars()[obj+'_data'].iloc[:,2], bins=[np.arange(0,960,20),np.arange(0,720,20)],cmap=colors[cont])
    plt.savefig('output/'+obj+'_track_plt.png')
    cont+=1


#sns.set_style("whitegrid")
cont=1
for obj in args.objects:
    plt.figure()
    ax=sns.kdeplot(vars()[obj+'_data'].iloc[:,1],vars()[obj+'_data'].iloc[:,2],cmap=colors[cont], shade=True, shade_lowest=False,gridsize=100)
    ax.set_frame_on(False)
    plt.xlim(0, 960)
    plt.ylim(0, 720)
    plt.axis('off')
    fig = ax.get_figure()
    fig.savefig('output/'+obj+'_track_sns.png', transparent=False, bbox_inches='tight', pad_inches=0)
    cont+=1

cont=1
plt.figure()
for obj in args.objects:
    ax=sns.kdeplot(vars()[obj+'_data'].iloc[:,1],vars()[obj+'_data'].iloc[:,2],cmap=colors[cont], shade=True, shade_lowest=False,gridsize=100)
    ax.set_frame_on(False)
    plt.xlim(0, 960)
    plt.ylim(0, 720)
    plt.axis('off')
    cont+=1
fig = ax.get_figure()
fig.savefig('output/'+'total_track_sns.png', transparent=False, bbox_inches='tight', pad_inches=0)
# save your KDE to disk


print(status_tracker)
try:
   final_end_pos = poses[-1]
except:
   final_end_pos=(args.image_width/2, args.image_height/2)

final_end = time.time()

cv2.destroyAllWindows()
previous_json_string = None

# status_tracker = {'SITTING': [{'start': 1, 'start_pos': 10, 'end': 2, 'end_pos': 12}],
# 'STANDING': [{'start': 1, 'start_pos': 10, 'end': 2, 'end_pos': 12}],
# 'LAYING': [{'start': 1, 'start_pos': 10, 'end': 2, 'end_pos': 12}],
# 'COOKING': [{'start': 1, 'start_pos': 10, 'end': 2, 'end_pos': 12}],
# 'NOTHING': []}
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
    def get_sitting(): return construct_output(updated_status_tracker['SITTING'], "sitting")
    @eel.expose   # Expose this function to Javascript
    def get_standing(): return construct_output(updated_status_tracker['STANDING'], "standing")
    @eel.expose
    def get_laying(): return construct_output(updated_status_tracker['LAYING'], "laying")
    @eel.expose
    def get_cooking(): return construct_output(updated_status_tracker['COOKING'], "cooking")
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
