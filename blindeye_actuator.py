

#------------------ OBJECT DETECTION & CLASSIFICATION ------------------------------
import argparse
import time
import cv2
from datetime import datetime
file_name='Video'+datetime.now().strftime('_%d_%m_%H_%M_%S')+'.avi'

from datetime import datetime
from PIL import Image as Image_pil
#from pydarknet import Detector, Image
from utils import is_moving, object_interaction, save_object, str2bool, draw_bounding_box

#------------------ GPIO INTERACTION ------------------------------
import Jetson.GPIO as GPIO

# Pin Definitions
input_pin = 18  # BCM pin 18, BOARD pin 12
pin_servo1= 13
pin_servo2=15
led_pins=[40,38,33]

#------------------------- IMPROVED VISUALS ------------------------------

from PIL import ImageFont, ImageDraw
from PIL import Image as Image_pil
import numpy as np
#------------------------- JETSON CAMERA OPENING ------------------------------

from camera_controller import open_cam_usb, open_cam_rtsp, open_cam_onboard

import logging
logger = logging.getLogger('TfPoseEstimator-WebCam')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
#------------------------- ARGUMENT DEFINITION ------------------------------

def parse_args():
    global file_name  #CHECK IF GLOBAL IS REALLY NEEDED HERE
    # Parse input arguments
    desc = 'Capture and display live camera video on Jetson TX2/TX1'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--objects', type=list, default=['person','cell phone','cup','keyboard','mouse'], 
                        help='list of objects to detect')
    parser.add_argument('--device',type=str,default='jetson',
                        help='if using a laptop or a jetson')
    parser.add_argument('--system', type=str, required=False,default='Other', help='Enter \'Mac\' or \'Other\'')
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
    parser.add_argument('--camera', type=int, default=1)
    parser.add_argument('--input_type', type=str, default='cam',
                        help='Camera or video')

    parser.add_argument('--width', dest='image_width',
                        help='image width [960]',
                        default=640, type=int)
    parser.add_argument('--height', dest='image_height',
                        help='image height [720]',
                        default=480, type=int)

    parser.add_argument('--save_video', type=bool, default=True,
                        help= 'To write output video.')
    parser.add_argument('--video_input', type=str,
                        help= 'File of the video to analyze')
    parser.add_argument('--video_file',type=str,default=file_name,
                        help='File to store the video, by default is todays date')
    parser.add_argument('--demo',dest='demo',
                        help='type of video demo we are running: total, objects, persons',
                        default='total', type=str)

#    parser.add_argument()
    args = parser.parse_args()
    return args

# -----------------------------  RUNNING CODE ---------------------------------------

if __name__ == "__main__":

    args = parse_args()
    print('Called with args:')
    print(args)

    if args.system == 'Other':
        from Linux_Unix_Ubuntu import load_pydarknet
        from pydarknet import Detector, Image
 
        net = load_pydarknet(args, logger)
    else:
        from Mac import load_YOLO
        LABELS, COLORS, ln, net = load_YOLO(args.labels, args.weights, args.cfg)

    #Input type:
    if (args.input_type == 'cam' and args.device=='jetson'):
        cap = open_cam_usb(args.camera,args.image_width,args.image_height)
    elif (args.input_type == 'cam' and args.device=='laptop'):
        cap = cv2.VideoCapture(args.camera)
    elif(args.input_type == 'video'):
        cap = cv2.VideoCapture(args.video_input)

    #Video write
    frame_width=int(cap.get(3))
    frame_height=int(cap.get(4))
    print(frame_width,frame_height)
    #Video write
    if (args.save_video==True):
        if frame_width > args.image_width:
                frame_width=args.image_width
                frame_height=args.image_height
        out= cv2.VideoWriter(args.video_file,cv2.VideoWriter_fourcc('M','J','P','G'),5,(frame_width,frame_height))
    
    count = 0
    count_f=0
    frames=0
    
    #Imrpoved visuals fonts
    font_title=ImageFont.truetype('helvetica-bold.ttf',round(frame_width*0.025))
    font_subtitle=ImageFont.truetype('helvetica-bold.ttf',round(frame_width*0.022))
    font_objects=ImageFont.truetype('helvetica-bold.ttf',round(frame_width*0.03))
    color_labels=(255,255,255)
    #GPIO SETUP
    GPIO.setmode(GPIO.BOARD)  
    GPIO.setup(input_pin, GPIO.IN)  # set pin as an input pin
    GPIO.setup(pin_servo1, GPIO.OUT)
    GPIO.setup(pin_servo2, GPIO.OUT)
    GPIO.setup(led_pins, GPIO.OUT)    
    objects='Detected: '
    while True:
        detect_obj=[]
        
        start_time=time.time()
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
            
            if args.system=='Other':
                for cat, score, bounds in results:
                    x, y, w, h = bounds
                    #cv2.putText(image, str(cat.decode("utf-8")), (int(x-w/2), int(y-h/2)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))
                    cv2.rectangle(image, (int(x-w/2),int(y-h/2)),(int(x+w/2),int(y+h/2)),(55,0,0))
            ####################################################

            ################################
            ######### ONLY FOR MAC #########
            ################################

            if (args.demo == 'total' or args.demo == 'objects') and args.system == 'Mac':
                from Mac import run_YOLO
                idxs, boxes, classIDs, confidences = run_YOLO(image, ln, args.confidence, args.threshold, net)


            if args.system == 'Mac':
                from Mac import annotate_image
                annotate_image(image, idxs, boxes, COLORS, LABELS, confidences, classIDs)
            ####################################################

            im_rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            pil_img=Image_pil.fromarray(im_rgb)
            draw=ImageDraw.Draw(pil_img)   

            for cat, score, bounds in results:
                x, y, w, h = bounds
                detect_obj.append(str(cat.decode("utf-8")))
                draw.text((int(x-w/2), int(y-h/2)-round(frame_width*0.021)), str(cat.decode("utf-8")),font=font_objects,fill=color_labels)
            s=', '
            objects=s.join(detect_obj)    
            
# -----------------------------  GPIO INTERACTION ---------------------------------------
            try:
                if 'Cup' in objects and 'Bottle' in objects:
                    value_str = "HIGH"
                    GPIO.output(led_pins,GPIO.LOW)
                    GPIO.output(led_pins[2],GPIO.HIGH)
                    GPIO.output(pin_servo1,GPIO.LOW)
                    GPIO.output(pin_servo2,GPIO.HIGH)
                    draw.text((10, 10), 'Object: Cup + Bottle'+' : Bringing the bottle opener',font=font_objects,fill=(0,255,0))
                elif 'Cup' in objects:
                    value_str = "LOW"
                    GPIO.output(led_pins,GPIO.LOW)
                    GPIO.output(led_pins[1],GPIO.HIGH)
                    GPIO.output(pin_servo1,GPIO.LOW)
                    GPIO.output(pin_servo2,GPIO.LOW)
                    draw.text((10, 10), 'Object: Cup',font=font_objects,fill=(255,153,51))
                else:
                    value_str = "LOW"
                    GPIO.output(led_pins,GPIO.LOW)
                    GPIO.output(led_pins[0],GPIO.HIGH)
                    GPIO.output(pin_servo1,GPIO.LOW)
                    GPIO.output(pin_servo2,GPIO.LOW)
                    draw.text((10, 10), 'Objects: ',font=font_objects,fill=(255,0,0))
                print("Value read from pin {} : {}".format(input_pin,
                                                           value_str))
                prev_value = value
                time.sleep(1)
            except: pass
# ----------------------------- SHOW ---------------------------------------

            end_time = time.time()
            fps = 1 / (end_time - start_time)
            
            image=cv2.cvtColor(np.array(pil_img),cv2.COLOR_RGB2BGR)
            #cv2.putText(image, 'FPS: '+ str(round(fps,2)), (10, 10), cv2.FONT_HERSHEY_SIMPLEX,
      #  0.35, (255,255,255), 2)
        cv2.imshow('Basic YOLO', image)

        if (args.save_video == True):
             out.write(image)

        end_time_out = time.time()
   
        frames+=1
        k = cv2.waitKey(1)
        if k == 0xFF & ord("q"):
            cap.release()
            break
print(frames) 
cap.release()
out.release()
GPIO.output(led_pins,GPIO.LOW)
GPIO.cleanup()
print(' camera released, GPIO cleaned')


#------------------------- Data Frame Save  -------------------------------------
#for obj in args.objects:
#    vars()[obj+'_data']=pd.DataFrame({'time':vars()[obj+'_track_time'],'x':vars()[obj+'_track_x'],'y':vars()[obj+'_track_y']})
#    vars()[obj+'_data'].to_csv('output/'+obj+'_data.csv')



