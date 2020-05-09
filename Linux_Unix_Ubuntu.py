import pydarknet
from pydarknet import Detector, Image
from tf_pose.networks import get_graph_path
import cv2
import datetime
from utils import is_moving, save_object, object_interaction

def load_pydarknet(args, logger):
    # Optional statement to configure preferred GPU. Available only in GPU version.
    pydarknet.set_cuda_device(0)

    #Object
    if (args.demo == 'total' or args.demo == 'objects'):
        net = Detector(bytes(args.cfg, encoding="utf-8"), bytes(args.weights, encoding="utf-8"), 0,
                        bytes("cfg/coco.data", encoding="utf-8"))
    ##Pose
    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))

    return net

def analyze_results(image, results, humans, height, width, object_list, objects, objects_detected, saved):
    count_f = 0

    for obj in object_list:
        vars()[obj+'s']=[]
        vars()[obj+'_mov']=False
        vars()['dist_r_'+obj]=1000
        vars()['dist_l_'+obj]=1000
        vars()[obj+'_track_time']=[]
        vars()[obj+'_track_x']=[]
        vars()[obj+'_track_y']=[]

    if results:
        print('objects detected')
        
        for obj in results:
            for measure in object_list:
                if (measure in str(obj)):
                    vars()[measure+'_detect']=True
                    vars()[measure+'s'].append(obj)
                    vars()[measure+'_mov']=is_moving(vars()[measure+'s'],20)
                    vars()[measure+'_index']=results.index(obj)
                    vars()[measure+'_track_time'].append((datetime.now().strftime('%d/%m/%H:%M:%S')))
                    vars()[measure+'_track_x'].append(obj[-1][0])
                    vars()[measure+'_track_y'].append(height-obj[-1][1])

        for cat, score, bounds in results:
            x, y, w, h = bounds
            objects+=str((str(cat.decode("utf-8")),round(score,3)))
            print(objects)
            if cat.decode("utf-8")=='cup':
                if saved:
                    print(saved)
                    for obj in saved:
                        print(obj[0])
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

            for obj in object_list:
                if (vars()[obj+'_detect']==True):
                    if r_wrist != None :
                        vars()['dist_r_'+obj]=object_interaction(results[vars()[obj+'_index']][2],r_wrist.x*width,r_wrist.y*height)

                    if l_wrist != None :
                        vars()['dist_l_'+obj]=object_interaction(results[vars()[obj+'_index']][2],l_wrist.x*width,l_wrist.y*height)
                        #    print( vars()['dist_r_'+obj])

        #Stablishing interaction
        dist_thresh=150
        for obj in object_list:
            if vars()[obj+'_detect']==True:
                if (vars()[obj+'_mov']==True or vars()['dist_r_'+obj]<dist_thresh or vars()['dist_l_'+obj]<dist_thresh):
                    print('Interaction with a ',obj,vars()[obj+'_mov'],round(vars()['dist_r_'+obj],2),round(vars()['dist_l_'+obj],2))
                    cv2.putText(image,
                            "INTERACTION with a: %s" %obj , #(1.0 / (time.time() - fps_time)),
                            (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (255, 255, 255),2)
        

    # return all the vars()
