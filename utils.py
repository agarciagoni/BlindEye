import math
import time
import cv2
import numpy as np

## ---------------Object Detection -----------------##
def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def is_moving(obj,thresh):
    obj_mov=False
    if (len(obj)>1):
       x,y,w,h = obj[-1][2]
       x_old,y_old,w_old,h_old = obj[-2][2]
       dist=math.sqrt((x-x_old)**2 + (y-y_old)**2)
    #   print(dist)
       if dist > thresh:
    #       print(obj[-1][0].decode('utf-8'),' is moving')
           obj_mov=True
    return obj_mov

def object_interaction(obj,wrist_x,wrist_y):
    x=obj[0]
    y=obj[1]
    dist=math.sqrt((x-wrist_x)**2 + (y-wrist_y)**2)
    return dist

def save_object(cat, score, bounds):
    global saved

    saved.append((str(cat.decode("utf-8")),round(score,3),bounds))
    return saved

def draw_bounding_box(image, color, label, conf, x, y, w, h):
    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
    #text = "{}: {:.4f}".format(label, conf)
    #cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
    #    0.35, color, 2)

## ---------------Open Pose-----------------##

def record_coordinate(point, movement_tracker_x, movement_tracker_y):

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

def analyze_pose(humans, image, movement_tracker_x, movement_tracker_y, status_tracker):

    center = None
    arm_angle = None
    spine=None
    spine_angle=None
    angle=None

    sitting_counter = 0
    standing_counter = 0
    laying_counter = 0
    nothing_counter = 0
    cooking_counter = 0

    number_assignments = ["nose", "neck", "r_shoulder", "l_shoulder", "r_hip", "l_hip", "r_knee", "l_knee", "l_ankle", "r_ankle", "r_eye", "l_eye", "r_ear", "l_ear", "r_elbow", "l_elbow", "l_wrist", "r_wrist"]
    detection_tracker = {}
    for part in number_assignments:
        detection_tracker[part] = 0

    poses = []

    main_body_detected = True
    other_parts_detected = True

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

    factor = None

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
        record_coordinate(center, movement_tracker_x, movement_tracker_y)
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

    # print(status_tracker)
    # print(detection_tracker)
    pose = generate_output(arm_angle, spine_angle, wrist_head_dist, rounded_center)
    poses.append(pose)

    return poses, status_tracker
