import os
import numpy as np
import cv2
from utils import draw_bounding_box

def load_YOLO(labels, weights, cfg):
    # load the COCO class labels
    labelsPath = os.path.join(labels)
    LABELS = open(labelsPath).read().strip().split("\n")
    
    COLORS = np.random.randint(0,255,size=(len(LABELS),3),dtype='uint8')

    weightsPath = os.path.join(weights)
    configPath = os.path.join(cfg)

    # load YOLO object detector
    # determine only the *output* layer names that we need from YOLO
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    print("loaded YOLO")

    return LABELS, COLORS, ln, net

def run_YOLO(image, ln, conf, threshold, net):
    (H,W) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
        swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    boxes = []
    confidences = []
    classIDs = []

    for output in layerOutputs:
        for detection in output:

            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > conf:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")

                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf,
        threshold)
    
    return idxs, boxes, classIDs, confidences

def annotate_image(image, idxs, boxes, COLORS, LABELS, confidences, classIDs):
    bounding_boxes = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            bounding_boxes.append(boxes[i])

            color = [int(c) for c in COLORS[classIDs[i]]]
            label = LABELS[classIDs[i]]

            draw_bounding_box(image, color, label, confidences[i], x, y, w, h)