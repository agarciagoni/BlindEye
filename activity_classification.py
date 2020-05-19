# -*- coding: utf-8 -*-
"""
Created on Mon May  4 20:01:58 2020

@author: Alejandro
"""
import pandas as pd
#classficiation approach
general_objects=['person','cell phone','book','clock','chair','vase','Pottedplant']
road_objects=['bicycle','car','motorbike','aeroplane','bus','train',
              'truck','boat','traffic light','fire hydrant','stop sign',
              'parking meter','bench']
animals=['bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe']
dress_objects=['backpack','umbrella','handbag','tie','suitcase']
sport_objects=['frisbee','skis','snowboard','sports ball','kite','baseball bat',
               'baseball glove','skateboard','surfboard','tennis racket']
utensils_objects=['Wine glass','Cup','Fork','Knive','Spoon','Bowl','Scissors']
food_objects=['Wine glass','Cup','Fork','Knive','Spoon','Bowl','Scissors',
              'Banana','Apple','Sandwich','Orange','Broccoli','Carrot',
              'Hot dog','Pizza','Donut','Cake']
livingroom_objects=['Remote','Sofa','Tvmonitor']
bedroom_objects=['Bed']
kitchen_objects=['Diningtable','Microwave','Oven','Toaster','Sink','Refrigerator']
toilet_objects=['Toilet']
work_objects=['Laptop','Mouse','Keyboard','Book']
toilet_utensils=['Hair drier','Toothbrush']

groups=[general_objects,road_objects,animals,dress_objects,sport_objects,utensils_objects,livingroom_objects,kitchen_objects,toilet_objects,work_objects]
group_names=['general','road','animals','clothes','sports','utensils','livingroom','kitchen','toilet','working']
objects=zip(group_names,groups)
objects_dict=dict(objects)
     
room_objects=[road_objects,livingroom_objects,kitchen_objects,toilet_objects,bedroom_objects]
room_names=['street','livingroom','kitchen','toilet','bedroom']
object_locat=zip(room_names,room_objects)
object_location=dict(object_locat)

activity_objects=[sport_objects,food_objects,toilet_utensils,work_objects]  
activity_names=['Playing sports','Preparing or eating food','Toileting','Working']
activity_info=zip(activity_names,activity_objects)  
activity_info=dict(activity_info)  
  

          
##maybe save all the rooms or activites an then get the most comon one?
def locate_activity(detected):
    rooms={}
    for key_room in object_location:
         furniture=[obj for obj in detected if obj in object_location[key_room]]
         if furniture: rooms[key_room]=furniture
    located_room=max(rooms, key=lambda k: len(rooms[k])) #we define the main room as the one with more objects
    return rooms,located_room           

def describe_activity(detected):
     activities={}
     for key_act in activity_info:
            objects=[obj for obj in detected if obj in activity_info[key_act]]
            if objects: activities[key_act]=objects
     main_activity=max(activities, key=lambda k:len(activities[k])) #we define the main activity as the one with more objects
     return activities, main_activity


detected=['Cup','Sofa','Toilet','Tvmonitor','Laptop','Cup','Orange','Person']
rooms,main_room=locate_activity(detected)
activities,main_activity=describe_activity(detected)        
print('Found in the ',main_room+' with : '+str(rooms[main_room]))
print(main_activity + ' using '+ str(activities[main_activity])) 
for key_act in activities:
    if key_act not in main_activity:
        print('Or maybe '+key_act+' using: '+str(activities[key_act]))
for obj in detected:
    other_objects=[obj for obj in detected if obj in general_objects]
print('Also detected: ' +str(other_objects))

#old version

#for obj in detected: find_object(obj)
#
#describe_activity(detected)

#def find_object(obj):
#    for key_room in object_location: 
#        if obj in object_location[key_room]: print('found in the ',key_room)
#    for key_act in activity_info:
#        if obj in activity_info[key_act]: print(str(key_act) + ' using: '+str(obj))
#        
#def describe_activity_old(detected):
#    for key_room in object_location: 
#        for furniture in detected:
#            if furniture in object_location[key_room]: print('found in the '+str(key_room)+' with a '+furniture)
#    for key_act in activity_info:
#        for obj in detected:
#            if obj in activity_info[key_act]: print(str(key_act) + ' using: '+str(obj))
