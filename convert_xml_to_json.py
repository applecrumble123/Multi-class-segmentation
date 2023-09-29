from bs4 import BeautifulSoup
from xml.etree import ElementTree as ET

import cv2
import numpy as np

import json

annotations = ET.parse('/home/johnathon/Desktop/annotations.xml')

root = annotations.getroot()

list_of_annotated_images = [0, 10, 100, 1000, 1020, 1040, 1050, 1060, 110, 1100, 
 1130, 1150, 1160, 1170, 1180, 120, 1200, 1220, 1230, 
 1240, 1250, 1260, 1270, 1280, 1290, 130, 1300, 1310, 
 1320, 1330, 1350, 1360, 1370, 1390, 140, 1400, 1420, 
 1430, 1470, 1490, 150, 1500, 1550, 1560, 160, 1600, 
 1620, 1630, 1640, 1660]

def class_x_y_coord(item):
    coord_list = []
    for x_y in item.attrib['points'].split(';'):
        print(x_y)
        x_y = [float(x_y) for x_y in x_y.split(',')]
        coord_list.append(x_y)
    return coord_list

for child in root.iter('image'):
    #print(child.tag, child.attrib)
    #print(child.attrib['id'])
    img_annotations = {"imgHeight": 960, "imgWidth": 1280}
    objects = []
    for item in child:
        if item.tag != 'polygon' and item.tag != 'polyline':
            continue
        else: 
            coord_dict = {}
            #print(item.tag, item.attrib)
            if item.attrib['label'] == 'sky':
                sky_list = class_x_y_coord(item)
                # if no sky coordinates, skip
                if len(sky_list) == 0:
                    pass
                else: 
                    #total_sky_coord.append(sky_list)
                    #image_coord['sky'] = sky_list
                    #print(sky_list)
                    #print(item.attrib["label"])
                    coord_dict["label"] = item.attrib["label"]
                    coord_dict["polygon"] = sky_list
                    objects.append(coord_dict)
                    #print(coord_dict)

            elif item.attrib['label'] == 'land':
                land_list = class_x_y_coord(item)
                #print(land_dict)
                # if no land coordinates, skip
                if len(land_list) == 0:
                    pass
                else: 
                    coord_dict["label"] = item.attrib["label"]
                    coord_dict["polygon"] = land_list
                    objects.append(coord_dict)

            elif item.attrib['label'] == 'sea':
                sea_list = class_x_y_coord(item)
                #print(sea_dict)
                # if no sea coordinates, skip
                if len(sea_list) == 0:
                    pass
                else: 
                    coord_dict["label"] = item.attrib["label"]
                    coord_dict["polygon"] = sea_list
                    objects.append(coord_dict)
            
            elif item.attrib['label'] == 'ship':
                ship_list = class_x_y_coord(item)
                #print(ship_dict)
                # if no ship coordinates, skip
                if len(ship_list) == 0:
                    pass
                else: 
                    coord_dict["label"] = item.attrib["label"]
                    coord_dict["polygon"] = ship_list
                    objects.append(coord_dict)

            elif item.attrib['label'] == 'buoy':
                buoy_list = class_x_y_coord(item)
                #print(buoy_dict)
                # if no buoy coordinates, skip
                if len(buoy_list) == 0:
                    pass
                else: 
                    coord_dict["label"] = item.attrib["label"]
                    coord_dict["polygon"] = buoy_list
                    objects.append(coord_dict)

            elif item.attrib['label'] == 'other':
                other_list = class_x_y_coord(item)
                #print(other_dict)
                # if no other coordinates, skip
                if len(other_list) == 0:
                    pass
                else: 
                    coord_dict["label"] = item.attrib["label"]
                    coord_dict["polygon"] = other_list
                    objects.append(coord_dict)
            
    #print("############",objects)
    if len(objects) == 0:
        pass
    else:
        img_annotations['objects'] = objects
        #print("############",img_annotations)
        jsonString = json.dumps(img_annotations)
        jsonFile = open("/home/johnathon/Desktop/anotated_frames/masked_annotations/{}.json".format(list_of_annotated_images[int(child.attrib['id'])]), "w")
        jsonFile.write(jsonString)
        jsonFile.close()
                    
    #break
    