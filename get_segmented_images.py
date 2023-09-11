from bs4 import BeautifulSoup
from xml.etree import ElementTree as ET

import cv2
import numpy as np

import os

annotated_images_root_folder = '/home/johnathon/Desktop/anotated_frames'

masked_images_root_folder = "/home/johnathon/Desktop/anotated_frames/masked_images"

annotations = ET.parse('/home/johnathon/Desktop/annotations.xml')
root = annotations.getroot()
#print(root.tag)

list_of_annotated_images = [0, 10, 100, 1000, 1020, 1040, 1050, 1060, 110, 1100, 
 1130, 1150, 1160, 1170, 1180, 120, 1200, 1220, 1230, 
 1240, 1250, 1260, 1270, 1280, 1290, 130, 1300, 1310, 
 1320, 1330, 1350, 1360, 1370, 1390, 140, 1400, 1420, 
 1430, 1470, 1490, 150, 1500, 1550, 1560, 160, 1600, 
 1620, 1630, 1640, 1660]

# sky, land, sea, ship, buoy, other
CLASS_LIST = ['sky', 'land', 'sea', 'ship', 'buoy', 'other']
PALETTE = [[128, 0, 0], [0, 128, 0], [128, 128, 0], 
           [0, 0, 128], [128, 0, 128], [0, 128, 128]]

def class_x_y_coord(item):
    coord_list = []
    for x_y in item.attrib['points'].split(';'):
        x_y = [float(x_y) for x_y in x_y.split(',')]
        coord_list.append(x_y)
    return coord_list

dict_of_image_coord = {}
for child in root.iter('image'):
    # append all the object coord into the dict
    image_coord = {}
    # collect all the coordinates for each object
    total_sky_coord = []
    total_land_coord = []
    total_sea_coord = []
    total_ship_coord = []
    total_buoy_coord = []
    total_other_coord = []
    #print(child.tag, child.attrib)
    for item in child:
        #print(j.tag, child.attrib['id'])
        # only retrieve plolygon and polyline
        if item.tag != 'polygon' and item.tag != 'polyline':
            continue
        else: 
            #print(item.tag, item.attrib)
            # create a dictionary
            if item.attrib['label'] == 'sky':
                sky_list = class_x_y_coord(item)
                # if no sky coordinates, skip
                if len(sky_list) == 0:
                    pass
                else: 
                    total_sky_coord.append(sky_list)
                    #image_coord['sky'] = sky_list

            elif item.attrib['label'] == 'land':
                land_list = class_x_y_coord(item)
                #print(land_dict)
                # if no land coordinates, skip
                if len(land_list) == 0:
                    pass
                else: 
                    total_land_coord.append(land_list)
                    #image_coord['land'] = land_list

            elif item.attrib['label'] == 'sea':
                sea_list = class_x_y_coord(item)
                #print(sea_dict)
                # if no sea coordinates, skip
                if len(sea_list) == 0:
                    pass
                else: 
                    total_sea_coord.append(sea_list)
                    #image_coord['sea'] = sea_list
            
            elif item.attrib['label'] == 'ship':
                ship_list = class_x_y_coord(item)
                #print(ship_dict)
                # if no ship coordinates, skip
                if len(ship_list) == 0:
                    pass
                else: 
                    total_ship_coord.append(ship_list)
                    #image_coord['ship'] = ship_list

            elif item.attrib['label'] == 'buoy':
                buoy_list = class_x_y_coord(item)
                #print(buoy_dict)
                # if no buoy coordinates, skip
                if len(buoy_list) == 0:
                    pass
                else: 
                    total_buoy_coord.append(buoy_list)
                    #image_coord['buoy'] = buoy_list

            elif item.attrib['label'] == 'other':
                other_list = class_x_y_coord(item)
                #print(other_dict)
                # if no other coordinates, skip
                if len(other_list) == 0:
                    pass
                else: 
                    total_other_coord.append(other_list)
                    #image_coord['other'] = other_list
            
            #print("#############",total_obj_coord)
    image_coord['sky'] = total_sky_coord
    image_coord['land'] = total_land_coord
    image_coord['sea'] = total_sea_coord
    image_coord['ship'] = total_ship_coord
    image_coord['buoy'] = total_buoy_coord
    image_coord['other'] = total_other_coord

    #print("################",image_coord)
    #print("################ sky",image_coord['sky'])
    #print("################ land",image_coord['land'])
    #print("################ sea",image_coord['sea'])
    #print("################",image_coord['ship'])
    #print("################",image_coord['buoy'])
    #print("################ other",image_coord['other'])
    if len(image_coord) ==0:
        continue
    else:
        dict_of_image_coord[child.attrib['id']] = image_coord
    #print(len(image_coord))
    #break
#print(dict_of_image_coord['0']['ship'])
#print(len(dict_of_image_coord))


#"""
for id in range(len(list_of_annotated_images)):
    print(id)
    image_path = os.path.join(annotated_images_root_folder, str(list_of_annotated_images[id]) + ".png")
    print(image_path)
    img = cv2.imread(image_path)
    img_shape = img.shape
    #print(img_shape)

    # create a black image
    black_img = np.zeros(img_shape, dtype = np.float64)

    #cv2.imshow("Black Image", black_img)
    #cv2.waitKey(0)

    #print(dict_of_image_coord[str(id)])

    for key in dict_of_image_coord[str(id)]:
        #print(key)
        #print(dict_of_image_coord[str(id)][key])
        if key in CLASS_LIST:
            class_list_index = CLASS_LIST.index(key)
            #print(class_list_index)

            # get the colour
            colour = PALETTE[class_list_index]
            #print(colour)

            points = dict_of_image_coord[str(id)][key]
            #print(points)
            # if there are no class objects, ignore
            if len(points) == 0:
                pass

            # fill the polygon with colour with respect to the class indice
            if len(points) >= 1:
                for i in points:
                    cv2.fillPoly(black_img, pts=np.int32([np.array(i)]), color=colour)
            """
            elif len(points) == 1:
                cv2.fillPoly(black_img, pts=np.int32([np.array(points)]), color=colour)
            """
            #break
    
    #cv2.imshow("Test", black_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    mask_image_path = os.path.join(masked_images_root_folder, str(list_of_annotated_images[id]) + "_mask.png")
    cv2.imwrite(mask_image_path, black_img)
    

    #break
#"""

