import os
import shutil
import cv2
import numpy as np
import natsort

def make_folders(ad):
    
    NewFolder2=os.path.join(ad,"SR_edge")
    try:
        os.makedirs(NewFolder2)
    except:
        shutil.rmtree(NewFolder2)
        os.makedirs(NewFolder2)
        
    NewFolder3=os.path.join(ad,"SR_result")
    try:
        os.makedirs(NewFolder3)
    except:
        shutil.rmtree(NewFolder3)
        os.makedirs(NewFolder3)

def find_reds(pic):
    red_xs=np.where(pic[:,:,0]!=pic[:,:,1])[1]
    red_ys=np.where(pic[:,:,0]!=pic[:,:,1])[0]
    return(red_xs,red_ys)

def rotate_image(image, angle):
    height, width = image.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated_image

def load_files(ad):
    FileName=sorted(os.listdir(ad))
    FileNames=[]
    for i in range(len(FileName)):
        try:
            if FileName[i].split(".")[1]=="tif":
                FileNames=FileNames+[FileName[i]]
        except:
            pass
    FileNames=natsort.natsorted(FileNames)
    return(FileNames)

def slope_measurement(ad):
    
    pic_slope1=cv2.imread(os.path.join(ad,"slope","1.bmp"))
    pic_slope2=cv2.imread(os.path.join(ad,"slope","2.bmp"))
    red1_xs,red1_ys=find_reds(pic_slope1)
    red2_xs,red2_ys=find_reds(pic_slope2)
    dx=red2_xs-red1_xs
    dy=red2_ys-red1_ys
    gradian=np.arctan((dy)/(dx))
    angle=gradian*180/np.pi
    rotated1=rotate_image(pic_slope1, angle[0])
    return(angle[0],rotated1, red1_xs[0], red1_ys[0], red2_xs[0], red2_ys[0])



def angle_polynomial_order(num_px_ratio,angle_degree):
    if angle_degree<=60 :
        return 2, int(60*num_px_ratio) 
    elif 60<angle_degree<=105:
        return 2, int(85*num_px_ratio)
    elif 105<angle_degree<=135:
        return 3, int(125*num_px_ratio) #175
    elif 135<angle_degree:
        return 4, int(145*num_px_ratio) #215