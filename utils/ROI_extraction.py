import numpy as np
import SimpleITK as sitk
import os 
# import csv
# from sklearn.metrics import recall_score,precision_score
# from scipy.spatial.distance import directed_hausdorff
# import argparse


def GetRoi(img:np.ndarray):
    x,y,z = img.shape   

    #  X-axis
    findStart = True
    findEnd = True
    for i in range(x):
        # find
        if findStart:
            slice = np.sum(img[i,:,:])
            if slice!=0:
                xStart = i
                findStart = False

        if findEnd:
            slice = np.sum(img[x-1-i,:,:])
            if slice!=0:
                xEnd = x-1-i
                findEnd = False

        if not findStart and not findEnd:
            # print("Start and end of X are found")
            break

    if xStart is None:
        xStart = 0
        print("Auto set the start of X as 0")
    if xEnd is None:
        xEnd = x
        print("Auto set the end of X as 0")


     #  Y-axis

    findStart = True
    findEnd = True
    for i in range(y):
        # find
        if findStart:
            slice = np.sum(img[:,i,:])
            if slice!=0:
                yStart = i
                findStart = False

        if findEnd:
            slice = np.sum(img[:,y-1-i,:])
            if slice!=0:
                yEnd = y-1-i
                findEnd = False

        if not findEnd and not findStart:
            # print("Start and end of Y are found")
            break

    if yStart is None:
        yStart = 0
        print("Auto set the start of Y as 0")
    if yEnd is None:
        yEnd = y
        print("Auto set the end of Y as 0")    

 #  Z-axis

    findStart = True
    findEnd = True
    for i in range(z):
        # find
        if findStart:
            slice = np.sum(img[:,:,i])
            if slice!=0:
                zStart = i
                findStart = False

        if findEnd:
            slice = np.sum(img[:,:,z-1-i])
            if slice!=0:
                zEnd = z-1-i
                findEnd = False

        if not findEnd and not findStart:
            # print("Start and end of Z are found")
            break

    if zStart is None:
        zStart = 0
        print("Auto set the start of Z as 0")
    if zEnd is None:
        zEnd = z
        print("Auto set the end of Z as 0") 

    
    return (xStart,xEnd),(yStart,yEnd),(zStart,zEnd)



def img_to_roi(image_path,segmentation_path,save_path,scale=0.1):
    image = sitk.ReadImage(image_path)
    predic = sitk.ReadImage(segmentation_path)
    imageArry = sitk.GetArrayFromImage(image)
    predicArry = sitk.GetArrayFromImage(predic)
    
    (xStart,xEnd),(yStart,yEnd),(zStart,zEnd) = GetRoi(predicArry)
    
    x_extendValue = int(scale*(xEnd-xStart))
    xStart = xStart-x_extendValue if xStart-x_extendValue>=0 else 0
    xEnd = xEnd+x_extendValue if xEnd+x_extendValue<= imageArry.shape[0] else imageArry.shape[0]
    
    y_extendValue = int(scale*(yEnd-yStart))
    yStart = yStart-y_extendValue if yStart-y_extendValue>=0 else 0
    yEnd = yEnd+y_extendValue if yEnd+y_extendValue<= imageArry.shape[0] else imageArry.shape[1]
    
    z_extendValue = int(scale*(zEnd-zStart))
    zStart = zStart-z_extendValue if zStart-z_extendValue>=0 else 0
    zEnd = zEnd+z_extendValue if zEnd+z_extendValue<= imageArry.shape[0] else imageArry.shape[2]
    

    roiArry = imageArry[xStart:xEnd,yStart:yEnd,zStart:zEnd]
    if roiArry.size==0:
        print(save_path+" error!!!")
    else:
        print(save_path+" finished.")
    
    roiImage = sitk.GetImageFromArray(roiArry)
    sitk.WriteImage(roiImage, save_path)



def main():

    image_dir = "/path/of/nii/data"
    mask_dir = "/path/of/sgementation"
    roi_dir = "/roi/save/path"

    n = 107
    save_file = np.empty([1,n])
    id_list = np.array([])
    patient_list = os.listdir(image_dir)
    count =1

    for patient in patient_list:
        print('Extracting:',patient)
        imagePath = os.path.join(image_dir,patient)
        maskPath= os.path.join(mask_dir,patient)
        roiPath = os.path.join(roi_dir,patient)
        img_to_roi(image_path=imagePath,segmentation_path=maskPath,roiPath=roiPath)


    

if __name__ == "__main__":
    main()
