import cv2
import numpy as np

def preprosesssing(image, imageWidth, imageHeight) :
    return cv2.resize(image, (imageWidth, imageHeight))

def l(path, augment = False, preprocess = None) :
    x = None
    y = np.empty(shape=(0,))
    subDirs = [ name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name)) ]
    
    for classNumber, subDir in enumerate(subDirs):
        tempX = []
        tempY = []
        classDirPath = os.path.join(path, subDir)
        imageDir = [ name for name in os.listdir(classDirPath) if os.path.isfile(os.path.join(classDirPath, name)) ]
        for index, imageName in  enumerate(imageDir) :
            image = cv2.imread(os.path.join(classDirPath, imageName))
            if (preprocess) :
                image = preprocess[0](image, preprocess[1], preprocess[2])
            
            tempX.insert(index,image)
            tempY.insert(index,subDir)
#             break;
        if x is None:
            x = np.array(tempX)
        else :
            x = np.append(x, np.array(tempX), axis = 0)
        y = np.append(y, np.array(tempY), axis = 0)
    return x, y