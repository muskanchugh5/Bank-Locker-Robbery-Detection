import os
import tensorflow as tf
from keras.models import load_model
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras import backend as K
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import time
from PIL import Image

IMG_SHAPE = (150,150,3)

def getLabel (X):
    """
    Return the label for an element based on the filename: 
        mask -> 1 
        non_mask -> 0
    """
    if ('M' in X):
        return 1
    else:
        return 0
    
def getLabels (X):
    """
    Returns an array that contains the label for each X
    """
    return np.array([getLabel(X[i]) for i in range(len(X))])

def getLabelFromScore (score):
    """
    Returns the label based on the probability
    if score >= 0.5, return 'mask'
    else return 'non_mask'
    """
    if (score >=0.95):
        return 'M'
    else:
        return 'N'
def loadAndResizeImage (img, w, h):
    '''
    loads the image in 'img' path and returns a PIL image of size (w,h)
    '''
    return image.load_img (img, target_size=(w,h))

def normalizedArrayFromImageInPath (image_path, img_shape):
    """
    returns an the image in 'image' path normalized in an np array
    """
    img = loadAndResizeImage (image_path, img_shape[0], img_shape[1])
    return image.img_to_array(img) / 255.

def loadResizeNormalizeImages (basepath, path_array, img_shape):
    """
    Loads the images from the path 
    and returns them in an array
    """
    images = np.empty ((len(path_array), img_shape[0], img_shape[1], img_shape[2]), dtype=np.float32)
    for i in range (len(path_array)):
        images[i] = normalizedArrayFromImageInPath (os.path.join(basepath,path_array[i]), img_shape)
    return images


model = load_model('model/model-10ep.h5') #This is the model that I have trained for masked and non-masked(trained this model on dataset as described in the report)
print ('Model loaded!')

def predictFromImg (img):
    x = np.expand_dims(img, axis=0)
    return model.predict(x)[0][0]

def predictFromPath (img_path, img_size=(150,150)):
    img = normalizedArrayFromImageInPath (img_path, img_size)
    return predictFromImg (img)

def check_mask(img_path):
# img_path = 'Dataset/t.jpg' 
    start = time.time()
    score = predictFromPath (img_path)
    end = time.time()
    print("Prediction took {:.3f} seconds".format (end - start))
    print("It's a {}!".format (getLabelFromScore (score)))
    # plt.imshow(display_img)
    return getLabelFromScore (score)


