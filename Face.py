##fetching wieghts through trained model
import numpy as np
import cv2
import dlib
import datetime

from align import Align
import face_recognition
import PIL


def loadModel():
  W = np.loadtxt(open("model/pcaWeights.txt","rb"),delimiter=",",skiprows=0)
  W1 = np.loadtxt(open("model/ldaWeights.txt","rb"),delimiter=",",skiprows=0)
  FisherMatrix = np.loadtxt(open("model/TrainData.txt","rb"),delimiter=",",skiprows=0)
  mean  = np.loadtxt(open("model/meanData.txt","rb"),delimiter=",",skiprows=0)
  trainEncodings = np.loadtxt(open("model/encodings.txt","rb"),delimiter=",",skiprows=0)
  trainEncodings = trainEncodings.tolist()
  labels=[]
  with open('model/labels.txt', 'r') as filehandle:
    for line in filehandle:
        # remove linebreak which is the last character of the string
        currentPlace = line[:-1]
        # add item to the list
        labels.append(currentPlace)
  label1=[]
  with open('model/labels.txt', 'r') as filehandle:
    for line in filehandle:
        # remove linebreak which is the last character of the string
        currentPlace = line[:-1]
        # add item to the list
        label1.append(currentPlace)

  return W,W1,FisherMatrix,mean,trainEncodings,labels,label1




# http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
predictor_model = "shape_predictor_68_face_landmarks.dat"


def FaceRecognition(imagePath): 
  W,W1,FisherMatrix,mean,trainEncodings,labels,label1 = loadModel()
  threshold = 3300
  img = cv2.imread(imagePath)


  try:
    t = face_recognition.face_locations(img, number_of_times_to_upsample=1)
  except:
    t = face_recognition.face_locations(img, number_of_times_to_upsample=1, model="cnn")

  # loop over detected faces
  if(len(t)==0):
    return True

  #timestamp check
  else:
    now = datetime.datetime.now()
    time9am = now.replace(hour=9, minute=0, second=0, microsecond=0)
    time6pm = now.replace(hour=18, minute=0, second=0, microsecond=0)

  # if now < time9am or now > time6pm :
  #   print("Activate alert alarm") # add the thread related alert system
  #   return False

  try:
    
    im = PIL.Image.open(imagePath)
    # im = im.resize((80,80))
    im = im.convert('RGB')
    img = np.array(im)
    # print("ff")
    align = Align(predictor_model)
    img=align.align(80,img)

    test_encoding = face_recognition.face_encodings(img, known_face_locations=[[0, 80,80, 0]] )[0]
    results =  face_recognition.compare_faces(trainEncodings,test_encoding,0.4)
    for i in range(len(results)):
        if(results[i]==True):
            print(labels[i])
            return True
    return False

  except:
    img = cv2.imread(imagePath)
    for face in t:
      x = t[0][3]
      y = t[0][0]
      w = t[0][1] - x
      h = t[0][2] - y
      
      if(x<0):
          x=0
      if(y<0):
          y=0
      
      img = img[y:y+h, x:x+w]

    # print(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print(img.shape)
    img = cv2.resize(img,(80,80))
    # print(img)
    img = img.flatten()

    mindist=1e19
    deviationFromMean = (img-mean)
    eigTestimg = np.dot(img,W)
    FisherTestImg = np.dot(eigTestimg,W1)
    for i in range(0,FisherMatrix.shape[0]):
        dist = np.linalg.norm(FisherMatrix[i]-FisherTestImg)
      #   print(dist,mindist,dataTrainY[i])
        if(dist<mindist):
          #   label = dataTrainY[i]
            mindist = dist
            ind = i
   
    
    if(mindist>threshold):
      return False
    else:
      print(label1[ind])
      return True 


  
