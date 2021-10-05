#$  SPARK FOUNDATION --> TASK-1 #$
#$ PROJECT ---> OBJECT DETECTION / OPTICAL CHARACTER RECOGNITION #$
#$ SUBMITTED BY ---> ANMOL ARYA

#Importing Required Libraries
import cv2 
import matplotlib.pyplot as plt
config_file = 'Object Detection using openCv/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
frozen_model = 'Object Detection using openCv/frozen_inference_graph.pb'
task = cv2.dnn_DetectionModel(frozen_model,config_file)
classLabels = []
file_name = 'Object Detection using openCv/Labels.txt'
with open(file_name, 'rt') as f:
    classLabels = f.read().rstrip('\n').split('\n')
print(classLabels)
print(len(classLabels))
task.setInputSize(320,320)
task.setInputScale(1.0/127.5)
task.setInputMean((127.5,127.5,127.5))
task.setInputSwapRB(True)
img = cv2.imread("Object Detection using openCv/human.jpg")
plt.imshow(img)
ClassIndex, confidece, bbox = task.detect(img,confThreshold=0.5)
print(ClassIndex)
font_scale = 7
font = cv2.FONT_HERSHEY_PLAIN
for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
    cv2.rectangle(img,boxes,(255,0,0),2)
    cv2.putText(img,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40), font, fontScale = font_scale,color=(0,255,0), thickness = 15)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

#FOR VIDEO
Vid = cv2.VideoCapture(r'Object Detection using openCv/video.mp4')
#cap = cv.VideoCapture(0) ==> webcam
if not Vid.isOpened():
    raise IOError("Cannot Open Video")
font_scale = 1.1
font = cv2.FONT_HERSHEY_COMPLEX
while True:
    ret,frame = Vid.read()
    ClassIndex, confidence, bbox = task.detect(frame,confThreshold=0.62)
    if len(ClassIndex)!= 0:
        for ClassInd,conf,boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
            if ClassInd <= 80:
                cv2.rectangle(frame,boxes,(255,0,0),2)
                cv2.putText(frame,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40),font, fontScale=font_scale, color = (0,255,0),thickness = 3)
    cv2.imshow("output",frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break   
Vid.release()
cv2.destroyAllWindows()


