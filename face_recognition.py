import pyrealsense2 as rs
import numpy as np
import cv2
import time
import pandas as pd
from PIL import Image
import os
import csv
import tkinter as tk
import shutil







def capture(name):
    # money=(txt.get())
    
    Id = len(os.listdir('dataSet'))//60
    if name.isalpha():
        detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        cam = cv2.VideoCapture(0)

        Id= str(len(os.listdir('dataSet'))//60)
        sampleNum=0

        while(True):

            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            
            if not color_frame:
                continue

            img = color_frame.get_data()
            img = np.asanyarray(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                #incrementing sample number
                sampleNum=sampleNum+1
                #saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("dataSet\ "+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                #display the frame
                cv2.imshow('frame',img)
            #wait for 100 miliseconds
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum>60:
                break
        cam.release()
        cv2.destroyAllWindows()
        row = [Id, name]
        with open('Details\StudentDetails.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        res = "Image Captured. Proceed To Train."#+",".join(str(f) for f in Id)
        print(res)
    else:
        print("Please Enter Name and Account Balance")




if __name__ == "__main__":

    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    #depth stream not needed
    #config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)

    

    capture("Shivam")