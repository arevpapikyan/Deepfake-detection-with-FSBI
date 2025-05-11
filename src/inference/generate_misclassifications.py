import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets,transforms,models,utils
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import sys
import random
import shutil
from model import *
import argparse
from datetime import datetime
from tqdm import tqdm
from retinaface.pre_trained_models import get_model
from preprocess import extract_frames
from datasets import *
from sklearn.metrics import confusion_matrix, roc_auc_score
import pywt
import warnings
import cv2
from skimage import feature
from skimage import filters
warnings.filterwarnings('ignore')

def main(args, model_path, w):
    model=None

    model1=Detector5()
    model2=Detector4()

    model1=model1.to(device)
    model2=model2.to(device)

    cnn_sd1=torch.load("weights/4_0.9991_val.tar", map_location=torch.device('cpu'))["model"]
    cnn_sd2=torch.load("weights/FFc23.tar", map_location=torch.device('cpu'))["model"]

    model1.load_state_dict(cnn_sd1)
    model2.load_state_dict(cnn_sd2)

    model1.eval()
    model2.eval()

    face_detector = get_model("resnet50_2020-07-20", max_size=2048,device=device)
    face_detector.eval()

    if args.dataset == 'FFIW':
        video_list,target_list=init_ffiw()
    elif args.dataset == 'FF':
        video_list,target_list=init_ff_t(args.type)
    elif args.dataset == 'DFD':
        video_list,target_list=init_dfd()
    elif args.dataset == 'DFDC':
        video_list,target_list=init_dfdc()
    elif args.dataset == 'DFDCP':
        video_list,target_list=init_dfdcp()
    elif args.dataset == 'CDF':
        video_list,target_list=init_cdf()
    else:
        NotImplementedError

    filenames = []
    for filename in tqdm(video_list):
        correct = target_list[video_list.index(filename)]
        if correct == 0: continue
        filenamee = filename.split("/")[-1].split("_")[0]
        if filenamee in filenames: continue

        face_list,_=extract_frames(filename,args.n_frames,face_detector)

        with torch.no_grad():
            for f in range(len(face_list)):
                face = face_list[f].astype('float32')/255
                facee = np.transpose(face.copy(), (1,2,0))

                # --------------------- RGB ---------------------
                # b, g, r = cv2.split(facee)

                # cA_r, (cH_r, cV_r, cD_r) = pywt.dwt2(r, 'sym2', mode='reflect')
                # cA_g, (cH_g, cV_g, cD_g) = pywt.dwt2(g, 'sym2', mode='reflect')
                # cA_b, (cH_b, cV_b, cD_b) = pywt.dwt2(b, 'sym2', mode='reflect')

                # cA_r = cv2.resize(cA_r, (380,380), interpolation=cv2.INTER_LINEAR).astype('float32')
                # cA_g = cv2.resize(cA_g, (380,380), interpolation=cv2.INTER_LINEAR).astype('float32')
                # cA_b = cv2.resize(cA_b, (380,380), interpolation=cv2.INTER_LINEAR).astype('float32')

                # cA_r = (cA_r + r)/2
                # cA_g = (cA_g + g)/2
                # cA_b = (cA_b + b)/2

                # img_dwt = np.array([cA_r, cA_g, cA_b])

                # --------------------- HSV ---------------------

                # img_hsv = cv2.cvtColor(facee, cv2.COLOR_BGR2HSV)
                # h, s, v = cv2.split(img_hsv)

                # cA_h, (cH_h, cV_h, cD_h) = pywt.dwt2(h, 'sym2', mode='reflect')
                # cA_s, (cH_s, cV_s, cD_s) = pywt.dwt2(s, 'sym2', mode='reflect')
                # cA_v, (cH_v, cV_v, cD_v) = pywt.dwt2(v, 'sym2', mode='reflect')

                # cA_h = cv2.resize(cA_h, (380,380), interpolation=cv2.INTER_LINEAR).astype('float32')
                # cA_s = cv2.resize(cA_s, (380,380), interpolation=cv2.INTER_LINEAR).astype('float32')
                # cA_v = cv2.resize(cA_v, (380,380), interpolation=cv2.INTER_LINEAR).astype('float32')

                # cA_h = (cA_h + h)/2
                # cA_s = (cA_s + s)/2
                # cA_v = (cA_v + v)/2

                # img_dwt = np.array([cA_h, cA_s, cA_v])

                # --------------------- YCbCr ---------------------

                img_ycbcr = cv2.cvtColor(facee, cv2.COLOR_BGR2YCrCb)
                y, cr, cb = cv2.split(img_ycbcr)

                cA_y, (cH_y, cV_y, cD_y) = pywt.dwt2(y, 'sym2', mode='reflect')
                cA_cb, (cH_cb, cV_cb, cD_cb) = pywt.dwt2(cr, 'sym2', mode='reflect')
                cA_cr, (cH_cr, cV_cr, cD_cr) = pywt.dwt2(cb, 'sym2', mode='reflect')

                cA_y = cv2.resize(cA_y, (380,380), interpolation=cv2.INTER_LINEAR).astype('float32')
                cA_cb = cv2.resize(cA_cb, (380,380), interpolation=cv2.INTER_LINEAR).astype('float32')
                cA_cr = cv2.resize(cA_cr, (380,380), interpolation=cv2.INTER_LINEAR).astype('float32')

                cA_y = (cA_y + y)/2
                cA_cb = (cA_cb + cr)/2
                cA_cr = (cA_cr + cb)/2

                img_dwt = np.array([cA_y, cA_cb, cA_cr])

                face_list[f] = img_dwt
            
            img = torch.tensor(face_list).to(device)

            pred1=model1(img).softmax(1)[:,1]
            pred2=model2(img).softmax(1)[:,1]

            correct = target_list[video_list.index(filename)]

            if correct == 1:
                if pred1.mean() < 0.5 and pred2.mean() < 0.5:
                    for i in range(len(face_list)):
                        if pred1[i] > 0.5 or pred2[i] > 0.5: continue

                        face = face_list[i]
                        face = np.transpose(face, (1,2,0))
                        # face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(f"./{filename.split('/')[-1]}_{i}.png", face*255)

                    print("saved----------------")
                    filenames.append(filenamee)
            

 
            

if __name__=='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser=argparse.ArgumentParser()
    parser.add_argument('-w',dest='weight_name',type=str)
    parser.add_argument('-d',dest='dataset',type=str)
    parser.add_argument('-n',dest='n_frames',default=32,type=int)
    parser.add_argument('-t',dest='type',default="Face2Face",type=str)
    args=parser.parse_args()

    weights = os.listdir("./weights")
    
    for w in weights:
        if w[0] == "d":continue
        print(w)
        main(args, os.path.join("./weights",w),w)
        print("-------------------------------------------------------------------")
