import os
import torch
import numpy as np
import sys
import random
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model1 import Detector
import argparse
from tqdm import tqdm
from retinaface.pre_trained_models import get_model
from preprocess import extract_frames
from datasets import *
from sklearn.metrics import roc_auc_score
import pywt
import warnings
import cv2
from skimage import feature
from skimage import filters
warnings.filterwarnings('ignore')

def main(args, model_path):
    model=None
    model=Detector()
    model=model.to(device)
    cnn_sd=torch.load(model_path, map_location=torch.device('cpu'))["model"]
    model.load_state_dict(cnn_sd)
    model.eval()

    face_detector = get_model("resnet50_2020-07-20", max_size=2048,device=device)
    face_detector.eval()

    if args.dataset == 'FF':
        video_list,target_list=init_ff_t(args.type)
    elif args.dataset == 'CDF':
        video_list,target_list=init_cdf()
    else:
        NotImplementedError

    output_list=[]
    omit_indices = []
    for filename in tqdm(video_list):
        
        face_list,idx_list=extract_frames(filename,args.n_frames,face_detector)

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
            print(img.shape)
            print(img.shape)

            pred=model(img).softmax(1)[:,1]

        pred_list=[]
        idx_img=-1
        for i in range(len(pred)):
            if idx_list[i]!=idx_img:
                pred_list.append([])
                idx_img=idx_list[i]
            pred_list[-1].append(pred[i].item())
        pred_res=np.zeros(len(pred_list))
        for i in range(len(pred_res)):
            pred_res[i]=max(pred_list[i])
        pred=pred_res.mean()

        output_list.append(pred)

    auc=roc_auc_score(target_list,output_list)
    print(f'{args.dataset}| AUC: {auc:.4f}')


if __name__=='__main__':
    seed=1
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

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
        main(args, os.path.join("./weights", w))
        print("-------------------------------------------------------------------")