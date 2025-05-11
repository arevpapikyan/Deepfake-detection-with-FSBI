from glob import glob
import os
import json
import pandas as pd


def init_ff(dataset='all',phase='test'):
	assert dataset in ['all','Deepfakes','Face2Face','FaceSwap','NeuralTextures']
	original_path='data/FaceForensics++/original_sequences/youtube/raw/videos/'
	folder_list = sorted(glob(original_path+'*'))

	list_dict = json.load(open(f'data/FaceForensics++/{phase}.json','r'))
	filelist=[]
	for i in list_dict:
		filelist+=i
	image_list = [i for i in folder_list if os.path.basename(i)[:3] in filelist]
	label_list=[0]*len(image_list)


	if dataset=='all':
		fakes=['Deepfakes','Face2Face','FaceSwap','NeuralTextures']
	else:
		fakes=[dataset]

	folder_list=[]
	for fake in fakes:
		fake_path=f'data/FaceForensics++/manipulated_sequences/{fake}/raw/videos/'
		folder_list_all=sorted(glob(fake_path+'*'))
		folder_list+=[i for i in folder_list_all if os.path.basename(i)[:3] in filelist]
	label_list+=[1]*len(folder_list)
	image_list+=folder_list
	return image_list,label_list

def init_ff_t(dataset='all',phase='test'):
	folder_list=[]
	label_list=[]

	print("Testing on", dataset)
	dataset_path = "data/FaceForensics++"
	fake_list = os.listdir(f"data/FaceForensics++/manipulated_sequences/{dataset}/c23/videos")
	real_list = os.listdir(f"data/FaceForensics++/original_sequences/youtube/c23/videos")
	
	for i in fake_list:
		folder_list.append(os.path.join(f"data/FaceForensics++/manipulated_sequences/{dataset}/c23/videos", i))
		label_list.append(1)
	
	for i in real_list:
		folder_list.append(os.path.join("data/FaceForensics++/original_sequences/youtube/c23/videos", i))
		label_list.append(0)

	return folder_list,label_list


def init_dfd():
	real_path='data/FaceForensics++/original_sequences/actors/raw/videos/*.mp4'
	real_videos=sorted(glob(real_path))
	fake_path='data/FaceForensics++/manipulated_sequences/DeepFakeDetection/raw/videos/*.mp4'
	fake_videos=sorted(glob(fake_path))

	label_list=[0]*len(real_videos)+[1]*len(fake_videos)

	image_list=real_videos+fake_videos

	return image_list,label_list


def init_dfdc():
		
	label=pd.read_csv('data/DFDC/labels.csv',delimiter=',')
	folder_list=[f'data/DFDC/videos/{i}' for i in label['filename'].tolist()]
	label_list=label['label'].tolist()
	
	return folder_list,label_list

def init_cdf():

	image_list=[]
	label_list=[]

	video_list_txt='data/Celeb-DF-v2/List_of_testing_videos.txt'
	with open(video_list_txt) as f:
		folder_list=[]
		for data in f:
			# print(data)
			line=data.split()
			if "real" in line: continue
			# print(line)
			path=line[1].split('/')
			folder_list+=['data/Celeb-DF-v2/'+path[0]+'/videos/'+path[1]]
			label_list+=[1-int(line[0])]
		return folder_list,label_list
		


