import ipdb
import os

import numpy as np
import scipy.misc as sm

from py_ops import _load_video_ffmpeg
from parse_annotations import read_parsed_annot

video_path  = "/home/allen/Downloads/UCF-101/"
output_path = "/home/allen/Downloads/UCF-101/"
temp_path   = "/home/allen/projects/disentangle_mfb/output/"
ilist_path  = "/home/allen/Downloads/UCF-101/trainlist.txt"
olist_path  = "/home/allen/Downloads/UCF-101/traincliplist.txt"
parsed_path = "/home/allen/Downloads/UCF-101/annotation/UCF101_24Action_Detection_Annotations/UCF101_24_Annotations/"

file_ilist = open(ilist_path).readlines()
file_olist = open(olist_path, 'w')

clip_length = 32
min_length  = 16

for k, line in enumerate(file_ilist):

	file_path = line.split(' ')[0]
	class_id  = line.split(' ')[1]
	class_name, video_name = file_path.split('/')[0], file_path.split('/')[1]
	parsed_file_path = parsed_path + class_name + '/' + video_name.split('.')[0] + '.txt'

	video, h, w, l  = _load_video_ffmpeg(video_path + file_path)
	bbxes, beg, end = read_parsed_annot(parsed_file_path)
	end = end if end < l else l
	l = end - beg + 1
	
	num_extra   = l % clip_length
	num_clips   = int(l / clip_length) #+ (1 if num_extra > min_length else 0)

	idx = beg - 1
	ranges = []
	for _ in range(num_clips):
		ranges.append([idx, idx+clip_length-1])
		idx += clip_length
	'''	for _ in range(num_clips-1):
			ranges.append([idx, idx+clip_length-1])
			idx += clip_length
		ranges.append([idx, end-1])'''

	for rang in ranges:
		cnt = 0
		for i in range(rang[0], rang[1]+1):
			sm.imsave(temp_path + 'frame%05d.jpg'%cnt, video[i])
			cnt += 1

		clip_name = video_name.split('.')[0] + '_' + str(rang[0]+1) \
		          + '_' + str(rang[1]+1) + '.avi'
		clip_path = os.path.join(output_path, class_name, 'clip')
		if not os.path.exists(clip_path):
			os.makedirs(clip_path)
		clip_path = os.path.join(clip_path, clip_name)
		os.system('ffmpeg -i ' + temp_path + 'frame%05d.jpg ' + clip_path)
		os.system('rm ' + temp_path + '*.jpg')

		file_olist.write(clip_path + ' ' + class_id + '\n')
		file_olist.flush()

	print('[%04d/%04d] %s\n\n\n\n'%(k+1, len(file_ilist), file_path))