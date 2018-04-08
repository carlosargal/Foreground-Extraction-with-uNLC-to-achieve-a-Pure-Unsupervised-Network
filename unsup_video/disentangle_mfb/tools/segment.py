import ipdb
import subprocess
import os
import xml

import numpy as np
import scipy.misc as sm

from py_ops import _load_video_ffmpeg
from parse_annotations import parse_annot, read_parsed_annot, resize_parsed_annot

video_path  = "/home/allen/Downloads/UCF-101/"
annot_path  = "/home/allen/Downloads/UCF-101/annotation/UCF101_24Action_Detection_Annotations/UCF101_24_Annotations/"
parsed_path = "/home/allen/Downloads/UCF-101/annotation/UCF101_24Action_Detection_Annotations/UCF101_24_Annotations/"
temp_path   = "/home/allen/projects/disentangle_mfb/output/"
output_path = "/home/allen/Downloads/UCF-101/"
list_path   = "/home/allen/Downloads/UCF-101/trainlist.txt"
err_path    = "/home/allen/projects/disentangle_mfb/"

file_list = open(list_path).readlines()
err_list  = open(err_path + "err", 'w')

resize_h  = 128
resize_w  = 128


def resize_video(video, resize_h, resize_w):
	n = video.shape[0]
	resize_vid = np.zeros([n, resize_h, resize_w, 3], dtype=np.uint8)
	for i in range(n):
		resize_vid[i] = sm.imresize(video[i], (resize_h, resize_w))
	return resize_vid


for idx, file_path in enumerate(file_list):

	if idx > 3:
		break

	file_path  = file_path.split(' ')[0]
	class_name, video_name = file_path.split('/')[0], file_path.split('/')[1]

	annot_file_path  = annot_path + class_name + '/' + video_name.split('.')[0] + '.xgtf'
	parsed_file_path = parsed_path + class_name + '/' + video_name.split('.')[0] + '.txt'
	try:
		parse_annot(annot_file_path, parsed_file_path)
	except xml.etree.ElementTree.ParseError:
		err_list.write(annot_file_path + '\n')
		err_list.flush()
		continue
	bbxes, beg, end = read_parsed_annot(parsed_file_path)

	video, h, w, l = _load_video_ffmpeg(video_path + file_path)
	end = end if end < l else l

	if resize_h != -1:
		video = resize_video(video, resize_h, resize_w)
		bbxes = resize_parsed_annot(bbxes, h, w, resize_h, resize_w)

	for i in range(beg-1, end):

		frame = video[i]
		mask  = np.zeros_like(frame, dtype=np.int)
		bbx   = bbxes[i-beg+1]

		for j in range(bbx.shape[0]):
			h, w, x, y = bbx[j,0], bbx[j,1], bbx[j,2], bbx[j,3]
			mask[(y if y > 0 else 0):y+h, (x if x > 0 else 0):x+w] = 1

		fg_frame = frame * mask
		bg_frame = frame * (1 - mask)

		sm.imsave(temp_path + 'fg_frame%05d.jpg'%(i-beg+1), fg_frame)
		sm.imsave(temp_path + 'bg_frame%05d.jpg'%(i-beg+1), bg_frame)


	os.system('ffmpeg -i ' + temp_path + 'fg_frame%05d.jpg ' \
		             + output_path + file_path + '_fg.gif')
	os.system('ffmpeg -i ' + temp_path + 'bg_frame%05d.jpg ' \
		             + output_path + file_path + '_bg.gif')
	os.system('rm ' + temp_path + '*.jpg')

	print('[%04d/%04d] %s\n\n\n\n'%(idx+1, len(file_list), file_path))