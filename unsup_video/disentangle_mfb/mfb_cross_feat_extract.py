import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
from os import listdir

import sys
import time
import h5py
import argparse
import tools.ops

import numpy as np
np.set_printoptions(threshold=np.nan)

import tensorflow as tf
import scipy.misc as sm

from models.mfb_net_cross_backup import mfb_net
from tools.utilities import *
from tools.ops import *


parser = argparse.ArgumentParser()
parser.add_argument('-lr', dest='lr', type=float, default='1e-3', help='original learning rate')
parser.add_argument('-version', dest='version', type=int, default='1', help='cross convolution version')
parser.add_argument('-feat', dest='feat', type=int, default='1', help='feature loss')
parser.add_argument('-cross_layer', dest='cross_layer', type=int, default='1', help='cross layer')
parser.add_argument('-diff_dec', dest='diff_dec', type=int, default='0', help='diff_dec')
parser.add_argument('-batch_size', dest='batch_size', type=int, default='1', help='batch_size')
args = parser.parse_args()

flags = tf.app.flags
flags.DEFINE_float('lr', args.lr, 'Original learning rate.')
flags.DEFINE_integer('version', args.version, 'Cross convolution version.')
flags.DEFINE_integer('batch_size', args.batch_size, 'Batch size.')
flags.DEFINE_integer('feat', args.feat, 'feat')
flags.DEFINE_integer('cross_layer', args.cross_layer, 'cross layer.')
flags.DEFINE_integer('diff_dec', args.diff_dec, 'diff_dec.')
flags.DEFINE_integer('num_epochs', 1, 'Number of epochs.')	# ~13 min per epoch
flags.DEFINE_integer('num_gpus', 1, 'Number of GPUs.')
flags.DEFINE_integer('seq_length', 16, 'Length of each video clip.')
flags.DEFINE_integer('height', 128, 'Height of video frame.')
flags.DEFINE_integer('width', 128, 'Width of video frame.')
flags.DEFINE_integer('channel', 3, 'Number of channels for each frame.')
flags.DEFINE_integer('num_sample', 9866, 'Number of samples in this dataset.')
flags.DEFINE_float('wd', 0.001, 'Weight decay rate.')

FLAGS = flags.FLAGS

prefix          = 'mfb_cross'
model_save_dir  = '/home/bsc31/bsc31190/gpfs_scratch/ckpt/' + prefix
loss_save_dir   = './loss'
test_save_dir   = './test/' + prefix
train_list_path = '/gpfs/home/bsc31/bsc31190/gpfs_projects/trainlist.txt'
dataset_path    = '/gpfs/home/bsc31/bsc31190/gpfs_projects/UCF-101-tf-records-publish'

use_pretrained_model = True
save_predictions     = True


def decode_frames(frame_list, h, w, l):
	clip = []
	for i in range(l):
		frame = frame_list[i]
		image = tf.cast(tf.image.decode_jpeg(frame), tf.float32)
		image.set_shape((h, w, 3))
		clip.append(image)

	return tf.stack(clip)


def generate_mask(img_mask_list, h, w, l):
	img_masks, loss_masks = [], []

	for i in range(l):
		# generate image mask
		img_mask = img_mask_list[i]
		img_mask = tf.cast(tf.image.decode_png(img_mask), tf.float32)
		img_mask = tf.reshape(img_mask, (h, w))
		img_masks.append(img_mask)

		# generate loss mask
		s_total   = h * w
		s_mask    = tf.reduce_sum(img_mask)
		def f1(): return img_mask*((s_total-s_mask)/s_mask-1)+1
		def f2(): return tf.zeros_like(img_mask)
		def f3(): return tf.ones_like(img_mask)
		loss_mask = tf.case([(tf.equal(s_mask, 0), f2), \
							 (tf.less(s_mask, s_total/2), f1)],
							 default=f3)
		
		"""
		loss_mask = loss_mask_list[i]
		loss_mask = tf.cast(tf.image.decode_png(loss_mask), tf.float32)
		loss_mask = tf.reshape(loss_mask, (h, w))
		"""
		loss_masks.append(loss_mask)

	return tf.stack(img_masks), tf.stack(loss_masks)
	

def read_my_file_format(filename_queue):
	reader = tf.TFRecordReader()
	_, serialized_example = reader.read(filename_queue)
	context_features = {
    	"height": tf.FixedLenFeature([], dtype=tf.int64),
    	"width": tf.FixedLenFeature([], dtype=tf.int64),
    	"sequence_length": tf.FixedLenFeature([], dtype=tf.int64),
    	"text": tf.FixedLenFeature([], dtype=tf.string),
    	"filename": tf.FixedLenFeature([], dtype=tf.string),
    	"label": tf.FixedLenFeature([], dtype=tf.int64)
	}
	sequence_features = {
	    "frames": tf.FixedLenSequenceFeature([], dtype=tf.string),
	    "masks": tf.FixedLenSequenceFeature([], dtype=tf.string)
	}
	context_parsed, sequence_parsed = tf.parse_single_sequence_example(
	    serialized=serialized_example,
	    context_features=context_features,
	    sequence_features=sequence_features
	)

	# start queue runner so it won't stuck
	tf.train.start_queue_runners(sess=tf.get_default_session())

	height = 128#context_parsed['height'].eval()
	width  = 128#context_parsed['width'].eval()
	sequence_length = 32#context_parsed['sequence_length'].eval()

	clip = decode_frames(sequence_parsed['frames'], height, width, sequence_length)
	img_mask, loss_mask = generate_mask(sequence_parsed['masks'], \
										height, width, sequence_length)

	# randomly sample clips of 16 frames
	idx  = 8#np.random.randint(0, sequence_length-FLAGS.seq_length+1)
	clip = clip[idx:idx+FLAGS.seq_length] / 255.0 * 2 - 1
	img_mask  = img_mask[idx:idx+FLAGS.seq_length]
	loss_mask = loss_mask[idx:idx+FLAGS.seq_length]
	loss_mask.set_shape([FLAGS.seq_length, height, width])

	clip = tf.tile(tf.expand_dims(clip[0],0), [16,1,1,1])
	img_mask = tf.tile(tf.expand_dims(img_mask[0],0), [16,1,1])
	loss_mask = tf.tile(tf.expand_dims(loss_mask[0],0), [16,1,1])

	filename = context_parsed['filename']

	return clip, img_mask, loss_mask, filename


def input_pipeline(filenames, batch_size, read_threads=1, num_epochs=None):
	filename_queue = tf.train.string_input_producer(
      					filenames, num_epochs=FLAGS.num_epochs, shuffle=False)
	# initialize local variables if num_epochs is not None or it'll raise uninitialized problem
	tf.get_default_session().run(tf.local_variables_initializer())

	example_list = read_my_file_format(filename_queue)

	min_after_dequeue = 10
	capacity = min_after_dequeue + 3 * batch_size
	clip_batch, img_mask_batch, loss_mask_batch, filename_batch = tf.train.batch(
	  	example_list, batch_size=batch_size, capacity=capacity)

	return clip_batch, img_mask_batch, loss_mask_batch, filename_batch


def tower_loss(name_scope, mfb, clips, img_masks, loss_masks): # masks 16 16 128 128
	# get reconstruction and ground truth
	first_fg_rec, first_bg_rec, last_fg_rec = mfb.reconstruct()
	#first_fg_mask, last_fg_mask = mfb.masks()
	#first_bg_mask  = 1 - first_fg_mask
	#ffg_masks_list = [first_fg_mask for i in range(FLAGS.channel)]
	#fbg_masks_list = [first_bg_mask for i in range(FLAGS.channel)]
	#lfg_masks_list = [last_fg_mask for i in range(FLAGS.channel)]
	#first_fg_mask = tf.concat(ffg_masks_list, axis=3)
	#first_bg_mask = tf.concat(fbg_masks_list, axis=3)
	#last_fg_mask  = tf.concat(lfg_masks_list, axis=3)

	#first_fg_rec  = first_fg_rec * first_fg_mask
	#first_bg_rec  = first_bg_rec * first_bg_mask
	#first_rec     = first_fg_rec + first_bg_rec
	#last_fg_rec   = last_fg_rec  * last_fg_mask

	img_masks_list  = [img_masks for i in range(FLAGS.channel)]
	loss_masks_list = [loss_masks for i in range(FLAGS.channel)]
	img_masks = tf.stack(img_masks_list, axis=-1)
	loss_masks = tf.stack(loss_masks_list, axis=-1)

	# mask the ground truth
	first_frames = clips[:,0,:,:,:]
	last_frames  = clips[:,-1,:,:,:]
	first_fg_gt  = first_frames * img_masks[:, 0]
	first_bg_gt  = first_frames * (1 - img_masks[:, 0])
	last_fg_gt   = last_frames  * img_masks[:, -1]

	mfb.gt_ffg   = first_fg_gt
	mfb.gt_fbg   = first_bg_gt
	mfb.gt_lfg   = last_fg_gt

	# calculate reconstruction loss
	first_fg_loss = tf.reduce_mean(tf.abs(first_fg_rec-first_fg_gt)*loss_masks[:,0])#*img_masks[:, 0])#*loss_masks[:,0])
	first_bg_loss = tf.reduce_mean(tf.abs(first_bg_rec-first_bg_gt))#*(1-img_masks[:, 0]))
	last_fg_loss  = tf.reduce_mean(tf.abs(last_fg_rec-last_fg_gt)*loss_masks[:,-1])#*img_masks[:, -1])#*loss_masks[:,-1])
	rec_loss      = first_fg_loss + first_bg_loss + last_fg_loss

	# calculate mask loss
	#dim = np.prod(first_fg_mask_logits.get_shape().as_list()[1:])
	#first_fg_mask_loss = tf.reduce_mean(tf.square(img_masks[:,0,:,:,0]-tf.nn.softmax(first_fg_mask_logits)))
	#first_bg_mask_loss = tf.reduce_mean(tf.square(1-img_masks[:,0,:,:,0]-tf.nn.softmax(first_bg_mask_logits)))
	#last_fg_mask_loss  = tf.reduce_mean(tf.square(img_masks[:,-1,:,:,0]-tf.nn.softmax(last_fg_mask_logits)))
	#first_fg_mask_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( \
	#						labels=tf.reshape(img_masks[:, 0, :, :, 0], [-1, dim]), \
	#						logits=tf.reshape(first_fg_mask_logits, [-1, dim])))
	#first_bg_mask_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( \
	#						labels=tf.reshape(1-img_masks[:, 0, :, :, 0], [-1, dim]), \
	#						logits=tf.reshape(first_bg_mask_logits, [-1, dim])))
	#last_fg_mask_loss  = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits( \
	#						labels=tf.reshape(img_masks[:, -1, :, :, 0], [-1, dim]), \
	#						logits=tf.reshape(last_fg_mask_logits, [-1, dim])))
	#mask_loss          = first_fg_mask_loss + first_bg_mask_loss + last_fg_mask_loss

	# feature loss
	feat_loss = tf.reduce_mean(tf.square(mfb.last_fg_feat-mfb.inv_last_fg_feat_gt))

	weight_decay_loss_list = tf.get_collection('losses', name_scope)
	weight_decay_loss = 0.0
	if len(weight_decay_loss_list) > 0:
		weight_decay_loss = tf.add_n(weight_decay_loss_list)

	tf.add_to_collection('losses', rec_loss)
	#tf.add_to_collection('losses', mask_loss)
	tf.add_to_collection('losses', feat_loss)
	losses = tf.get_collection('losses', name_scope)

	# Calculate the total loss for the current tower.
	total_loss = tf.add_n(losses, name='total_loss')

	return total_loss, first_fg_loss, first_bg_loss, last_fg_loss, feat_loss, weight_decay_loss#, \
		   #first_fg_mask_loss, first_bg_mask_loss, last_fg_mask_loss


def average_gradients(tower_grads):
	average_grads = []
	for grad_and_vars in zip(*tower_grads):
		grads = []
		for g, v in grad_and_vars:
			expanded_g = tf.expand_dims(g, 0)
			grads.append(expanded_g)
		grad = tf.concat(grads, 0)
		grad = tf.reduce_mean(grad, 0)
		v = grad_and_vars[0][1]
		grad_and_var = (grad, v)
		average_grads.append(grad_and_var)

	return average_grads	


def run_training():

	# Create model directory
	if not os.path.exists(model_save_dir):
		os.makedirs(model_save_dir)
	model_filename = "./mfb_baseline_ucf24.model"

	tower_ffg_losses, tower_fbg_losses, tower_lfg_losses, tower_feat_losses = [], [], [], []
	tower_ffg_m_losses, tower_fbg_m_losses, tower_lfg_m_losses = [], [], []

	global_step = tf.get_variable(
                'global_step',
                [],
                initializer=tf.constant_initializer(0),
                trainable=False
                )
	starter_learning_rate = 1e-4
	learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           100000000, 0.5, staircase=True)
	opt = tf.train.AdamOptimizer(learning_rate)

	# Create a session for running Ops on the Graph.
	config = tf.ConfigProto(allow_soft_placement=True,
							log_device_placement=False)
	#config.operation_timeout_in_ms = 10000
	sess = tf.Session(config=config)
	coord = tf.train.Coordinator()
	threads = None

	train_list_file = open(train_list_path, 'r')
	train_list = train_list_file.read().splitlines()
	for i, line in enumerate(train_list):
		train_list[i] = os.path.join(dataset_path, train_list[i])

	num_val = len(train_list)
	assert(num_val % FLAGS.num_gpus == 0)
	num_for_each_gpu = num_val // FLAGS.num_gpus

	clips_list, img_masks_list, loss_masks_list, filenames_list = [], [], [], []
	with sess.as_default():
		for i in range(FLAGS.num_gpus):
			clips, img_masks, loss_masks, filenames = input_pipeline(train_list[i*num_for_each_gpu:(i+1)*num_for_each_gpu], \
															FLAGS.batch_size, num_epochs=FLAGS.num_epochs)
			clips_list.append(clips)
			img_masks_list.append(img_masks)
			loss_masks_list.append(loss_masks)
			filenames_list.append(filenames)

	mfb_list = []
	with tf.variable_scope('vars') as var_scope:
		for gpu_index in range(FLAGS.num_gpus):
			with tf.device('/gpu:%d' % (gpu_index)):
				with tf.name_scope('%s_%d' % ('tower', gpu_index)) as scope:
				
					# construct model
					mfb = mfb_net(clips_list[gpu_index], FLAGS.height, FLAGS.width, FLAGS.seq_length, FLAGS.channel, FLAGS.batch_size, is_training=False)
					mfb_list.append(mfb)
					_, first_fg_loss, first_bg_loss, last_fg_loss, feat_loss, _ = tower_loss(scope, mfb, clips_list[gpu_index], img_masks_list[gpu_index], loss_masks_list[gpu_index])

					var_scope.reuse_variables()

					tower_ffg_losses.append(first_fg_loss)
					tower_fbg_losses.append(first_bg_loss)
					tower_lfg_losses.append(last_fg_loss)
					tower_feat_losses.append(feat_loss)


	# concatenate the losses of all towers
	ffg_loss_op  = tf.reduce_mean(tower_ffg_losses)
	fbg_loss_op  = tf.reduce_mean(tower_fbg_losses)
	lfg_loss_op  = tf.reduce_mean(tower_lfg_losses)
	feat_loss_op = tf.reduce_mean(tower_feat_losses)

	# saver for saving checkpoints
	saver = tf.train.Saver()
	init = tf.initialize_all_variables()

	sess.run(init)
	if not os.path.exists(model_save_dir):
		os.makedirs(model_save_dir)
	if use_pretrained_model:
		print('[*] Loading checkpoint ...')
		#model = tf.train.latest_checkpoint(model_save_dir)
		model = '/home/bsc31/bsc31190/gpfs_scratch/ckpt/mfb_cross/best/mfb_baseline_ucf24.model-31000'
		if model is not None:
			saver.restore(sess, model)
			print('[*] Loading success: %s!'%model)
		else:
			print('[*] Loading failed ...')

	# Create loss output folder
	if not os.path.exists(loss_save_dir):
		os.makedirs(loss_save_dir)
	loss_file = open(os.path.join(loss_save_dir, prefix+'_test.txt'), 'a+')

	# Create test output folder
	if not os.path.exists(test_save_dir):
		os.makedirs(test_save_dir)

	total_steps = (FLAGS.num_sample / (FLAGS.num_gpus * FLAGS.batch_size)) * FLAGS.num_epochs

	# start queue runner
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	ffg_loss_list, fbg_loss_list, lfg_loss_list, feat_loss_list = [], [], [], []
	h5_file = h5py.File('features_nca_yfa.dat', 'w')
	try:
		with sess.as_default():
			print('\n\n\n*********** start testing ***********\n\n\n')
			step = global_step.eval()
			print('[step = %d]'%step)
			cnt = 0
			last_filename = ''
			mfb = mfb_list[0]
			fg_list, bg_list, mt_list = None, None, None
			avg_fg_feat = tf.squeeze(tf.contrib.layers.avg_pool2d(mfb.fg_feat, 4, 4, 'SAME'), [1, 2])
			avg_bg_feat = tf.squeeze(tf.contrib.layers.avg_pool2d(mfb.bg_feat, 4, 4, 'SAME'), [1, 2])
			avg_mt_feat = tf.squeeze(tf.contrib.layers.avg_pool2d(mfb.mt_feat, 4, 4, 'SAME'), [1, 2])
			#avg_fg_feat = tf.contrib.layers.flatten(mfb.fg_feat)
			#avg_bg_feat = tf.contrib.layers.flatten(mfb.bg_feat)
			#avg_mt_feat = tf.contrib.layers.flatten(mfb.mt_feat)
			while not coord.should_stop():
				# Run training steps or whatever
				cnt += 1
				filenames, fg, bg, mt = sess.run([filenames_list[0], avg_fg_feat, avg_bg_feat, avg_mt_feat])
				filename, fg, bg, mt = filenames[0].decode("utf-8"), fg, bg, mt

				if filename == last_filename:
					fg_list = np.concatenate((fg_list, fg))
					bg_list = np.concatenate((bg_list, bg))
					mt_list = np.concatenate((mt_list, mt))
				else:
					if fg_list is not None:
						#avg_vid_fg = np.average(fg_list, axis=0)
						#avg_vid_bg = np.average(bg_list, axis=0)
						#avg_vid_mt = np.average(mt_list, axis=0)
						avg_vid_fg = fg_list[fg_list.shape[0]//2]
						avg_vid_bg = bg_list[bg_list.shape[0]//2]
						avg_vid_mt = mt_list[mt_list.shape[0]//2]
						h5_file.attrs[last_filename + ':fg'] = avg_vid_fg
						h5_file.attrs[last_filename + ':bg'] = avg_vid_bg
						h5_file.attrs[last_filename + ':mt'] = avg_vid_mt

					last_filename = filename
					fg_list = fg
					bg_list = bg
					mt_list = mt

				for name in filenames:
					print(cnt, name)
				

	except tf.errors.OutOfRangeError:
		print('Done training -- epoch limit reached')
	finally:
		# When done, ask the threads to stop.
		coord.request_stop()

	# Wait for threads to finish.
	coord.join(threads)
	sess.close()
	h5_file.close()



def main(_):
	run_training()


if __name__ == '__main__':
	tf.app.run()