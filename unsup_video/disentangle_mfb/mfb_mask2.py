import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
from os import listdir

import sys
import time
import tools.ops
import subprocess

import numpy as np
np.set_printoptions(threshold=np.nan)

import tensorflow as tf
import scipy.misc as sm

from models.mfb_net_mask2 import mfb_net
from tools.utilities import *
from tools.ops import *


flags = tf.app.flags
flags.DEFINE_integer('batch_size', 12, 'Batch size.')
flags.DEFINE_integer('num_epochs', 2000, 'Number of epochs.')	# ~13 min per epoch
flags.DEFINE_integer('num_gpus', 4, 'Number of GPUs.')
flags.DEFINE_integer('seq_length', 16, 'Length of each video clip.')
flags.DEFINE_integer('height', 128, 'Height of video frame.')
flags.DEFINE_integer('width', 128, 'Width of video frame.')
flags.DEFINE_integer('channel', 3, 'Number of channels for each frame.')
flags.DEFINE_integer('num_sample', 9866, 'Number of samples in this dataset.')

FLAGS = flags.FLAGS

prefix          = 'mfb_mask2_loss1'
model_save_dir  = './ckpt/' + prefix
logs_save_dir   = './logs/' + prefix
pred_save_dir   = './output/' + prefix
loss_save_dir   = './loss'
train_list_path = '/gpfs/home/bsc31/bsc31190/gpfs_projects/trainlist.txt'
dataset_path    = '/gpfs/home/bsc31/bsc31190/gpfs_projects/UCF-101-tf-records-new'
evaluation_job  = './jobs/mfb_mask2_val'

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
    	"label": tf.FixedLenFeature([], dtype=tf.int64)
	}
	sequence_features = {
	    "frames": tf.FixedLenSequenceFeature([], dtype=tf.string),
	    "img_masks": tf.FixedLenSequenceFeature([], dtype=tf.string)
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
	img_mask, loss_mask = generate_mask(sequence_parsed['img_masks'], height, width, sequence_length)

	# randomly sample clips of 16 frames
	idx  = tf.squeeze(tf.random_uniform([1], 0, sequence_length-FLAGS.seq_length+1, dtype=tf.int32))
	clip = clip[idx:idx+FLAGS.seq_length] / 255.0 * 2 - 1
	img_mask  = img_mask[idx:idx+FLAGS.seq_length]
	loss_mask = loss_mask[idx:idx+FLAGS.seq_length]

	# randomly reverse data
	reverse   = tf.squeeze(tf.random_uniform([1], 0, 2, dtype=tf.int32))
	clip      = tf.cond(tf.equal(reverse,0), lambda: clip, lambda: clip[::-1])
	img_mask  = tf.cond(tf.equal(reverse,0), lambda: img_mask, lambda: img_mask[::-1])
	loss_mask = tf.cond(tf.equal(reverse,0), lambda: loss_mask, lambda: loss_mask[::-1])

	# randomly horizontally flip data
	flip      = tf.squeeze(tf.random_uniform([1], 0, 2, dtype=tf.int32))
	clip      = tf.cond(tf.equal(flip,0), lambda: clip, lambda: \
						tf.map_fn(lambda img: tf.image.flip_left_right(img), clip))
	img_mask  = tf.cond(tf.equal(flip,0), lambda: img_mask, lambda: \
						tf.squeeze(tf.map_fn(lambda img: tf.image.flip_left_right(img), tf.expand_dims(img_mask,-1)),-1))
	loss_mask = tf.cond(tf.equal(flip,0), lambda: loss_mask, lambda: \
						tf.squeeze(tf.map_fn(lambda img: tf.image.flip_left_right(img), tf.expand_dims(loss_mask,-1)),-1))

	clip.set_shape([FLAGS.seq_length, height, width, 3])
	img_mask.set_shape([FLAGS.seq_length, height, width])
	loss_mask.set_shape([FLAGS.seq_length, height, width])

	return clip, img_mask, loss_mask


def input_pipeline(filenames, batch_size, read_threads=4, num_epochs=None):
	filename_queue = tf.train.string_input_producer(
      					filenames, num_epochs=FLAGS.num_epochs, shuffle=True)
	# initialize local variables if num_epochs is not None or it'll raise uninitialized problem
	tf.get_default_session().run(tf.local_variables_initializer())

	example_list = [read_my_file_format(filename_queue) \
						for _ in range(read_threads)]

	min_after_dequeue = 300
	capacity = min_after_dequeue + 3 * batch_size
	clip_batch, img_mask_batch, loss_mask_batch = tf.train.shuffle_batch_join(
	  	example_list, batch_size=batch_size, capacity=capacity,
		min_after_dequeue=min_after_dequeue)

	return clip_batch, img_mask_batch, loss_mask_batch


def tower_loss(name_scope, mfb, clips, img_masks, loss_masks): # masks 16 16 128 128
	# get reconstruction and ground truth
	first_fg_rec, first_bg_rec, last_fg_rec = mfb.reconstruct()
	first_fg_mask, first_bg_mask, last_fg_mask = mfb.masks()
	first_fg_mask_logits, last_fg_mask_logits = mfb.masks_logits()

	# calculate mask loss
	#first_fg_mask_loss = tf.reduce_mean(tf.square( \
	#							tf.reshape(img_masks[:,0,:,:],[-1,128,128,1])-first_fg_mask))
	#last_fg_mask_loss  = tf.reduce_mean(tf.square( \
	#							tf.reshape(img_masks[:,-1,:,:],[-1,128,128,1])-last_fg_mask))
	first_fg_mask_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( \
								labels=img_masks[:,0,:,:], \
								logits=first_fg_mask_logits))
	last_fg_mask_loss  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( \
								labels=img_masks[:,-1,:,:], \
								logits=last_fg_mask_logits))
	mask_loss          = first_fg_mask_loss + last_fg_mask_loss

	ffg_masks_list = [first_fg_mask for i in range(FLAGS.channel)]
	fbg_masks_list = [first_bg_mask for i in range(FLAGS.channel)]
	lfg_masks_list = [last_fg_mask for i in range(FLAGS.channel)]
	first_fg_mask = tf.concat(ffg_masks_list, axis=3)
	first_bg_mask = tf.concat(fbg_masks_list, axis=3)
	last_fg_mask  = tf.concat(lfg_masks_list, axis=3)

	first_fg_rec  = first_fg_rec * first_fg_mask
	first_bg_rec  = first_bg_rec * first_bg_mask
	first_rec     = first_fg_rec + first_bg_rec
	last_fg_rec   = last_fg_rec  * last_fg_mask

	#s_total = FLAGS.height * FLAGS.width
	#s_ffg   = tf.reduce_sum(img_masks[:,0], [1,2])
	#s_lfg   = tf.reduce_sum(img_masks[:,-1], [1,2])
	#w1      = 1 + tf.abs(s_total - 2*s_ffg) / (s_ffg + 1e-5)
	#w2      = 1 + tf.abs(s_total - 2*s_ffg) / (s_total - s_ffg)
	#w3      = 1 + tf.abs(s_total - 2*s_lfg) / (s_lfg + 1e-5)

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
	first_fg_loss = tf.reduce_mean(tf.abs(first_fg_rec-first_fg_gt)*loss_masks[:,0])
	first_bg_loss = tf.reduce_mean(tf.abs(first_bg_rec-first_bg_gt))#*(1 - img_masks[:, 0]))
	last_fg_loss  = tf.reduce_mean(tf.abs(last_fg_rec-last_fg_gt)*loss_masks[:,-1])
	#first_fg_loss = tf.reduce_mean(tf.abs(first_fg_rec-first_frames)*img_masks[:,0]*tf.reshape(w1,[FLAGS.batch_size,1,1,1]))
	#first_bg_loss = tf.reduce_mean(tf.abs(first_bg_rec-first_frames)*(1-img_masks[:,0])*tf.reshape(w2,[FLAGS.batch_size,1,1,1]))
	#last_fg_loss  = tf.reduce_mean(tf.abs(last_fg_rec-last_frames)*img_masks[:,-1]*tf.reshape(w3,[FLAGS.batch_size,1,1,1]))
	rec_loss      = first_fg_loss + first_bg_loss + last_fg_loss

	weight_decay_loss_list = tf.get_collection('losses', name_scope)
	if len(weight_decay_loss_list) > 0:
		weight_decay_loss = tf.add_n(weight_decay_loss_list)

	tf.add_to_collection('losses', rec_loss)
	tf.add_to_collection('losses', mask_loss)
	losses = tf.get_collection('losses', name_scope)

	# Calculate the total loss for the current tower.
	total_loss = tf.add_n(losses, name='total_loss')

	return total_loss, first_fg_loss, first_bg_loss, last_fg_loss, \
		   first_fg_mask_loss, last_fg_mask_loss


def average_gradients(tower_grads):
	average_grads = []
	for grad_and_vars in zip(*tower_grads):
		grads = []
		skip  = False
		for g, v in grad_and_vars:
			skip = g is None
			if skip: continue
			expanded_g = tf.expand_dims(g, 0)
			grads.append(expanded_g)
		if skip: continue
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

	# Consturct computational graph
	"""clips_placeholder = tf.placeholder(tf.float32, shape=(
											FLAGS.batch_size*FLAGS.num_gpus,
											FLAGS.seq_length,
											FLAGS.height,
											FLAGS.width,
											FLAGS.channel))
	masks_placeholder = tf.placeholder(tf.bool, shape=(
											FLAGS.batch_size*FLAGS.num_gpus
											2,
											FLAGS.height,
											FLAGS.width))"""
	tower_grads  = []
	tower_losses, tower_ffg_losses, tower_fbg_losses, tower_lfg_losses = [], [], [], []
	tower_mffg_losses, tower_mfbg_losses, tower_mlfg_losses = [], [], []

	global_step = tf.get_variable(
                'global_step',
                [],
                initializer=tf.constant_initializer(0),
                trainable=False
                )
	starter_learning_rate = 1e-4
	learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           29000, 0.5, staircase=True)
	opt = tf.train.AdamOptimizer(learning_rate)

	# Create a session for running Ops on the Graph.
	config = tf.ConfigProto(allow_soft_placement=True,
							log_device_placement=True)
	#config.operation_timeout_in_ms = 10000
	sess = tf.Session(config=config)
	coord = tf.train.Coordinator()
	threads = None

	train_list_file = open(train_list_path, 'r')
	train_list = train_list_file.read().splitlines()
	for i, line in enumerate(train_list):
		train_list[i] = os.path.join(dataset_path, train_list[i]) 

	clips_list, img_masks_list, loss_masks_list = [], [], []
	with sess.as_default():
		for _ in range(FLAGS.num_gpus):
			clips, img_masks, loss_masks = input_pipeline(train_list, FLAGS.batch_size, num_epochs=FLAGS.num_epochs)
			clips_list.append(clips)
			img_masks_list.append(img_masks)
			loss_masks_list.append(loss_masks)		

	mfb_list = []
	with tf.variable_scope('vars') as var_scope:
		for gpu_index in range(FLAGS.num_gpus):
			with tf.device('/gpu:%d' % (gpu_index)):
				with tf.name_scope('%s_%d' % ('tower', gpu_index)) as scope:
				
					# construct model
					#clips = clips_placeholder[gpu_index*FLAGS.batch_size:(gpu_index+1)*FLAGS.batch_size,:,:,:,:]
					#masks = masks_placeholder[gpu_index*FLAGS.batch_size:(gpu_index+1)*FLAGS.batch_size,:,:,:]
					mfb = mfb_net(clips_list[gpu_index], FLAGS.height, FLAGS.width, FLAGS.seq_length, FLAGS.channel, FLAGS.batch_size)
					mfb_list.append(mfb)
					loss, first_fg_loss, first_bg_loss, last_fg_loss, mffg_loss, mlfg_loss = tower_loss(scope, mfb, clips_list[gpu_index], img_masks_list[gpu_index], loss_masks_list[gpu_index])

					var_scope.reuse_variables()

					vars_to_optimize = tf.trainable_variables()
					grads = opt.compute_gradients(loss, var_list=vars_to_optimize)

					tower_grads.append(grads)
					tower_losses.append(loss)
					tower_ffg_losses.append(first_fg_loss)
					tower_fbg_losses.append(first_bg_loss)
					tower_lfg_losses.append(last_fg_loss)
					tower_mffg_losses.append(mffg_loss)
					tower_mlfg_losses.append(mlfg_loss)


	# concatenate the losses of all towers
	loss_op      = tf.reduce_mean(tower_losses)
	ffg_loss_op  = tf.reduce_mean(tower_ffg_losses)
	fbg_loss_op  = tf.reduce_mean(tower_fbg_losses)
	lfg_loss_op  = tf.reduce_mean(tower_lfg_losses)
	mffg_loss_op = tf.reduce_mean(tower_mffg_losses)
	mlfg_loss_op = tf.reduce_mean(tower_mlfg_losses)

	tf.summary.scalar('loss', loss_op)
	tf.summary.scalar('ffg_loss', ffg_loss_op)
	tf.summary.scalar('fbg_loss', fbg_loss_op)
	tf.summary.scalar('lfg_loss', lfg_loss_op)
	tf.summary.scalar('mffg_loss', mffg_loss_op)
	tf.summary.scalar('mlfg_loss', mlfg_loss_op)

	update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
	grads = average_gradients(tower_grads)
	with tf.control_dependencies(update_ops):
		train_op = opt.apply_gradients(grads, global_step=global_step)

	# saver for saving checkpoints
	saver = tf.train.Saver(max_to_keep=10)
	init = tf.initialize_all_variables()

	sess.run(init)
	if not os.path.exists(model_save_dir):
		os.makedirs(model_save_dir)
	if use_pretrained_model:
		print('[*] Loading checkpoint ...')
		#saver.restore(sess, './ckpt/mfb_mask2_loss1/mfb_baseline_ucf24.model-76000')
		model = tf.train.latest_checkpoint(model_save_dir)
		if model is not None:
			saver.restore(sess, model)
			print('[*] Loading success: %s!'%model)
		else:
			print('[*] Loading failed ...')

	# Create summary writer
	merged = tf.summary.merge_all()
	if not os.path.exists(logs_save_dir):
		os.makedirs(logs_save_dir)
	sum_writer = tf.summary.FileWriter(logs_save_dir, sess.graph)

	# Create prediction output folder
	if not os.path.exists(pred_save_dir):
		os.makedirs(pred_save_dir)

	# Create loss output folder
	if not os.path.exists(loss_save_dir):
		os.makedirs(loss_save_dir)
	loss_file = open(os.path.join(loss_save_dir, prefix+'.txt'), 'a+')
	loss_file.write('\n\n\n\n')
	loss_file.flush()

	total_steps = (FLAGS.num_sample / (FLAGS.num_gpus * FLAGS.batch_size)) * FLAGS.num_epochs

	# start queue runner
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)

	try:
		with sess.as_default():
			print('\n\n\n*********** start training ***********\n\n\n')
			while not coord.should_stop():

				# Run training steps or whatever
				start_time = time.time()
				sess.run(train_op)
				duration = time.time() - start_time
				step = global_step.eval()

				if step == 1 or step % 10 == 0: # evaluate loss
					loss, ffg_loss, fbg_loss, lfg_loss, mffg_loss, mlfg_loss, lr = \
						sess.run([loss_op, ffg_loss_op, fbg_loss_op, lfg_loss_op, mffg_loss_op, mlfg_loss_op, learning_rate])
					line = 'step %d/%d, loss=%.8f, dur=%.3f, lr=%.5f:\n' %(step, total_steps, loss, duration, lr) + \
						   '\t ffg=%.8f,  fbg=%.8f,  lfg=%.8f\n' %(ffg_loss, fbg_loss, lfg_loss) + \
						   '\tmffg=%.8f, mlfg=%.8f\n' %(mffg_loss, mlfg_loss)
					print(line)
					loss_file.write(line + '\n')
					loss_file.flush()

				if step == 1 or step % 10 == 0: # save summary
					summary = summary_str = sess.run(merged)
					sum_writer.add_summary(summary, step)

				if step % 100 == 0 and save_predictions: # save current predictions
					mfb = mfb_list[0] # only visualize prediction in first tower
					gt_ffg, gt_fbg, gt_lfg, ffg, fbg, lfg, mffg, mfbg, mlfg = sess.run([
						mfb.gt_ffg, mfb.gt_fbg, mfb. gt_lfg, mfb.first_fg_rec,  mfb.first_bg_rec,  mfb.last_fg_rec,
						mfb.first_fg_mask, mfb.first_bg_mask, mfb.last_fg_mask])
					gt_ffg, gt_fbg, gt_lfg, ffg, fbg, lfg, mffg, mfbg, mlfg = \
							gt_ffg[0], gt_fbg[0], gt_lfg[0], ffg[0], fbg[0], lfg[0], mffg[0], mfbg[0], mlfg[0]
					ffg, fbg, lfg = (ffg+1)/2*255.0, (fbg+1)/2*255.0, (lfg+1)/2*255.0
					img = gen_pred_img(ffg, fbg, lfg)

					mask = gen_pred_img(mffg*255, mfbg*255, mlfg*255)
					mask = np.reshape(mask, (mask.shape[0], mask.shape[1]))
					mask = np.stack([mask,mask,mask],-1)

					ffg, fbg, lfg = ffg * mffg, fbg * mfbg, lfg * mlfg
					final = gen_pred_img(ffg, fbg, lfg)

					gt_ffg, gt_fbg, gt_lfg = (gt_ffg+1)/2*255.0, (gt_fbg+1)/2*255.0, (gt_lfg+1)/2*255.0
					gt = gen_pred_img(gt_ffg, gt_fbg, gt_lfg)

					save_img = np.concatenate((img, mask, final, gt))
					sm.imsave(os.path.join(pred_save_dir, '%07d.jpg'%step), save_img)

				if step % 500 == 0 and step >= 500: # save checkpoint
					saver.save(sess, os.path.join(model_save_dir, model_filename), global_step=global_step)

				if step % 500 == 0:
					subprocess.check_output(['mnsubmit', evaluation_job])


	except tf.errors.OutOfRangeError:
		print('Done training -- epoch limit reached')
	finally:
		# When done, ask the threads to stop.
		coord.request_stop()

	# Wait for threads to finish.
	coord.join(threads)
	sess.close()



def main(_):
	run_training()


if __name__ == '__main__':
	tf.app.run()