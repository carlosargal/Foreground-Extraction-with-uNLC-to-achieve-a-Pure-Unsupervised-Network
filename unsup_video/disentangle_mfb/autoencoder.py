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

from models.autoencoder_net import autoencoder_net
from tools.utilities import *
from tools.ops import *
from random import randint


flags = tf.app.flags
flags.DEFINE_integer('batch_size', 10, 'Batch size.')
flags.DEFINE_integer('num_epochs', 2000, 'Number of epochs.')	# ~13 min per epoch
flags.DEFINE_integer('num_gpus', 4, 'Number of GPUs.')
flags.DEFINE_integer('seq_length', 16, 'Length of each video clip.')
flags.DEFINE_integer('height', 128, 'Height of video frame.')
flags.DEFINE_integer('width', 128, 'Width of video frame.')
flags.DEFINE_integer('channel', 3, 'Number of channels for each frame.')
flags.DEFINE_integer('num_sample', 9866, 'Number of samples in this dataset.')

FLAGS = flags.FLAGS

prefix          = 'autoencoder'
model_save_dir  = './ckpt/' + prefix
logs_save_dir   = './logs/' + prefix
pred_save_dir   = './output/' + prefix
loss_save_dir   = './loss'
train_list_path = '/gpfs/home/bsc31/bsc31190/gpfs_projects/trainlist.txt'
dataset_path    = '/gpfs/home/bsc31/bsc31190/gpfs_projects/UCF-101-tf-records-new'
evaluation_job  = './jobs/autoencoder_val'

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

	# randomly sample clips of 16 frames
	idx  = tf.squeeze(tf.random_uniform([1], 0, sequence_length-FLAGS.seq_length+1, dtype=tf.int32))
	clip = clip[idx:idx+FLAGS.seq_length] / 255.0 * 2 - 1

	# randomly reverse data
	reverse   = tf.squeeze(tf.random_uniform([1], 0, 2, dtype=tf.int32))
	clip      = tf.cond(tf.equal(reverse,0), lambda: clip, lambda: clip[::-1])
	clip.set_shape([FLAGS.seq_length, height, width, 3])

	# randomly horizontally flip data
	flip      = tf.squeeze(tf.random_uniform([1], 0, 2, dtype=tf.int32))
	img_list  = tf.unstack(clip)
	flip_clip = []
	for img in img_list:
		flip_clip.append(tf.cond(tf.equal(flip, 0), lambda: img, lambda: tf.image.flip_left_right(img)))
	clip = tf.stack(flip_clip)
	#flip_clip = tf.map_fn(lambda img: tf.image.flip_left_right(img), clip)
	#clip      = tf.cond(tf.equal(flip,0), lambda: clip, lambda: flip_clip)	

	clip.set_shape([FLAGS.seq_length, height, width, 3])

	return [clip]


def input_pipeline(filenames, batch_size, read_threads=4, num_epochs=None):
	filename_queue = tf.train.string_input_producer(
      					filenames, num_epochs=FLAGS.num_epochs, shuffle=True)
	# initialize local variables if num_epochs is not None or it'll raise uninitialized problem
	tf.get_default_session().run(tf.local_variables_initializer())

	example_list = [read_my_file_format(filename_queue) \
						for _ in range(read_threads)]

	min_after_dequeue = 300
	capacity = min_after_dequeue + 3 * batch_size
	clip_batch = tf.train.shuffle_batch_join(
	  	example_list, batch_size=batch_size, capacity=capacity,
		min_after_dequeue=min_after_dequeue)

	return clip_batch


def tower_loss(name_scope, autoencoder, clips): # masks 16 16 128 128
	# calculate reconstruction loss
	rec_loss = tf.reduce_mean(tf.abs(clips-autoencoder.rec_vid))

	weight_decay_loss_list = tf.get_collection('losses', name_scope)
	weight_decay_loss = 0.0
	if len(weight_decay_loss_list) > 0:
		weight_decay_loss = tf.add_n(weight_decay_loss_list)

	tf.add_to_collection('losses', rec_loss)
	#tf.add_to_collection('losses', mask_loss)
	losses = tf.get_collection('losses', name_scope)

	# Calculate the total loss for the current tower.
	total_loss = tf.add_n(losses, name='total_loss')

	return total_loss, rec_loss, weight_decay_loss


def average_gradients(tower_grads):
	average_grads = []
	for grad_and_vars in zip(*tower_grads):
		grads = []
		for g, v in grad_and_vars:
			if g is None:
				print(v.name)
				input('here')
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

	# Consturct computational graph
	tower_grads  = []
	tower_losses, tower_rec_losses, tower_wd_losses = [], [], []

	global_step = tf.get_variable(
                'global_step',
                [],
                initializer=tf.constant_initializer(0),
                trainable=False
                )
	starter_learning_rate = 1e-4
	learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           5150000, 0.8, staircase=True)
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

	num_val = len(train_list)
	assert(num_val % 4 == 0)
	num_for_each_gpu = num_val // 4

	clips_list = []
	with sess.as_default():
		for i in range(FLAGS.num_gpus):
			clips = input_pipeline(train_list[i*num_for_each_gpu:(i+1)*num_for_each_gpu], \
									FLAGS.batch_size, num_epochs=FLAGS.num_epochs)
			clips_list.append(clips)

	autoencoder_list = []
	with tf.variable_scope('vars') as var_scope:
		for gpu_index in range(FLAGS.num_gpus):
			with tf.device('/gpu:%d' % (gpu_index)):
				with tf.name_scope('%s_%d' % ('tower', gpu_index)) as scope:
				
					# construct model
					autoencoder = autoencoder_net(clips_list[gpu_index], FLAGS.height, FLAGS.width, FLAGS.seq_length, FLAGS.channel, FLAGS.batch_size)
					autoencoder_list.append(autoencoder)
					loss, rec_loss, wd_loss = tower_loss(scope, autoencoder, clips_list[gpu_index])

					var_scope.reuse_variables()

					vars_to_optimize = tf.trainable_variables()
					grads = opt.compute_gradients(loss, var_list=vars_to_optimize)

					tower_grads.append(grads)
					tower_losses.append(loss)
					tower_rec_losses.append(rec_loss)
					tower_wd_losses.append(wd_loss)

	# concatenate the losses of all towers
	loss_op     = tf.reduce_mean(tower_losses)
	rec_loss_op = tf.reduce_mean(tower_rec_losses)
	wd_loss_op  = tf.reduce_mean(tower_wd_losses)

	tf.summary.scalar('loss', loss_op)
	tf.summary.scalar('rec_loss', rec_loss_op)
	tf.summary.scalar('wd_loss', wd_loss_op)

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
		#saver.restore(sess, './ckpt/mfb_baseline/mfb_baseline_ucf24.model-51000')
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
	loss_file = open(os.path.join(loss_save_dir, prefix+'.txt'), 'w')

	total_steps = (FLAGS.num_sample / (FLAGS.num_gpus * FLAGS.batch_size)) * FLAGS.num_epochs

	# start queue runner
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	gpu_idx = 0

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
					loss, rec_loss, wd_loss, lr = sess.run([loss_op, rec_loss_op, wd_loss_op, learning_rate])
					line = 'step %d/%d, loss=%.8f, rec=%.8f, lwd=%.8f, dur=%.3f, lr=%.8f' \
							%(step, total_steps, loss, rec_loss, wd_loss, duration, lr)
					print(line)
					loss_file.write(line + '\n')
					loss_file.flush()

				if step == 1 or step % 10 == 0: # save summary
					summary = summary_str = sess.run(merged)
					sum_writer.add_summary(summary, step)

				if step % 100 == 0 and save_predictions: # save current predictions
					clips = clips_list[gpu_idx]
					autoencoder = autoencoder_list[gpu_idx]
					gt_vid, rec_vid = sess.run([clips, autoencoder.rec_vid])
					gt_vid, rec_vid = gt_vid[0], rec_vid[0]
					gt_vid, rec_vid = (gt_vid+1)/2*255.0, (rec_vid+1)/2*255.0
					rec_img = gen_pred_vid(rec_vid)
					gt_img  = gen_pred_vid(gt_vid)
					save_img = np.concatenate((rec_img, gt_img))
					sm.imsave(os.path.join(pred_save_dir, '%07d.jpg'%step), save_img)

					gpu_idx += 1
					if gpu_idx == FLAGS.num_gpus:
						gpu_idx = 0

				if step % 500 == 0: # save checkpoint
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