import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
from os import listdir

import sys
import time
import tools.ops

import numpy as np
np.set_printoptions(threshold=np.nan)

import tensorflow as tf
import scipy.misc as sm

from models.mfb_net_mask import mfb_net
from tools.utilities import *
from tools.ops import *


flags = tf.app.flags
flags.DEFINE_integer('batch_size', 16, 'Batch size.')
flags.DEFINE_integer('num_epochs', 2000, 'Number of epochs.')	# ~13 min per epoch
flags.DEFINE_integer('num_gpus', 4, 'Number of GPUs.')
flags.DEFINE_integer('seq_length', 16, 'Length of each video clip.')
flags.DEFINE_integer('height', 128, 'Height of video frame.')
flags.DEFINE_integer('width', 128, 'Width of video frame.')
flags.DEFINE_integer('channel', 3, 'Number of channels for each frame.')
flags.DEFINE_integer('num_sample', 11107, 'Number of samples in this dataset.')

FLAGS = flags.FLAGS

prefix          = 'mfb_mask'
model_save_dir  = './ckpt/' + prefix
logs_save_dir   = './logs/' + prefix
pred_save_dir   = './output/' + prefix
loss_save_dir   = './loss'
train_list_path = '/gpfs/home/bsc31/bsc31190/gpfs_projects/trainlist.txt'
dataset_path    = '/gpfs/home/bsc31/bsc31190/gpfs_projects/UCF-101-tf-records'

use_pretrained_model = True
save_loss            = False
save_summaries       = False
save_predictions     = True
save_checkpoint      = False


def decode_frames(frame_list, h, w, l):
	clip = []
	for i in range(l):
		frame = frame_list[i]
		image = tf.cast(tf.image.decode_jpeg(frame), tf.float32)
		image.set_shape((h, w, 3))
		clip.append(image)

	return tf.stack(clip)


def generate_mask(bbx_list, h, w, l):
	img_masks, loss_masks = [], []

	for i in range(l):
		# generate image mask
		img_mask = np.zeros([h, w])
		bbxes_str = bbx_list[i].eval().decode("utf-8")
		bbxes = bbxes_str.split(';')
		for i, bbx_str in enumerate(bbxes):
			bbx = bbx_str.split(',')
			mh, mw, x, y = \
				int(round(float(bbx[0]))), int(round(float(bbx[1]))), \
				int(round(float(bbx[2]))), int(round(float(bbx[3])))
			img_mask[(y if y > 0 else 0):y+mh, (x if x > 0 else 0):x+mw] = 1
		img_masks.append(tf.cast(tf.convert_to_tensor(img_mask), tf.float32))

		# generate loss mask
		s_total = h * w
		s_mask  = np.sum(img_mask)
		if s_mask == 0:
			loss_mask = np.zeros_like(img_mask)
		elif s_mask < s_total / 2:
			loss_mask = img_mask * ((s_total-s_mask)/s_mask-1)
			loss_mask = loss_mask + 1
		else:
			loss_mask = np.ones_like(img_mask)
		loss_masks.append(tf.cast(tf.convert_to_tensor(loss_mask), tf.float32))

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
	    "bbx": tf.FixedLenSequenceFeature([], dtype=tf.string)
	}
	context_parsed, sequence_parsed = tf.parse_single_sequence_example(
	    serialized=serialized_example,
	    context_features=context_features,
	    sequence_features=sequence_features
	)

	# start queue runner so it won't stuck
	tf.train.start_queue_runners(sess=tf.get_default_session())

	height = context_parsed['height'].eval()
	width  = context_parsed['width'].eval()
	sequence_length = context_parsed['sequence_length'].eval()

	clip = decode_frames(sequence_parsed['frames'], height, width, sequence_length)
	img_mask, loss_mask = generate_mask(sequence_parsed['bbx'], height, width, sequence_length)

	# randomly sample clips of 16 frames
	idx  = 16#np.random.randint(0, sequence_length-FLAGS.seq_length+1)
	clip = clip[idx:idx+FLAGS.seq_length] #/ 255.0 * 2 - 1
	img_mask  = img_mask[idx:idx+FLAGS.seq_length]
	loss_mask = loss_mask[idx:idx+FLAGS.seq_length]

	#save_fg_bg(clip.eval(), mask.eval())
	#input('here')

	return clip, img_mask, loss_mask


def input_pipeline(filenames, batch_size, read_threads=8, num_epochs=None):
	filename_queue = tf.train.string_input_producer(
      					filenames, num_epochs=FLAGS.num_epochs, shuffle=True)
	# initialize local variables if num_epochs is not None or it'll raise uninitialized problem
	tf.get_default_session().run(tf.local_variables_initializer())

	example_list = [read_my_file_format(filename_queue) \
						for _ in range(read_threads)]

	min_after_dequeue = 500
	capacity = min_after_dequeue + 3 * batch_size
	clip_batch, img_mask_batch, loss_mask_batch = tf.train.shuffle_batch_join(
	  	example_list, batch_size=batch_size, capacity=capacity,
		min_after_dequeue=min_after_dequeue)

	return clip_batch, img_mask_batch, loss_mask_batch


def tower_loss(name_scope, mfb, clips, img_masks, loss_masks): # masks 16 16 128 128
	# get reconstruction and ground truth
	first_fg_rec, first_bg_rec, last_fg_rec = mfb.reconstruct()
	first_fg_mask, first_bg_mask, last_fg_mask = mfb.masks()
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

	# calculate reconstruction loss
	first_fg_loss = tf.reduce_mean(tf.abs(first_fg_rec-first_fg_gt)*loss_masks[:,0])
	first_bg_loss = tf.reduce_mean(tf.abs(first_bg_rec-first_bg_gt))
	last_fg_loss  = tf.reduce_mean(tf.abs(last_fg_rec-last_fg_gt)*loss_masks[:,-1])
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

	weight_decay_loss_list = tf.get_collection('losses', name_scope)
	if len(weight_decay_loss_list) > 0:
		weight_decay_loss = tf.add_n(weight_decay_loss_list)

	tf.add_to_collection('losses', rec_loss)
	#tf.add_to_collection('losses', mask_loss)
	losses = tf.get_collection('losses', name_scope)

	# Calculate the total loss for the current tower.
	total_loss = tf.add_n(losses, name='total_loss')

	return total_loss, first_fg_loss, first_bg_loss, last_fg_loss#, \
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
	tower_ffg_m_losses, tower_fbg_m_losses, tower_lfg_m_losses = [], [], []

	global_step = tf.get_variable(
                'global_step',
                [],
                initializer=tf.constant_initializer(0),
                trainable=False
                )
	starter_learning_rate = 1e-4
	learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step,
                                           100000, 0.5, staircase=True)
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
	with sess.as_default():
		clips, img_masks, loss_masks = input_pipeline(train_list, FLAGS.batch_size, num_epochs=FLAGS.num_epochs)

	mfb_list = []
	with tf.variable_scope('vars') as var_scope:
		for gpu_index in range(FLAGS.num_gpus):
			with tf.device('/gpu:%d' % (gpu_index)):
				with tf.name_scope('%s_%d' % ('tower', gpu_index)) as scope:
				
					# construct model
					#clips = clips_placeholder[gpu_index*FLAGS.batch_size:(gpu_index+1)*FLAGS.batch_size,:,:,:,:]
					#masks = masks_placeholder[gpu_index*FLAGS.batch_size:(gpu_index+1)*FLAGS.batch_size,:,:,:]
					mfb = mfb_net(clips, FLAGS.height, FLAGS.width, FLAGS.seq_length, FLAGS.channel, FLAGS.batch_size)
					mfb_list.append(mfb)
					loss, first_fg_loss, first_bg_loss, last_fg_loss = tower_loss(scope, mfb, clips, img_masks, loss_masks)

					var_scope.reuse_variables()

					vars_to_optimize = tf.trainable_variables()
					grads = opt.compute_gradients(loss, var_list=vars_to_optimize)

					tower_grads.append(grads)
					tower_losses.append(loss)
					tower_ffg_losses.append(first_fg_loss)
					tower_fbg_losses.append(first_bg_loss)
					tower_lfg_losses.append(last_fg_loss)


	# concatenate the losses of all towers
	loss_op     = tf.reduce_mean(tower_losses)
	ffg_loss_op = tf.reduce_mean(tower_ffg_losses)
	fbg_loss_op = tf.reduce_mean(tower_fbg_losses)
	lfg_loss_op = tf.reduce_mean(tower_lfg_losses)

	tf.summary.scalar('loss', loss_op)
	tf.summary.scalar('ffg_loss', ffg_loss_op)
	tf.summary.scalar('fbg_loss', fbg_loss_op)
	tf.summary.scalar('lfg_loss', lfg_loss_op)

	grads = average_gradients(tower_grads)
	apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
	train_op = apply_gradient_op

	# saver for saving checkpoints
	saver = tf.train.Saver()
	init = tf.initialize_all_variables()

	sess.run(init)
	if not os.path.exists(model_save_dir):
		os.makedirs(model_save_dir)
	if use_pretrained_model:
		print('[*] Loading checkpoint ...')
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
	if save_loss:
		loss_file = open(os.path.join(loss_save_dir, prefix+'.txt'), 'w')

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
				#sess.run(train_op)
				duration = time.time() - start_time
				step = global_step.eval()

				if step % 10 == 0 and save_loss: # evaluate loss
					loss, ffg_loss, fbg_loss, lfg_loss, lr = \
						sess.run([loss_op, ffg_loss_op, fbg_loss_op, lfg_loss_op, learning_rate])
					line = 'step %d/%d, loss=%.8f,  ffg=%.8f,  fbg=%.8f,  lfg=%.8f, dur=%.3f, lr=%.8f' \
							%(step, total_steps, loss, ffg_loss, fbg_loss, lfg_loss, duration, lr)
					print(line)
					loss_file.write(line + '\n')
					loss_file.flush()

				if step % 10 == 0 and save_summaries: # save summary
					summary = summary_str = sess.run(merged)
					sum_writer.add_summary(summary, step)

				if step % 10 == 0 and save_predictions: # save current predictions
					mfb = mfb_list[0] # only visualize prediction in first tower
					ffg, fbg, lfg, mffg, mfbg, mlfg = sess.run([
						mfb.first_fg_rec[0],  mfb.first_bg_rec[0],  mfb.last_fg_rec[0],
						mfb.first_fg_mask[0], mfb.first_bg_mask[0], mfb.last_fg_mask[0]])
					#ffg, fbg, lfg = ffg * mffg, fbg * mfbg, lfg * mlfg
					#ffg, fbg, lfg = (ffg+1)/2*255.0, (fbg+1)/2*255.0, (lfg+1)/2*255.0
					msk = gen_pred_img(mffg*255, mfbg*255, mlfg*255)
					img = gen_pred_img(ffg, fbg, lfg)

					sm.imsave(os.path.join(pred_save_dir, '%07d_img.jpg'%step), img)
					sm.imsave(os.path.join(pred_save_dir, '%07d_msk.jpg'%step), np.reshape(msk, [132,392]))

					ffg, fbg, lfg = ffg * mffg, fbg * mfbg, lfg * mlfg
					img = gen_pred_img(ffg, fbg, lfg)
					sm.imsave(os.path.join(pred_save_dir, '%07d_final.jpg'%step), img)

				if step % 30 == 0 and save_checkpoint: # save checkpoint
					saver.save(sess, os.path.join(model_save_dir, model_filename), global_step=global_step)

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