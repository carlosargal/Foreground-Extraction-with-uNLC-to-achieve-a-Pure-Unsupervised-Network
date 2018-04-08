import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='1'
from os import listdir

import sys
import time
import tools.opstmp

import numpy as np
np.set_printoptions(threshold=np.nan)

import tensorflow as tf
import scipy.misc as sm

from models.mfb_net_mask3 import mfb_net
from tools.utilities import *
from tools.opstmp import *


flags = tf.app.flags
flags.DEFINE_integer('batch_size', 1, 'Batch size.')
flags.DEFINE_integer('num_epochs', 2000, 'Number of epochs.')	# ~13 min per epoch
flags.DEFINE_integer('num_gpus', 1, 'Number of GPUs.')
flags.DEFINE_integer('seq_length', 16, 'Length of each video clip.')
flags.DEFINE_integer('height', 128, 'Height of video frame.')
flags.DEFINE_integer('width', 128, 'Width of video frame.')
flags.DEFINE_integer('channel', 3, 'Number of channels for each frame.')
flags.DEFINE_integer('num_sample', 1, 'Number of samples in this dataset.')

FLAGS = flags.FLAGS

prefix          = 'mfb_mask3'
model_save_dir  = './ckpt/' + prefix
logs_save_dir   = './logs/' + prefix
pred_save_dir   = './output/' + prefix
loss_save_dir   = './loss'
train_list_path = '/gpfs/home/bsc31/bsc31190/gpfs_projects/trainlist_single.txt'
dataset_path    = '/gpfs/home/bsc31/bsc31190/gpfs_projects/'

use_pretrained_model = False
save_predictions     = True


def decode_frames(frame_list, h, w, l):
	clip = []
	for i in range(l):
		frame = frame_list[i]
		image = tf.cast(tf.image.decode_jpeg(frame), tf.float32)
		image.set_shape((h, w, 3))
		clip.append(image)

	return tf.stack(clip)


def clip_value(x, min_, max_):
	if x < min_: x = min_
	if x > max_: x = max_
	return x
	

def generate_mask_and_bbx(bbx_list, h, w, l):
	img_masks, mask_bbxes = [], []

	for i in range(l):
		# generate image mask
		img_mask = np.zeros([h, w, 1])
		mask_bbx = np.zeros(4)
		minx, miny, maxx, maxy = w, h, 0, 0

		bbxes_str = bbx_list[i].eval().decode("utf-8")
		bbxes = bbxes_str.split(';')
		for i, bbx_str in enumerate(bbxes):
			bbx = bbx_str.split(',')
			mh, mw, x, y = \
				int(round(float(bbx[0]))), int(round(float(bbx[1]))), \
				int(round(float(bbx[2]))), int(round(float(bbx[3])))
			img_mask[(y if y > 0 else 0):y+mh, (x if x > 0 else 0):x+mw] = 1

			if x < minx: minx = x
			if y < miny: miny = y
			if x+mw > maxx: maxx = x+mw
			if y+mh > maxy: maxy = y+mh

		bh = maxy - miny + 1 # h
		bw = maxx - minx + 1 # w
		bx = minx			 # x
		by = miny			 # y
		mask_bbx[0] = clip_value(bh, 0, h-by)
		mask_bbx[1] = clip_value(bw, 0, w-bx)
		mask_bbx[2] = clip_value(bx, 0, w)
		mask_bbx[3] = clip_value(by, 0, h)
		mask_bbxes.append(mask_bbx)

		img_masks.append(tf.cast(tf.convert_to_tensor(img_mask), tf.float32))

	return tf.stack(img_masks), mask_bbxes


def crop_fg(clip, img_masks, mask_bbxes, h, w):
	fg_masks, fg_imgs = [], []

	for i in range(len(mask_bbxes)):
		mask_bbx = mask_bbxes[i]
		fg_img  = tf.image.resize_images(tf.image.crop_to_bounding_box( \
						clip[i], int(mask_bbx[3]), int(mask_bbx[2]), \
						int(mask_bbx[0]), int(mask_bbx[1])), [h,w])
		fg_mask = tf.image.resize_images(tf.image.crop_to_bounding_box( \
						img_masks[i], int(mask_bbx[3]), int(mask_bbx[2]), \
						int(mask_bbx[0]), int(mask_bbx[1])), [h,w])
		fg_masks.append(fg_mask)
		fg_imgs.append(fg_img)

		# normalize bbx to [-1,1]
		mask_bbx[0] = mask_bbx[0] / h * 2 - 1
		mask_bbx[1] = mask_bbx[1] / w * 2 - 1
		mask_bbx[2] = mask_bbx[2] / w * 2 - 1
		mask_bbx[3] = mask_bbx[3] / h * 2 - 1

		mask_bbxes[i] = tf.cast(tf.convert_to_tensor(mask_bbx), tf.float32)

	return tf.stack(fg_masks), tf.stack(fg_imgs), tf.stack(mask_bbxes)
	

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
	img_mask, mask_bbx = generate_mask_and_bbx(sequence_parsed['bbx'], height, width, sequence_length)
	fg_mask, fg_img, mask_bbx = crop_fg(clip, img_mask, mask_bbx, height, width)

	# randomly sample clips of 16 frames
	idx  = 16#np.random.randint(0, sequence_length-FLAGS.seq_length+1)
	clip = clip[idx:idx+FLAGS.seq_length] / 255.0 * 2 - 1
	img_mask = img_mask[idx:idx+FLAGS.seq_length]
	fg_img   = fg_img[idx:idx+FLAGS.seq_length] / 255.0 * 2 - 1
	fg_mask  = fg_mask[idx:idx+FLAGS.seq_length]
	mask_bbx = mask_bbx[idx:idx+FLAGS.seq_length]

	#save_fg_bg(clip.eval(), mask.eval())

	return clip, img_mask, fg_img, fg_mask, mask_bbx


def input_pipeline(filenames, batch_size, read_threads=1, num_epochs=None):
	filename_queue = tf.train.string_input_producer(
      					filenames, num_epochs=FLAGS.num_epochs, shuffle=True)
	# initialize local variables if num_epochs is not None or it'll raise uninitialized problem
	tf.get_default_session().run(tf.local_variables_initializer())

	example_list = [read_my_file_format(filename_queue) \
						for _ in range(read_threads)]

	min_after_dequeue = 500
	capacity = min_after_dequeue + 3 * batch_size
	clip_batch, img_mask_batch, fg_img_batch, fg_mask_batch, mask_bbx_batch = \
		tf.train.shuffle_batch_join(
	  		example_list, batch_size=batch_size, capacity=capacity,
				min_after_dequeue=min_after_dequeue)

	return clip_batch, img_mask_batch, fg_img_batch, fg_mask_batch, mask_bbx_batch


def tower_loss(name_scope, mfb, clips, img_masks, fg_imgs, fg_masks, mask_bbxes): # masks 16 16 128 128 1
	# get reconstruction and ground truth
	first_fg_rec, first_bg_rec, last_fg_rec = mfb.reconstruct()
	first_fg_mask, last_fg_mask = mfb.masks()
	first_fg_bbx, last_fg_bbx = mfb.bbx()
	#first_fg_mask_logits, last_fg_mask_logits = mfb.masks_logits()

	# calculate mask loss
	#first_fg_mask_loss = tf.reduce_mean(tf.square( \
	#							tf.reshape(img_masks[:,0,:,:],[-1,128,128,1])-first_fg_mask))
	#last_fg_mask_loss  = tf.reduce_mean(tf.square( \
	#							tf.reshape(img_masks[:,-1,:,:],[-1,128,128,1])-last_fg_mask))
	#first_fg_mask_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( \
	#							labels=img_masks[:,0,:,:], \
	#							logits=first_fg_mask_logits))
	#last_fg_mask_loss  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits( \
	#							labels=img_masks[:,-1,:,:], \
	#							logits=last_fg_mask_logits))
	#mask_loss          = first_fg_mask_loss + last_fg_mask_loss
	first_fg_mask_loss, last_fg_mask_loss = 0.0, 0.0

	# calculate mask bbx loss
	ffg_bbx_loss = tf.reduce_mean(tf.square(first_fg_bbx-mask_bbxes[:,0]))
	lfg_bbx_loss = tf.reduce_mean(tf.square(last_fg_bbx-mask_bbxes[:,-1]))
	bbx_loss     = ffg_bbx_loss + lfg_bbx_loss

	ffg_masks_list = [first_fg_mask for i in range(FLAGS.channel)]
	lfg_masks_list = [last_fg_mask for i in range(FLAGS.channel)]
	first_fg_mask = tf.concat(ffg_masks_list, axis=3)
	last_fg_mask  = tf.concat(lfg_masks_list, axis=3)

	first_fg_rec  = first_fg_rec * first_fg_mask
	last_fg_rec   = last_fg_rec  * last_fg_mask

	img_masks_list = [img_masks for i in range(FLAGS.channel)]
	fg_masks_list  = [fg_masks for i in range(FLAGS.channel)]
	img_masks = tf.concat(img_masks_list, axis=-1)
	fg_masks  = tf.concat(fg_masks_list, axis=-1)

	# mask the ground truth
	first_frames = clips[:,0,:,:,:]
	last_frames  = clips[:,-1,:,:,:]	
	first_fg_gt  = fg_imgs[:,0,:,:,:]  * fg_masks[:, 0]
	last_fg_gt   = fg_imgs[:,-1,:,:,:] * fg_masks[:, -1]

	# calculate reconstruction loss
	first_fg_loss = tf.reduce_mean(tf.abs(first_fg_rec-first_fg_gt))
	first_bg_loss = tf.reduce_mean(tf.abs(first_bg_rec-first_frames)*(1-img_masks[:, 0]))
	last_fg_loss  = tf.reduce_mean(tf.abs(last_fg_rec-last_fg_gt))
	rec_loss      = first_fg_loss + first_bg_loss + last_fg_loss

	weight_decay_loss_list = tf.get_collection('losses', name_scope)
	if len(weight_decay_loss_list) > 0:
		weight_decay_loss = tf.add_n(weight_decay_loss_list)

	tf.add_to_collection('losses', rec_loss)
	#tf.add_to_collection('losses', mask_loss)
	tf.add_to_collection('losses', bbx_loss)
	losses = tf.get_collection('losses', name_scope)

	# Calculate the total loss for the current tower.
	total_loss = tf.add_n(losses, name='total_loss')

	return total_loss, first_fg_loss, first_bg_loss, last_fg_loss, \
		   first_fg_mask_loss, last_fg_mask_loss, \
		   ffg_bbx_loss, lfg_bbx_loss


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
	tower_grads  = []
	tower_losses, tower_ffg_losses, tower_fbg_losses, tower_lfg_losses = [], [], [], []
	tower_mffg_losses, tower_mfbg_losses, tower_mlfg_losses = [], [], []
	tower_bffg_losses, tower_blfg_losses = [], []

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
		clips, img_masks, fg_imgs, fg_masks, mask_bbxes = \
				input_pipeline(train_list, FLAGS.batch_size, num_epochs=FLAGS.num_epochs)

	mfb_list = []
	with tf.variable_scope('vars') as var_scope:
		for gpu_index in range(FLAGS.num_gpus):
			with tf.device('/gpu:%d' % (gpu_index)):
				with tf.name_scope('%s_%d' % ('tower', gpu_index)) as scope:
				
					# construct model
					mfb = mfb_net(clips, FLAGS.height, FLAGS.width, FLAGS.seq_length, FLAGS.channel, FLAGS.batch_size)
					mfb_list.append(mfb)
					loss, first_fg_loss, first_bg_loss, last_fg_loss, mffg_loss, mlfg_loss, bffg_loss, blfg_loss = \
							tower_loss(scope, mfb, clips, img_masks, fg_imgs, fg_masks, mask_bbxes)

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
					tower_bffg_losses.append(bffg_loss)
					tower_blfg_losses.append(blfg_loss)


	# concatenate the losses of all towers
	loss_op      = tf.reduce_mean(tower_losses)
	ffg_loss_op  = tf.reduce_mean(tower_ffg_losses)
	fbg_loss_op  = tf.reduce_mean(tower_fbg_losses)
	lfg_loss_op  = tf.reduce_mean(tower_lfg_losses)
	mffg_loss_op = tf.reduce_mean(tower_mffg_losses)
	mlfg_loss_op = tf.reduce_mean(tower_mlfg_losses)
	bffg_loss_op = tf.reduce_mean(tower_bffg_losses)
	blfg_loss_op = tf.reduce_mean(tower_blfg_losses)

	tf.summary.scalar('loss', loss_op)
	tf.summary.scalar('ffg_loss', ffg_loss_op)
	tf.summary.scalar('fbg_loss', fbg_loss_op)
	tf.summary.scalar('lfg_loss', lfg_loss_op)
	tf.summary.scalar('mffg_loss', mffg_loss_op)
	tf.summary.scalar('mlfg_loss', mlfg_loss_op)
	tf.summary.scalar('bffg_loss', bffg_loss_op)
	tf.summary.scalar('blfg_loss', blfg_loss_op)

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
				sess.run(train_op)
				duration = time.time() - start_time
				step = global_step.eval()

				if step == 1 or step % 10 == 0: # evaluate loss
					loss, ffg_loss, fbg_loss, lfg_loss, mffg_loss, mlfg_loss, bffg_loss, blfg_loss, lr = \
						sess.run([loss_op, ffg_loss_op, fbg_loss_op, lfg_loss_op, mffg_loss_op, mlfg_loss_op, bffg_loss_op, blfg_loss_op, learning_rate])
					line = 'step %d/%d, loss=%.8f, dur=%.3f, lr=%.5f:\n' %(step, total_steps, loss, duration, lr) + \
						   '\t ffg=%.8f,  fbg=%.8f,  lfg=%.8f\n' %(ffg_loss, fbg_loss, lfg_loss) + \
						   '\tbffg=%.8f, blfg=%.8f\n' %(bffg_loss, blfg_loss)
						   #'\tmffg=%.8f, mlfg=%.8f\n' %(mffg_loss, mlfg_loss) + \
					print(line)
					loss_file.write(line + '\n')
					loss_file.flush()

				if step == 1 or step % 10 == 0: # save summary
					summary = summary_str = sess.run(merged)
					sum_writer.add_summary(summary, step)

				if step % 100 == 0 and save_predictions: # save current predictions
					mfb = mfb_list[0] # only visualize prediction in first tower
					ffg_, fbg, lfg_, mffg_, mlfg_, bffg, blfg = sess.run([
						mfb.first_fg_rec[0],  mfb.first_bg_rec[0],  mfb.last_fg_rec[0],
						mfb.first_fg_mask[0], mfb.last_fg_mask[0],
						mfb.first_fg_bbx[0],  mfb.last_fg_bbx[0]])

					ffg  = np.zeros_like(fbg)
					lfg  = np.zeros_like(fbg)
					mffg = np.zeros_like(mffg_)
					mlfg = np.zeros_like(mlfg_)

					bffg[bffg<-1] = -1
					bffg[bffg>1]  = 1
					blfg[blfg<-1] = -1
					blfg[blfg>1]  = 1					
					bffg, blfg = (bffg+1)/2*128, (blfg+1)/2*128
					fh, fw, fx, fy = int(bffg[0]), int(bffg[1]), int(bffg[2]), int(bffg[3])
					lh, lw, lx, ly = int(blfg[0]), int(blfg[1]), int(blfg[2]), int(blfg[3])

					# save reconstruction
					ffg_, fbg, lfg_ = (ffg_+1)/2*255.0, (fbg+1)/2*255.0, (lfg_+1)/2*255.0
					img  = gen_pred_img(ffg_, fbg, lfg_)

					# save mask
					mffg = add_patch_to_image(mffg, np.expand_dims(sm.imresize(np.squeeze(mffg_,-1), [fh, fw]), -1), fx, fy)
					mlfg = add_patch_to_image(mlfg, np.expand_dims(sm.imresize(np.squeeze(mlfg_,-1), [lh, lw]), -1), lx, ly)
					mfbg = 1 - mffg / 255
					mask = gen_pred_img(mffg, mfbg*255, mlfg)
					mask = np.reshape(mask, (mask.shape[0], mask.shape[1]))
					mask = np.stack([mask,mask,mask],-1)

					# save final image
					ffg = add_patch_to_image(ffg, sm.imresize(ffg_ * mffg_, [fh, fw]), fx, fy)
					lfg = add_patch_to_image(lfg, sm.imresize(lfg_ * mlfg_, [lh, lw]), lx, ly)
					fbg =  fbg * mfbg
					final = gen_pred_img(ffg, fbg, lfg)

					save_img = np.concatenate((img, mask, final))
					sm.imsave(os.path.join(pred_save_dir, '%07d.jpg'%step), save_img)

				if step % 100000 == 0 and step >= 100000: # save checkpoint
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