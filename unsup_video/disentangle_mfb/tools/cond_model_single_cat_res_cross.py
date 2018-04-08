import os
import time
import ipdb
from glob import glob
import tensorflow as tf

from ops import *
from utils import *

class CONDNET(object):
  def __init__(self, image_size=128, batch_size=32, c_dim=3,
               seen_step=10, gap_step=5, checkpoint_dir=None):

    self.batch_size  = batch_size
    self.image_size  = image_size

    self.gf_dim = 64

    self.c_dim      = 3
    self.num_step   = seen_step + gap_step
    self.seen_step  = seen_step
    self.gap_step   = gap_step
    self.seq_shape  = [image_size, image_size, 1+self.num_step, self.c_dim]
    self.pose_shape = [image_size, image_size, self.num_step, 48]

    self.checkpoint_dir = checkpoint_dir
    self.build_model()

  def build_model(self):
    self.seq = tf.placeholder(tf.float32,
                              [self.batch_size] + self.seq_shape,
                              name='seq')
    self.pose   = tf.placeholder(tf.float32,
                                 [self.batch_size] + self.pose_shape,
                                 name='pose')

#    img_in = self.seq[:,:,:,0,:]
    self.G = tf.reshape(self.generator(self.seq, self.pose),
                        [self.batch_size,
                         self.image_size,
                         self.image_size,
                         1,
                         self.c_dim])

    self.loss     = tf.reduce_mean(tf.abs(self.G[:,:,:,0,:]-self.seq[:,:,:,self.seq.get_shape()[3]-1,:]))
    self.G_batch  = tf.transpose(self.G,[0,3,1,2,4])
    self.G_batch  = tf.reshape(self.G_batch,[-1,self.image_size,self.image_size,3])
    self.G_sum    = tf.image_summary("G_batch", self.G_batch)
    self.loss_sum = tf.scalar_summary("loss", self.loss)
  
    self.t_vars = tf.trainable_variables()
    self.g_vars = [var for var in self.t_vars]
    self.saver = tf.train.Saver(self.g_vars)

  def generator(self, reuse=False):
    if reuse:
      tf.get_variable_scope().reuse_variables()

    xtm1 = self.seq[:,:,:,self.seen_step-1,:]
    xpt = self.pose[:,:,:,self.num_step-1,:]
    # img encoder
    h_img, res_in_img = self.img_encoder(xtm1)

    # content encoder
    h_pose, h_3, h_2, h_1, res_in_pose = self.pose_encoder(xpt)

    # combination layers
    h_comb = self.comb_layers(h_img, h_pose)

    # residual computation
    assert len(res_in_img)==len(res_in_pose)
    residual_vals = self.residual(res_in_img,res_in_pose)

    # difference encoder
    h_diff = self.diff_encoder()

    # kernel decoder
    kernels = self.kernel_decoder(h_diff)

    # decoder
    xtp1   = self.decoder(h_comb, kernels, h_3, h_2, h_1, residual_vals)

    return xtp1

  def img_encoder(self, xp):
    res_in  = []

    conv1_1 = relu(conv2d(xp, output_dim=self.gf_dim,k_h=3,k_w=3,d_h=1,d_w=1,name='iconv1_1'))
    conv1_2 = relu(conv2d(conv1_1, output_dim=self.gf_dim,k_h=3,k_w=3,d_h=1,d_w=1,name='iconv1_2'))
    res_in.append(conv1_2)
    pool1   = MaxPooling(conv1_2, [2,2])

    conv2_1 = relu(conv2d(pool1, output_dim=self.gf_dim*2,k_h=3,k_w=3,d_h=1,d_w=1,name='iconv2_1'))
    conv2_2 = relu(conv2d(conv2_1, output_dim=self.gf_dim*2,k_h=3,k_w=3,d_h=1,d_w=1,name='iconv2_2'))
    res_in.append(conv2_2)
    pool2  = MaxPooling(conv2_2, [2,2])

    conv3_1 = relu(conv2d(pool2, output_dim=self.gf_dim*4,k_h=3,k_w=3,d_h=1,d_w=1,name='iconv3_1'))
    conv3_2 = relu(conv2d(conv3_1, output_dim=self.gf_dim*4,k_h=3,k_w=3,d_h=1,d_w=1,name='iconv3_2'))
    conv3_3 = relu(conv2d(conv3_2, output_dim=self.gf_dim*4,k_h=3,k_w=3,d_h=1,d_w=1,name='h_img'))
    res_in.append(conv3_3)
    h_img  = MaxPooling(conv3_3, [2,2])

    return h_img, res_in

 
  def pose_encoder(self, xp):
    res_in  = []

    conv1_1 = relu(conv2d(xp, output_dim=self.gf_dim,k_h=3,k_w=3,d_h=1,d_w=1,name='pconv1_1'))
    conv1_2 = relu(conv2d(conv1_1, output_dim=self.gf_dim,k_h=3,k_w=3,d_h=1,d_w=1,name='pconv1_2'))
    res_in.append(conv1_2)
    pool1   = MaxPooling(conv1_2, [2,2])

    conv2_1 = relu(conv2d(pool1, output_dim=self.gf_dim*2,k_h=3,k_w=3,d_h=1,d_w=1,name='pconv2_1'))
    conv2_2 = relu(conv2d(conv2_1, output_dim=self.gf_dim*2,k_h=3,k_w=3,d_h=1,d_w=1,name='pconv2_2'))
    res_in.append(conv2_2)
    pool2  = MaxPooling(conv2_2, [2,2])

    conv3_1 = relu(conv2d(pool2, output_dim=self.gf_dim*4,k_h=3,k_w=3,d_h=1,d_w=1,name='pconv3_1'))
    conv3_2 = relu(conv2d(conv3_1, output_dim=self.gf_dim*4,k_h=3,k_w=3,d_h=1,d_w=1,name='pconv3_2'))
    conv3_3 = relu(conv2d(conv3_2, output_dim=self.gf_dim*4,k_h=3,k_w=3,d_h=1,d_w=1,name='h_pose'))
    res_in.append(conv3_3)
    h_pose  = MaxPooling(conv3_3, [2,2])

    return h_pose, conv3_3, conv2_2, conv1_2, res_in

  def diff_encoder(self):
    h_diff_list = list()
    
    for i in xrange(self.seen_step-1):
        img_diff  = tf.sub(self.seq[:,:,:,i+1,:], self.seq[:,:,:,i,:])
        pose_diff = tf.sub(self.pose[:,:,:,i+1,:], self.pose[:,:,:,i,:])
        xp = tf.concat(3, [img_diff, pose_diff])

        conv1_1 = relu(conv2d(xp, output_dim=self.gf_dim,k_h=3,k_w=3,d_h=1,d_w=1,name='dconv1_1'))
        conv1_2 = relu(conv2d(conv1_1, output_dim=self.gf_dim,k_h=3,k_w=3,d_h=1,d_w=1,name='dconv1_2'))
        pool1   = MaxPooling(conv1_2, [2,2])

        conv2_1 = relu(conv2d(pool1, output_dim=self.gf_dim*2,k_h=3,k_w=3,d_h=1,d_w=1,name='dconv2_1'))
        conv2_2 = relu(conv2d(conv2_1, output_dim=self.gf_dim*2,k_h=3,k_w=3,d_h=1,d_w=1,name='dconv2_2'))
        pool2  = MaxPooling(conv2_2, [2,2])

        conv3_1 = relu(conv2d(pool2, output_dim=self.gf_dim*4,k_h=3,k_w=3,d_h=1,d_w=1,name='dconv3_1'))
        conv3_2 = relu(conv2d(conv3_1, output_dim=self.gf_dim*4,k_h=3,k_w=3,d_h=1,d_w=1,name='dconv3_2'))
        conv3_3 = relu(conv2d(conv3_2, output_dim=self.gf_dim*4,k_h=3,k_w=3,d_h=1,d_w=1,name='h_diff'))
        h_diff  = MaxPooling(conv3_3, [2,2])
        h_diff_list.append(h_diff)

    return h_diff_list

  def kernel_decoder(self, h_diff):
    h_kernel = conv_lstm(h_diff, self.seen_steps, num_layers=3, num_out_channel=256, num_channels=[256,256,256])
    assert h_kernel.shape[1] == h_kernel.shape[2] == 16
    
    conv1 = conv2d(h_kernel, output_dim=256*2,k_h=3,k_w=3,d_h=1,d_w=1,name='kernel_conv1')
    pool1 = MaxPooling(conv1, [2,2])
    conv2 = conv2d(pool1, output_dim=256*4,k_h=3,k_w=3,d_h=1,d_w=1,name='kernel_conv2')
    pool2 = MaxPooling(conv2, [2,2])
    conv3 = conv2d(pool2, output_dim=256*256+256*128+128*64,k_h=2,k_w=2,d_h=1,d_w=1,
                   padding='VALID',name='kernel_conv3')
    assert conv3.shape[1] == conv3.shape[2] == 3

    kernels = list()
    kernels.append(tf.reshape(conv3[:,:,:,:2*256*256], [-1,3,3,256,2*256]))
    kernels.append(tf.reshape(conv3[:,:,:,:2*128*128], [-1,3,3,128,2*128]))
    kernels.append(tf.reshape(conv3[:,:,:,:2*64*64], [-1,3,3,64,64]))

    return kernels

  def residual(self, input_dyn, input_cont):
    n_layers  = len(input_dyn)
    res_out   = []
    for l in xrange(n_layers):
      input   = tf.concat(3,[input_dyn[l],input_cont[l]])
      out_dim = input_cont[l].get_shape()[3]
      res1 = relu(conv2d(input,output_dim=out_dim,
                         k_h=3,k_w=3,d_h=1,d_w=1,name='res'+str(l)+'_1'))
      res2 = conv2d(res1,output_dim=out_dim,
                    k_h=3,k_w=3,d_h=1,d_w=1,name='res'+str(l)+'_2')
      res_out.append(res2)
    return res_out

  def comb_layers(self, h_dyn, h_cont):
    comb1  = relu(conv2d(tf.concat(3,[h_dyn, h_cont]),
                        output_dim=self.gf_dim*4,k_h=3,k_w=3,d_h=1,d_w=1,name='comb1'))
    comb2  = relu(conv2d(comb1, output_dim=self.gf_dim*2,k_h=3,k_w=3,d_h=1,d_w=1,name='comb2'))
    h_comb = relu(conv2d(comb2, output_dim=self.gf_dim*4,k_h=3,k_w=3,d_h=1,d_w=1,name='h_comb'))

    return h_comb

  def decoder(self, h_comb, kernels, h_3, h_2, h_1, res_connect):
    shapel3   = [self.batch_size, self.image_size/4, self.image_size/4, self.gf_dim*4]
    shapeout3 = [self.batch_size, self.image_size/4, self.image_size/4, self.gf_dim*2]
    depool3   = FixedUnPooling(h_comb, [2,2])
    deconv3_3 = relu(cross_deconv2d(tf.concat(3,[relu(tf.add(depool3,res_connect[2])),h_3]),
                              output_shape=shapel3,kernels[0],k_h=3,k_w=3,d_h=1,d_w=1,name='deconv3_3'))
    deconv3_2 = relu(deconv2d(deconv3_3,output_shape=shapel3,k_h=3,k_w=3,d_h=1,d_w=1,name='deconv3_2'))
    deconv3_1 = relu(deconv2d(deconv3_2,output_shape=shapeout3,k_h=3,k_w=3,d_h=1,d_w=1,name='deconv3_1'))

    shapel2   = [self.batch_size, self.image_size/2, self.image_size/2, self.gf_dim*2]
    shapeout3 = [self.batch_size, self.image_size/2, self.image_size/2, self.gf_dim]
    depool2   = FixedUnPooling(deconv3_1, [2,2])
    deconv2_2 = relu(cross_deconv2d(tf.concat(3,[relu(tf.add(depool2,res_connect[1])),h_2]),
                              output_shape=shapel2,kernels[1],k_h=3,k_w=3,d_h=1,d_w=1,name='deconv2_2'))
    deconv2_1 = relu(deconv2d(deconv2_2,output_shape=shapeout3,k_h=3,k_w=3,d_h=1,d_w=1,name='deconv2_1'))

    shapel1   = [self.batch_size, self.image_size, self.image_size, self.gf_dim]
    shapeout1 = [self.batch_size, self.image_size, self.image_size, self.c_dim]
    depool1   = FixedUnPooling(deconv2_1, [2,2])
    deconv1_2 = relu(cross_deconv2d(tf.concat(3,[relu(tf.add(depool1,res_connect[0])),h_1]),
                     output_shape=shapel1,kernels[2],k_h=3,k_w=3,d_h=1,d_w=1,name='deconv1_2'))
    xtp1      = deconv2d(deconv1_2,output_shape=shapeout1,k_h=3,k_w=3,d_h=1,d_w=1,name='deconv1_1')

    return xtp1

  def save(self, sess, checkpoint_dir, step):
    model_name = "CONDNET.model"

    if not os.path.exists(checkpoint_dir):
      os.makedirs(checkpoint_dir)

    self.saver.save(sess,
                    os.path.join(checkpoint_dir, model_name),
                    global_step=step)

  def load(self, sess, checkpoint_dir):
    print(" [*] Reading checkpoints...")

    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
      self.saver.restore(sess, os.path.join(checkpoint_dir, ckpt_name))
      return True
    else:
      return False
