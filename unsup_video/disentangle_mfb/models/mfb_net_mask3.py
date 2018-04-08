
import tensorflow as tf

from tools.ops import *

class mfb_net(object):

    def __init__(self, input_, height=128, width=128, seq_length=16, c_dim=3, batch_size=32):

        self.seq        = input_
        self.batch_size = batch_size
        self.height     = height
        self.width      = width
        self.seq_length = seq_length
        self.c_dim      = c_dim

        self.seq_shape  = [seq_length, height, width, c_dim]

        self.build_model()


    def build_model(self):

        c3d_feat = self.c3d(self.seq)
        mt_feat = c3d_feat[:, :, :, :, 340:]        # motion feature
        fg_feat = c3d_feat[:, :, :, :, 170:340]     # foreground feature
        bg_feat = c3d_feat[:, :, :, :, :170]        # background feature

        # reconstruction
        self.first_fg_rec, first_fg_feat = self.decoder2d(fg_feat, name='first_fg_dec')
        self.first_bg_rec, _             = self.decoder2d(bg_feat, name='first_bg_dec')
        self.last_fg_rec,  last_fg_feat  = self.decoder2d( \
                            tf.concat([mt_feat, fg_feat], axis=4), name='last_fg_dec')

        self.first_fg_mask, self.first_fg_mask_logits = \
                            self.mask_dec(first_fg_feat, name='first_fg_mask_dec')
        self.last_fg_mask, self.last_fg_mask_logits = \
                            self.mask_dec(last_fg_feat, name='last_fg_mask_dec')

        self.first_fg_bbx = self.bbx_dec(tf.concat([fg_feat, bg_feat], axis=4), name='ffg_bbx_dec')
        self.last_fg_bbx  = self.bbx_dec(tf.concat([fg_feat, mt_feat], axis=4), name='lfg_bbx_dec')


    def reconstruct(self):
        return self.first_fg_rec, self.first_bg_rec, self.last_fg_rec


    def masks(self):
        return self.first_fg_mask, self.last_fg_mask


    def masks_logits(self):
        return self.first_fg_mask_logits, self.last_fg_mask_logits


    def bbx(self):
        return self.first_fg_bbx, self.last_fg_bbx


    def mask_dec(self, input_, name='mask_dec'):

        with tf.variable_scope(name):
            # mapping layer        

            deconv2 = relu(deconv2d(input_,
                            output_shape=[self.batch_size,self.map_height*8,self.map_width*8,self.map_dim//4],
                                k_h=5,k_w=5,d_h=2,d_w=2,padding='SAME',name='deconv2')) # 64x64
            deconv1 = deconv2d(deconv2,
                            output_shape=[self.batch_size,self.map_height*16,self.map_width*16,1],
                                k_h=5,k_w=5,d_h=2,d_w=2,padding='SAME',name='deconv1') #128x128

            mask, logits = tf.sigmoid(deconv1), deconv1

        return mask, logits


    def bbx_dec(self, input_, name='bbx_dec'):

        with tf.variable_scope(name):

            input_ = tf.reshape(input_, [self.batch_size, -1])
            bbx = relu(fc(input_, 1024, name='fc1'))
            bbx = fc(bbx, 4, name='fc2')

        return bbx


    def decoder2d(self, input_, name='decoder2d'):
        # mirror decoder of c3d but 2d version
        with tf.variable_scope(name):

            # project 3d - 2d

            feat = relu(conv3d(input_, self.map_dim, k_t=self.map_length, k_h=1, k_w=1, padding='VALID', name='mapping1'))
            feat = tf.reshape(feat, [-1, self.map_height, self.map_width, self.map_dim])
            feat = relu(conv2d(feat, self.map_dim*2, k_h=2, k_w=2, d_h=2, d_w=2, padding='VALID', name='mapping2')) # 4x4

            deconv1 = relu(deconv2d(feat,
                            output_shape=[self.batch_size,self.map_height,self.map_width,self.map_dim],
                                k_h=5,k_w=5,d_h=2,d_w=2,padding='SAME',name='deconv1')) # 8x8
            deconv2 = relu(deconv2d(deconv1,
                            output_shape=[self.batch_size,self.map_height*2,self.map_width*2,self.map_dim],
                                k_h=5,k_w=5,d_h=2,d_w=2,padding='SAME',name='deconv2')) # 16x16
            deconv3 = relu(deconv2d(deconv2,
                            output_shape=[self.batch_size,self.map_height*4,self.map_width*4,self.map_dim//2],
                                k_h=5,k_w=5,d_h=2,d_w=2,padding='SAME',name='deconv3')) # 32x32
            deconv4 = relu(deconv2d(deconv3,
                            output_shape=[self.batch_size,self.map_height*8,self.map_width*8,self.map_dim//4],
                                k_h=5,k_w=5,d_h=2,d_w=2,padding='SAME',name='deconv4')) # 64x64
            deconv5 = deconv2d(deconv4,
                            output_shape=[self.batch_size,self.map_height*16,self.map_width*16,self.c_dim],
                                k_h=5,k_w=5,d_h=2,d_w=2,padding='SAME',name='deconv5') #128x128

            img, feat = tf.tanh(deconv5), deconv3
            
            
            return img, feat


    def c3d(self, input_, _dropout=1.0, name='c3d'):

        with tf.variable_scope(name):

            # Convolution Layer
            conv1 = relu(conv3d(input_, 64, name='conv1'))
            pool1 = max_pool3d(conv1, k=1, name='pool1')

            # Convolution Layer
            conv2 = relu(conv3d(pool1, 128, name='conv2'))
            pool2 = max_pool3d(conv2, k=2, name='pool2')

            # Convolution Layer
            conv3 = relu(conv3d(pool2, 256, name='conv3a'))
            conv3 = relu(conv3d(conv3, 256, name='conv3b'))
            pool3 = max_pool3d(conv3, k=2, name='pool3')

            # Convolution Layer
            conv4 = relu(conv3d(pool3, 512, name='conv4a'))
            conv4 = relu(conv3d(conv4, 512, name='conv4b'))
            pool4 = max_pool3d(conv4, k=2, name='pool4')

            # Convolution Layer
            conv5 = relu(conv3d(pool4, 512, name='conv5a'))
            conv5 = relu(conv3d(conv5, 512, name='conv5b'))

            conv5_shape     = conv5.get_shape().as_list()
            self.map_length = conv5_shape[1]
            self.map_height = conv5_shape[2]
            self.map_width  = conv5_shape[3]
            self.map_dim    = conv5_shape[4]

            feature = conv5

        return feature

    