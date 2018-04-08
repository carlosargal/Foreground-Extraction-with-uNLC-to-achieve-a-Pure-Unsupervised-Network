
import tensorflow as tf

from tools.ops import *

class mfb_net(object):

    def __init__(self, input_, height=128, width=128, seq_length=16, c_dim=3, batch_size=32, is_training=True):

        self.seq        = input_
        self.batch_size = batch_size
        self.height     = height
        self.width      = width
        self.seq_length = seq_length
        self.c_dim      = c_dim
        self.gt_ffg     = None
        self.gt_fbg     = None
        self.gt_lfg     = None
        self.flags      = tf.app.flags.FLAGS

        self.seq_shape  = [seq_length, height, width, c_dim]

        self.batch_norm_params = {
            'is_training': is_training,
            'decay': 0.9,
            'epsilon': 1e-5,
            'scale': True,
            'center': True,
            'updates_collections': tf.GraphKeys.UPDATE_OPS
        }

        self.build_model()


    def build_model(self):

        c3d_feat = self.mapping_layer(self.c3d(self.seq))

        mt_feat = c3d_feat[:, :, :, 340:]         # motion feature
        fg_feat = c3d_feat[:, :, :, 170:340]      # foreground feature
        bg_feat = c3d_feat[:, :, :, :170]         # background feature
        self.mt_feat, self.fg_feat, self.bg_feat = mt_feat, fg_feat, bg_feat

        # reconstruction
        self.first_fg_rec  = self.decoder2d(fg_feat, self.c_dim, name='fg_dec')
        self.first_bg_rec  = self.decoder2d(bg_feat, self.c_dim, name='bg_dec')
        #self.first_fg_mask = tf.sigmoid(self.decoder2d(tf.concat([fg_feat, bg_feat], axis=4), 1, name='first_fg_mask'))

        kernel             = self.kernel_decoder(mt_feat)
        fg_feat            = tf.stop_gradient(fg_feat)

        fg_feat = self.reduction_layer(fg_feat)
        self.last_fg_feat = cross_conv2d(fg_feat, kernel, d_h=1, d_w=1)
        self.last_fg_feat = self.restoring_layer(self.last_fg_feat)

        self.last_fg_rec   = self.decoder2d(self.last_fg_feat, self.c_dim, reuse=True, name='fg_dec')
        #self.last_fg_mask  = tf.sigmoid(self.decoder2d(tf.concat([mt_feat, fg_feat], axis=4), 1, name='last_fg_mask'))

        inv_c3d_feat = self.mapping_layer(self.c3d(self.seq[:,::-1], reuse=True), reuse=True)
        self.inv_last_fg_feat_gt = tf.stop_gradient(inv_c3d_feat[:, :, :, 170:340])


    def reconstruct(self):
        return self.first_fg_rec, self.first_bg_rec, self.last_fg_rec


    def masks(self):
        return self.first_fg_mask, self.last_fg_mask


    def bn(self, x):
        return tf.contrib.layers.batch_norm(x, **self.batch_norm_params)


    def mapping_layer(self, input_, reuse=False, name='mapping'):
        with tf.variable_scope(name, reuse=reuse):
            feat = relu(self.bn(conv3d(input_, self.map_dim, k_t=self.map_length, k_h=2, k_w=2, d_h=2, d_w=2, padding='VALID', name='mapping1')))
            feat = tf.reshape(feat, [self.batch_size, self.map_height//2, self.map_width//2, self.map_dim])

        return feat


    def reduction_layer(self, input_, reuse=False, name='reduction'):
        with tf.variable_scope(name, reuse=reuse):
            input_ = relu(self.bn(conv2d(input_, self.fg_feat_c_dim, k_h=1, k_w=1, \
                           d_h=1, d_w=1, padding='SAME', name='reduction')))
        return input_


    def restoring_layer(self, input_, reuse=False, name='restoring'):
        with tf.variable_scope(name, reuse=reuse):
            input_ = relu(self.bn(conv2d(input_, 170, k_h=1, k_w=1, \
                           d_h=1, d_w=1, padding='SAME', name='restoring')))
        return input_


    def kernel_decoder(self, input_, reuse=False, name='kernel_dec'):
        with tf.variable_scope(name, reuse=reuse):

            kernel_out_dim     = 48
            self.fg_feat_c_dim = 48
            kernel_in_dim      = self.fg_feat_c_dim

            feat = relu(self.bn(conv2d(input_, kernel_in_dim*kernel_out_dim, k_h=3, k_w=3, \
                                                d_h=1, d_w=1, padding='SAME', name='mapping1')))

            kernel = feat
            kernel = tf.reshape(kernel, [self.batch_size, 4, 4, kernel_in_dim*kernel_out_dim])

            kernel = relu(self.bn(conv2d(kernel, kernel_in_dim*kernel_out_dim, k_h=3, k_w=3, d_h=1, d_w=1, padding='SAME', name='conv1')))
            kernel = conv2d(kernel, kernel_in_dim*kernel_out_dim, k_h=3, k_w=3, d_h=1, d_w=1, padding='SAME', name='conv2')
            kernel = tf.reshape(kernel, [self.batch_size, 4, 4, kernel_in_dim, kernel_out_dim])                     


        return kernel


    def decoder2d(self, input_, out_dim, kernels=None, reuse=False, name='decoder2d'):
        # mirror decoder of c3d but 2d version
        with tf.variable_scope(name, reuse=reuse):

            feat = relu(self.bn(conv2d(input_, self.map_dim*2, k_h=1, k_w=1, padding='SAME', name='mapping'))) # 4x4

            deconv1 = relu(self.bn(deconv2d(feat,
                            output_shape=[self.batch_size,self.map_height,self.map_width,self.map_dim],
                                k_h=5,k_w=5,d_h=2,d_w=2,padding='SAME',name='deconv1'))) # 8x8

            deconv2 = relu(self.bn(deconv2d(deconv1,
                            output_shape=[self.batch_size,self.map_height*2,self.map_width*2,self.map_dim],
                                k_h=5,k_w=5,d_h=2,d_w=2,padding='SAME',name='deconv2'))) # 16x16

            deconv3 = relu(self.bn(deconv2d(deconv2,
                            output_shape=[self.batch_size,self.map_height*4,self.map_width*4,self.map_dim//2],
                                k_h=5,k_w=5,d_h=2,d_w=2,padding='SAME',name='deconv3'))) # 32x32

            deconv4 = relu(self.bn(deconv2d(deconv3,
                            output_shape=[self.batch_size,self.map_height*8,self.map_width*8,self.map_dim//4],
                                k_h=5,k_w=5,d_h=2,d_w=2,padding='SAME',name='deconv4'))) # 64x64

            deconv5 = deconv2d(deconv4,
                            output_shape=[self.batch_size,self.map_height*16,self.map_width*16,self.c_dim],
                                k_h=5,k_w=5,d_h=2,d_w=2,padding='SAME',name='deconv5') #128x128

            img = tf.tanh(deconv5)
            

            return img


    def c3d(self, input_, reuse=False, _dropout=1.0, name='c3d'):

        with tf.variable_scope(name, reuse=reuse):

            # Convolution Layer
            conv1 = relu(self.bn(conv3d(input_, 64, name='conv1')))
            pool1 = max_pool3d(conv1, k=1, name='pool1')

            # Convolution Layer
            conv2 = relu(self.bn(conv3d(pool1, 128, name='conv2')))
            pool2 = max_pool3d(conv2, k=2, name='pool2')

            # Convolution Layer
            conv3 = relu(self.bn(conv3d(pool2, 256, name='conv3a')))
            conv3 = relu(self.bn(conv3d(conv3, 256, name='conv3b')))
            pool3 = max_pool3d(conv3, k=2, name='pool3')

            # Convolution Layer
            conv4 = relu(self.bn(conv3d(pool3, 512, name='conv4a')))
            conv4 = relu(self.bn(conv3d(conv4, 512, name='conv4b')))
            pool4 = max_pool3d(conv4, k=2, name='pool4')

            # Convolution Layer
            conv5 = relu(self.bn(conv3d(pool4, 512, name='conv5a')))
            conv5 = relu(self.bn(conv3d(conv5, 512, name='conv5b')))
            #pool5 = max_pool3d(conv5, k=2, name='pool5')

            conv5_shape     = conv5.get_shape().as_list()
            self.map_length = conv5_shape[1]
            self.map_height = conv5_shape[2]
            self.map_width  = conv5_shape[3]
            self.map_dim    = conv5_shape[4]
            """
            # record map size after pool5
            pool5_shape     = pool5.get_shape().as_list()
            self.map_length = int(pool5_shape[1])
            self.map_height = int(pool5_shape[2])
            self.map_width  = int(pool5_shape[3])
            self.map_dim    = int(pool5_shape[4])

            # Fully connected layer
            # Before transpose: [batch_size, seq_length, h, w, c]
            # After  transpose: [batch_size, seq_length, c, h, w]
            pool5  = tf.transpose(pool5, perm=[0,1,4,2,3])
            dense1 = tf.reshape(pool5, [self.batch_size, -1])

            dense1 = relu(fc(dense1, 4096, name='fc1')) # Relu activation
            dense1 = tf.nn.dropout(dense1, _dropout)

            dense2 = relu(fc(dense1, 4096, name='fc2')) # Relu activation
            dense2 = tf.nn.dropout(dense2, _dropout)
            """

            feature = conv5

        return feature

    
