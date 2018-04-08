
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

        # disentangle c3d features into 3 categories
        #mt_feat = c3d_feat[:, :1365]        # motion feature
        #fg_feat = c3d_feat[:, 1365:2730]    # foreground feature
        #bg_feat = c3d_feat[:, 2730:]        # background feature
        mt_feat = c3d_feat[:, :, :, :, 170:]        # motion feature
        fg_feat = c3d_feat[:, :, :, :, 85:170]     # foreground feature
        bg_feat = c3d_feat[:, :, :, :, :85]        # background feature

        # reconstruction
        self.first_fg_rec = self.decoder2d(fg_feat, name='first_fg_dec')
        self.first_bg_rec = self.decoder2d(bg_feat, name='first_bg_dec')
        self.last_fg_rec  = self.decoder2d(tf.concat([mt_feat, fg_feat], axis=4), name='last_fg_dec')


    def reconstruct(self):
        return self.first_fg_rec, self.first_bg_rec, self.last_fg_rec


    def decoder2d(self, input_, name='decoder2d'):
        # mirror decoder of c3d but 2d version
        with tf.variable_scope(name):
            """
            fc1  = fc(input_, 4096, 'fc1')
            fc2  = fc(fc1, self.c_dim*self.map_height*self.map_width*self.map_length, 'fc2')
            feat = tf.reshape(fc2, [self.batch_size, self.c_dim, self.map_height, self.map_width])
            feat = tf.transpose(feat, perm=[0,2,3,1])

            depool5  = FixedUnPooling(feat, [2,2])
            """

            # project 3d - 2d
            depool2  = relu(conv3d(input_, self.map_dim, k_d=self.map_length, k_h=1, k_w=1, padding='VALID', name='mapping'))
            depool2  = tf.reshape(depool2, [-1, self.map_height, self.map_width, self.map_dim])
            deconv2  = relu(deconv2d(depool2,
                            output_shape=[self.batch_size,self.map_height,self.map_width,self.map_dim//2],
                                k_h=3,k_w=3,d_h=1,d_w=1,name='deconv2b'))

            depool1  = FixedUnPooling(deconv2, [2,2])
            deconv1  = relu(deconv2d(depool1,
                            output_shape=[self.batch_size,self.map_height*2,self.map_width*2,self.c_dim],
                                k_h=3,k_w=3,d_h=1,d_w=1,name='deconv1'))
            assert(self.height==self.map_height*16 and self.width==self.map_width*16)

            return deconv1


    def c3d(self, input_, _dropout=1.0, name='c3d'):

        with tf.variable_scope(name):

            # Convolution Layer
            conv1 = relu(conv3d(input_, 64, name='conv1'))
            pool1 = max_pool3d(conv1, k=1, name='pool1')

            # Convolution Layer
            conv2 = relu(conv3d(pool1, 128, name='conv2'))

            conv2_shape     = conv2.get_shape().as_list()
            self.map_length = conv2_shape[1]
            self.map_height = conv2_shape[2]
            self.map_width  = conv2_shape[3]
            self.map_dim    = conv2_shape[4]
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

            feature = conv2

        return feature

    