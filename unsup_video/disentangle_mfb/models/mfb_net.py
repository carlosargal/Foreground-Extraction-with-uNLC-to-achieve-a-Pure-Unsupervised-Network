
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

        # disentangle c3d features into 3 categories
        #mt_feat = c3d_feat[:, :1365]        # motion feature
        #fg_feat = c3d_feat[:, 1365:2730]    # foreground feature
        #bg_feat = c3d_feat[:, 2730:]        # background feature
        self.mt_feat = c3d_feat[:, :, :, 340:]         # motion feature
        self.fg_feat = c3d_feat[:, :, :, 170:340]      # foreground feature
        self.bg_feat = c3d_feat[:, :, :, :170]         # background feature

        # reconstruction
        self.first_fg_rec  = self.decoder2d(self.fg_feat, self.c_dim, name='first_fg_dec')
        self.first_bg_rec  = self.decoder2d(self.bg_feat, self.c_dim, name='first_bg_dec')
        #self.first_fg_mask = tf.sigmoid(self.decoder2d(tf.concat([fg_feat, bg_feat], axis=4), 1, name='first_fg_mask'))

        self.last_fg_rec   = self.decoder2d(tf.concat([self.mt_feat, self.fg_feat], axis=3), self.c_dim, name='last_fg_dec')
        #self.last_fg_mask  = tf.sigmoid(self.decoder2d(tf.concat([mt_feat, fg_feat], axis=4), 1, name='last_fg_mask'))


    def reconstruct(self):
        return self.first_fg_rec, self.first_bg_rec, self.last_fg_rec


    def masks(self):
        return self.first_fg_mask, self.last_fg_mask


    def bn(self, x):
        return tf.contrib.layers.batch_norm(x, **self.batch_norm_params)


    def mapping_layer(self, input_, name='mapping'):
        with tf.variable_scope(name):
            feat = relu(self.bn(conv3d(input_, self.map_dim, k_t=self.map_length, k_h=2, k_w=2, d_h=2, d_w=2, padding='VALID', name='mapping1')))
            feat = tf.reshape(feat, [self.batch_size, self.map_height//2, self.map_width//2, self.map_dim])

        return feat


    def decoder2d(self, input_, out_dim, name='decoder2d'):
        # mirror decoder of c3d but 2d version
        with tf.variable_scope(name):
            """
            fc1  = fc(input_, 4096, 'fc1')
            fc2  = fc(fc1, self.c_dim*self.map_height*self.map_width*self.map_length, 'fc2')
            feat = tf.reshape(fc2, [self.batch_size, self.c_dim, self.map_height, self.map_width])
            feat = tf.transpose(feat, perm=[0,2,3,1])

            depool5  = FixedUnPooling(feat, [2,2])
            """

            # project 3d - 2d:
            """           
            depool5  = relu(conv3d(input_, self.map_dim, k_t=self.map_length, k_h=1, k_w=1, padding='VALID', name='mapping'))
            depool5  = tf.reshape(depool5, [-1, self.map_height, self.map_width, self.map_dim])
            deconv5b = relu(deconv2d(depool5,
                            output_shape=[self.batch_size,self.map_height,self.map_width,self.map_dim],
                                k_h=3,k_w=3,d_h=1,d_w=1,name='deconv5b'))
            deconv5a = relu(deconv2d(deconv5b,
                            output_shape=[self.batch_size,self.map_height,self.map_width,self.map_dim],
                                k_h=3,k_w=3,d_h=1,d_w=1,name='deconv5a'))

            depool4  = FixedUnPooling(deconv5a, [2,2])
            deconv4b = relu(deconv2d(depool4,
                            output_shape=[self.batch_size,self.map_height*2,self.map_width*2,self.map_dim],
                                k_h=3,k_w=3,d_h=1,d_w=1,name='deconv4b'))
            deconv4a = relu(deconv2d(deconv4b,
                            output_shape=[self.batch_size,self.map_height*2,self.map_width*2,self.map_dim//2],
                                k_h=3,k_w=3,d_h=1,d_w=1,name='deconv4a'))

            depool3  = FixedUnPooling(deconv4a, [2,2])
            deconv3b = relu(deconv2d(depool3,
                            output_shape=[self.batch_size,self.map_height*4,self.map_width*4,self.map_dim//2],
                                k_h=3,k_w=3,d_h=1,d_w=1,name='deconv3b'))
            deconv3a = relu(deconv2d(deconv3b,
                            output_shape=[self.batch_size,self.map_height*4,self.map_width*4,self.map_dim//4],
                                k_h=3,k_w=3,d_h=1,d_w=1,name='deconv3a'))

            depool2  = FixedUnPooling(deconv3a, [2,2])
            deconv2  = relu(deconv2d(depool2,
                            output_shape=[self.batch_size,self.map_height*8,self.map_width*8,self.map_dim//8],
                                k_h=3,k_w=3,d_h=1,d_w=1,name='deconv2b'))

            depool1  = FixedUnPooling(deconv2, [2,2])
            deconv1  = deconv2d(depool1,
                            output_shape=[self.batch_size,self.map_height*16,self.map_width*16,out_dim],
                                k_h=3,k_w=3,d_h=1,d_w=1,name='deconv1')
            assert(self.height==self.map_height*16 and self.width==self.map_width*16)

            img = tf.tanh(deconv1)
            """

            feat = relu(self.bn(conv2d(input_, self.map_dim*2, k_h=2, k_w=2, padding='SAME', name='mapping'))) # 4x4

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


    def c3d(self, input_, _dropout=1.0, name='c3d'):

        with tf.variable_scope(name):

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





class mfb_dis_net(object):

    def __init__(self, clips, labels, class_num=24, height=128, width=128, seq_length=16, c_dim=3, \
                 batch_size=32, keep_prob=1.0, is_training=True, encoder_gradient_ratio=1.0, use_pretrained_encoder=False):

        self.seq        = clips
        self.labels     = labels
        self.class_num  = class_num
        self.batch_size = batch_size
        self.height     = height
        self.width      = width
        self.seq_length = seq_length
        self.c_dim      = c_dim
        self.dropout    = keep_prob
        self.encoder_gradient_ratio = encoder_gradient_ratio
        self.use_pretrained_encoder = use_pretrained_encoder

        self.seq_shape  = [seq_length, height, width, c_dim]

        self.batch_norm_params = {
            'is_training': is_training,
            'decay': 0.9,
            'epsilon': 1e-5,
            'scale': True,
            'center': True,
            'updates_collections': tf.GraphKeys.UPDATE_OPS
        }

        pred_logits  = self.build_model()
        self.ac_loss = tf.reduce_mean(\
                        tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=pred_logits))

        prob = tf.nn.softmax(pred_logits)
        pred = tf.one_hot(tf.nn.top_k(prob).indices, self.class_num)
        pred = tf.squeeze(pred, axis=1)
        pred = tf.cast(pred, tf.bool)
        labels  = tf.cast(labels, tf.bool)
        self.ac = tf.reduce_sum(tf.cast(tf.logical_and(labels, pred), tf.float32)) / self.batch_size


    def build_model(self):

        c3d_feat = self.mapping_layer(self.c3d(self.seq))

        if self.use_pretrained_encoder and self.encoder_gradient_ratio == 0.0:
            c3d_feat = tf.stop_gradient(c3d_feat)

        with tf.variable_scope('classifier'):
            dense1 = tf.reshape(c3d_feat, [self.batch_size, -1])

            dense1 = fc(dense1, self.class_num, name='fc1') # Relu activation
            pred   = tf.nn.dropout(dense1, self.dropout)

        return pred


    def bn(self, x):
        return tf.contrib.layers.batch_norm(x, **self.batch_norm_params)


    def mapping_layer(self, input_, name='mapping'):
        with tf.variable_scope(name):
            feat = relu(self.bn(conv3d(input_, self.map_dim, k_t=self.map_length, k_h=2, k_w=2, d_h=2, d_w=2, padding='VALID', name='mapping1')))
            feat = tf.reshape(feat, [self.batch_size, self.map_height//2, self.map_width//2, self.map_dim])

        return feat


    def c3d(self, input_, _dropout=1.0, name='c3d'):

        with tf.variable_scope(name):

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

            feature = conv5

        return feature