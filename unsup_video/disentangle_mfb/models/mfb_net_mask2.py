
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
        mt_feat  = c3d_feat[:, :, :, 340:]        # motion feature
        fg_feat  = c3d_feat[:, :, :, 170:340]     # foreground feature
        bg_feat  = c3d_feat[:, :, :, :170]        # background feature

        # reconstruction
        self.first_fg_rec  = self.decoder2d(fg_feat, name='first_fg_dec')
        self.first_bg_rec  = self.decoder2d(bg_feat, name='first_bg_dec')
        self.first_fg_mask, self.first_fg_mask_logits = self.mask_dec( \
            tf.concat([fg_feat, bg_feat], axis=3), name='first_fg_mask_dec')
        self.first_bg_mask = 1 - self.first_fg_mask

        self.last_fg_rec   = self.decoder2d(tf.concat([mt_feat, fg_feat], axis=3), name='last_fg_dec')
        self.last_fg_mask, self.last_fg_mask_logits  = self.mask_dec( \
            tf.concat([fg_feat, mt_feat], axis=3), name='last_fg_mask_dec')


    def reconstruct(self):
        return self.first_fg_rec, self.first_bg_rec, self.last_fg_rec


    def masks(self):
        return self.first_fg_mask, self.first_bg_mask, self.last_fg_mask


    def masks_logits(self):
        return self.first_fg_mask_logits, self.last_fg_mask_logits


    def bn(self, x):
        #return x
        return tf.contrib.layers.batch_norm(x, **self.batch_norm_params)


    def mapping_layer(self, input_, name='mapping'):
        with tf.variable_scope(name):
            feat = relu(self.bn(conv3d(input_, self.map_dim, k_t=self.map_length, k_h=2, k_w=2, d_h=2, d_w=2, padding='VALID', name='mapping1')))
            feat = tf.reshape(feat, [self.batch_size, self.map_height//2, self.map_width//2, self.map_dim])

        return feat


    def mask_dec(self, input_, name='mask_dec'):

        with tf.variable_scope(name):
            # mapping layer

            mask   = tf.reshape(input_, [self.batch_size, -1])
            #mask   = relu(fc(mask, 4096, name='fc1'))
            mask   = tf.reshape(fc(mask, 32*32, name='fc2'), [self.batch_size, 32, 32, 1])
            #logits = FixedUnPooling(mask, [4,4], unpool_mat=tf.ones([4,4]))
            logits = tf.image.resize_bilinear(mask, [128, 128])

            #mask   = deconv2d(mask,
            #                output_shape=[self.batch_size,64,64,1],
            #                    k_h=5,k_w=5,d_h=2,d_w=2,padding='SAME',name='deconv1')
            #mask   = tf.sigmoid(deconv2d(mask,
            #                output_shape=[self.batch_size,128,128,1],
            #                    k_h=5,k_w=5,d_h=2,d_w=2,padding='SAME',name='deconv2'))
            mask = tf.sigmoid(logits)
            logits = tf.reshape(logits, [self.batch_size, 128, 128])


        return mask, logits


    def decoder2d(self, input_, name='decoder2d'):
        # mirror decoder of c3d but 2d version
        with tf.variable_scope(name):

            # project 3d - 2d
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
                            output_shape=[self.batch_size,self.map_height*16,self.map_width*16,self.c_dim],
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

            conv5_shape     = conv5.get_shape().as_list()
            self.map_length = conv5_shape[1]
            self.map_height = conv5_shape[2]
            self.map_width  = conv5_shape[3]
            self.map_dim    = conv5_shape[4]

            feature = conv5

        return feature

    