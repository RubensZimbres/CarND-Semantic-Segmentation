# from main import *
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import numpy as np

vgg_path = './data/vgg'

log_dir = './TFlog'
# graph = tf.get_default_graph()
# tf.summary.FileWriter('./TFlog',graph)
# tensorboard --logdir ./TFlog

inp =np.random.randn(1, 1242, 375, 3)
num_classes = 2
learning_rate = 0.01
correct_label = tf.placeholder(tf.float32, [None, None, None, num_classes])
#inp =np.random.randn(1, 1024, 256, 3)
#print(inp.shape)
a = False
if(a):
    with tf.Session() as sess:
        ############### load the model
        import_scope_name = 'VGG16/'
        vgg_tag = 'vgg16'
        vgg_input_tensor_name = 'image_input:0'
        vgg_keep_prob_tensor_name = 'keep_prob:0'
        vgg_layer3_out_tensor_name = 'layer3_out:0'
        vgg_layer4_out_tensor_name = 'layer4_out:0'
        vgg_layer7_out_tensor_name = 'layer7_out:0'

        tf.saved_model.loader.load(sess, [vgg_tag], vgg_path,import_scope='VGG16')


        graph = tf.get_default_graph()
        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out= \
            graph.get_tensor_by_name(import_scope_name+vgg_input_tensor_name), \
            graph.get_tensor_by_name(import_scope_name+vgg_keep_prob_tensor_name), \
            graph.get_tensor_by_name(import_scope_name+vgg_layer3_out_tensor_name), \
            graph.get_tensor_by_name(import_scope_name+vgg_layer4_out_tensor_name), \
            graph.get_tensor_by_name(import_scope_name+vgg_layer7_out_tensor_name)

        with tf.variable_scope('MyExtension'):
            shape_input = tf.shape(input_image, name='input_shape')
            shape_vgg_l4_out = tf.shape(vgg_layer4_out, name='vgg_l4_out_shape')
            shape_vgg_l3_out = tf.shape(vgg_layer3_out, name='vgg_l3_out_shape')
            ################### layers
            conv_t1 = tf.layers.conv2d_transpose(vgg_layer7_out, 512, (3, 3), strides=(2, 2), padding='same',
                                                 activation=tf.nn.relu, name="conv_t1")
            before_skip_1 = tf.slice(conv_t1, [0, 0, 0, 0], shape_vgg_l4_out, name='crop_1')
            conv_t1_skip = tf.add(before_skip_1, vgg_layer4_out, name='skip_layer_4')
            conv_t2 = tf.layers.conv2d_transpose(conv_t1_skip, 256, (3, 3), strides=(2, 2), padding='same',
                                                 activation=tf.nn.relu, name="conv_t2")
            before_skip_2 = tf.slice(conv_t2, [0, 0, 0, 0], shape_vgg_l3_out, name='crop_2')
            conv_t2_skip = tf.add(before_skip_2, vgg_layer3_out, name='skip_layer_3')
            conv_t3 = tf.layers.conv2d_transpose(conv_t2_skip, 256, (3, 3), strides=(2, 2), padding='same',
                                                 activation=tf.nn.relu, name="conv_t3")
            conv_t4 = tf.layers.conv2d_transpose(conv_t3, 128, (3, 3), strides=(2, 2), padding='same',
                                                 activation=tf.nn.relu, name='conv_t4')
            conv_t5 = tf.layers.conv2d_transpose(conv_t4, 64, (3, 3), strides=(2, 2), padding='same', activation=tf.nn.relu,
                                                 name='conv_t5')
            conv_last = tf.layers.conv2d(conv_t5, num_classes, (15, 15), strides=(1, 1), padding='same', activation=tf.nn.relu,
                                         name='conv_last')
            final_crop = tf.slice(conv_last,[0,0,0,0],[shape_input[0],shape_input[1],shape_input[2],2],name='final_crop')


        # crossEntropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
        # crossEntropy = tf.reduce_mean(crossEntropy)
        # loss = tf.reduce_mean(tf.multiply(crossEntropy, weights))
        # tf.add_to_collection('my_losses', tf.multiply(.1, loss))
        #
        # imu, imuOp = tf.metrics.mean_iou(labels, tf.argmax(logits, axis=2), numberOfClasses, weights, name= ‘meanIMU’)
        # with tf.control_dependencies([imuOp]):
        #     imu = tf.subtract(tf.constant(1.), imu)
        #     tf.add_to_collection('my_losses', imu)
        #     loss = tf.reduce_sum(tf.stack(tf.get_collection('my_losses')))
        #     trainStep = tf.train.AdamOptimizer(5e4).minimize(loss)

        ##################### check the sizes

        # sess.run(tf.global_variables_initializer())
        # target=[final_crop,
        #         graph.get_tensor_by_name('pool1:0'),
        #         graph.get_tensor_by_name('pool2:0'),
        #         graph.get_tensor_by_name('pool3:0'),
        #         graph.get_tensor_by_name('pool4:0'),
        #         graph.get_tensor_by_name('pool5:0'),
        #         conv_t1,
        #         conv_t2,
        #         vgg_layer7_out,
        #         shape_vgg_l4_out,
        #         shape_vgg_l3_out]
        # out = sess.run(target,feed_dict={input_image:inp, keep_prob:1.0})
        # for tensor in out:
        #     print(tensor.shape)
        #
        # print(out[-1])
        # print(out[-2])

        ############### optimizer
        updatable_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='MyExtension')
        print (updatable_variables)
        logits = tf.reshape(final_crop, (-1, num_classes),name='logits')
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits,labels=correct_label)
                                            ,name='crossEntropyLoss')
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate,name='optimizer')
        training_operation = optimizer.minimize(cross_entropy_loss,var_list=updatable_variables)
        print('trainable variables: ', tf.trainable_variables())

        tf.summary.FileWriter('./TFlog', graph)

if not a:
    n=3
    inp = np.array(range(n*n*2),dtype=np.float64)
    inp = inp.reshape((1,n,n,2))
    #print (inp.reshape(n,n,2))
    print(inp)
    inPlace = tf.placeholder(tf.float64,shape=[None,None,None,2],name='in')
    #out = tf.layers.max_pooling2d(inPlace,2,2,padding='same',name='pool1')
    out = tf.argmax(inPlace, axis = 3)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        res, inP = sess.run([out,inPlace],feed_dict={inPlace:inp})
        print(res.shape)
        print(res)
        #print(inP[0,0,0,0])