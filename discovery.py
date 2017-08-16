from main import *
import numpy as np

vgg_path = './data/vgg'

log_dir = './TFlog'
# graph = tf.get_default_graph()
# tf.summary.FileWriter('./TFlog',graph)
# tensorboard --logdir ./TFlog

inp =np.random.randn(1, 1242, 375, 3)
#print(inp.shape)

with tf.Session() as sess:
    input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess,vgg_path)
    num_classes = 2
    layers_output = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
    graph = tf.get_default_graph()
    sess.run(tf.global_variables_initializer())
    target=[#layers_output,
            graph.get_tensor_by_name('pool1:0'),
            graph.get_tensor_by_name('pool2:0'),
            graph.get_tensor_by_name('pool3:0'),
            graph.get_tensor_by_name('pool4:0'),
            graph.get_tensor_by_name('pool5:0'),
            graph.get_tensor_by_name('conv_t1/Relu:0'),
            graph.get_tensor_by_name('conv_t2/Relu:0'),
            vgg_layer7_out]
    out = sess.run(target,feed_dict={input_image:inp, keep_prob:1.0})
    for tensor in out:
        print(tensor.shape)

    tf.summary.FileWriter('./TFlog', graph)