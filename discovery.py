from main import *

vgg_path = './data/vgg'

log_dir = './TFlog'
# graph = tf.get_default_graph()
# tf.summary.FileWriter('./TFlog',graph)
# tensorboard --logdir ./TFlog

with tf.Session() as sess:
    input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess,vgg_path)
    num_classes = 2
    layers_output = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
    graph = tf.get_default_graph()
    tf.summary.FileWriter('./TFlog', graph)
