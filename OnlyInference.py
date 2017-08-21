import tensorflow as tf
from glob import glob
#import random
import numpy as np
import os.path
import scipy.misc



def load_graph(graph_file, use_xla=False):
    jit_level = 0
    config = tf.ConfigProto()
    if use_xla:
        jit_level = tf.OptimizerOptions.ON_1
        config.graph_options.optimizer_options.global_jit_level = jit_level

    with tf.Session(graph=tf.Graph(), config=config) as sess:
        gd = tf.GraphDef()
        with tf.gfile.Open(graph_file, 'rb') as f:
            data = f.read()
            gd.ParseFromString(data)
        tf.import_graph_def(gd, name='')
        ops = sess.graph.get_operations()
        n_ops = len(ops)
        return sess, ops #sess.graph

sess, _ = load_graph('./model/eightbit_graph.pb')
graph = sess.graph

image_input = graph.get_tensor_by_name('image_input:0')
keep_prob = graph.get_tensor_by_name('keep_prob:0')
output = graph.get_tensor_by_name('labeled_output:0')

runs_dir = './run2'
data_dir = './data/data_road/testing/image_2'
image_shape = (160, 576)

for image_file in glob(os.path.join(data_dir, '*.png')):
    image = scipy.misc.imresize(scipy.misc.imread(image_file), image_shape)
    output_img = sess.run(output,{keep_prob: 1.0, image_input: [image]})
    sq_output_img = np.squeeze(output_img,axis=0)
    sh = sq_output_img.shape
    mask = np.zeros(shape=(sh[0],sh[1],4))
    mask[sq_output_img==1,:] = [0, 255, 0, 127]
    mask = scipy.misc.toimage(mask, mode="RGBA")
    street_im = scipy.misc.toimage(image)
    street_im.paste(mask, box=None, mask=mask)
    scipy.misc.imsave(os.path.join(runs_dir, os.path.basename(image_file)), np.array(street_im))