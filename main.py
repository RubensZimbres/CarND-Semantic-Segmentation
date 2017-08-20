import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import numpy as np


# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    #import_scope_name = 'VGG16/'
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    tf.saved_model.loader.load(sess,[vgg_tag],vgg_path)
    
    return tf.get_default_graph().get_tensor_by_name(vgg_input_tensor_name),\
           tf.get_default_graph().get_tensor_by_name(vgg_keep_prob_tensor_name),\
           tf.get_default_graph().get_tensor_by_name(vgg_layer3_out_tensor_name),\
           tf.get_default_graph().get_tensor_by_name(vgg_layer4_out_tensor_name),\
           tf.get_default_graph().get_tensor_by_name(vgg_layer7_out_tensor_name)


#tests.test_load_vgg(load_vgg, tf)


def layers(input_image,vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    with tf.variable_scope('MyExtension'):
        ## required shapes
        shape_input = tf.shape(input_image, name='input_shape')
        shape_vgg_l4_out = tf.shape(vgg_layer4_out, name='vgg_l4_out_shape')
        shape_vgg_l3_out = tf.shape(vgg_layer3_out, name='vgg_l3_out_shape')
        ##
        conv_1x1 = tf.layers.conv2d(vgg_layer7_out,4096,(1,1),strides=(1,1),padding='same', activation=tf.nn.relu, name="conv_1x1")
        conv_t1 = tf.layers.conv2d_transpose(conv_1x1, 512, (3, 3), strides=(2, 2), padding='same',
                                             activation=tf.nn.relu, name="conv_t1")
        conv_t1_1 = tf.layers.conv2d(conv_t1,512,(3,3), padding='same',activation=tf.nn.relu,name="conv_t1_1")
        before_skip_1 = tf.slice(conv_t1_1, [0, 0, 0, 0], shape_vgg_l4_out, name='crop_1')
        conv_t1_skip = tf.add(before_skip_1, vgg_layer4_out, name='skip_layer_4')
        conv_t2 = tf.layers.conv2d_transpose(conv_t1_skip, 256, (3, 3), strides=(2, 2), padding='same',
                                             activation=tf.nn.relu, name="conv_t2")
        conv_t2_1 = tf.layers.conv2d(conv_t2, 256, (3, 3), padding='same',activation=tf.nn.tanh, name="conv_t2_1")
        before_skip_2 = tf.slice(conv_t2_1, [0, 0, 0, 0], shape_vgg_l3_out, name='crop_2')
        conv_t2_skip = tf.add(before_skip_2, vgg_layer3_out, name='skip_layer_3')
        conv_t3 = tf.layers.conv2d_transpose(conv_t2_skip, 256, (3, 3), strides=(2, 2), padding='same',
                                             activation=tf.nn.tanh, name="conv_t3")
        conv_t3_1 = tf.layers.conv2d(conv_t3, 256, (3, 3), padding='same', activation=tf.nn.relu,name='conv_t3_1')
        conv_t4 = tf.layers.conv2d_transpose(conv_t3_1, 128, (3, 3), strides=(2, 2), padding='same',
                                             activation=tf.nn.relu, name='conv_t4')
        conv_t4_1 = tf.layers.conv2d(conv_t4, 128, (3, 3), padding='same', activation=tf.nn.tanh,name='conv_t4_1')
        conv_t5 = tf.layers.conv2d_transpose(conv_t4_1, 64, (3, 3), strides=(2, 2), padding='same', activation=tf.nn.relu,
                                             name='conv_t5')
        conv_t5_1 = tf.layers.conv2d(conv_t5, 64, (3, 3), padding='same', activation=tf.nn.relu,name='conv_t5_1')
        conv_last = tf.layers.conv2d(conv_t5_1, num_classes, (15, 15), strides=(1, 1), padding='same',
                                     activation=tf.nn.relu,
                                     name='conv_last')
        final_crop = tf.slice(conv_last, [0, 0, 0, 0], [shape_input[0], shape_input[1], shape_input[2], num_classes],
                              name='final_crop')
        #final_crop_shape = tf.shape(final_crop , name='final_crop_shape')
        tf.summary.image('final_crop_Image' ,tf.expand_dims(final_crop[:,:,:,0],axis=3),
                         #tf.cast(final_crop[:,:,:,0],tf.uint8),
                         max_outputs= 1000)
    return final_crop

#tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    updatable_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='MyExtension')
    #print(type(updatable_variables))
    #print(updatable_variables)
    with tf.variable_scope('Optimization'):
        logits = tf.reshape(nn_last_layer,(-1,num_classes),name='logits')
        labels = tf.reshape(correct_label,(-1,num_classes),name='lables')
        cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        tf.summary.scalar('cross entropy loss',cross_entropy_loss)
        IoU,IoUop = tf.metrics.mean_iou(tf.argmax(labels, axis = 1), tf.argmax(logits, axis = 1),num_classes,name='IoUmean')
        tf.summary.scalar('IoUmean', IoU)
        total_loss = cross_entropy_loss + 10.0*(1.0 - IoU)
        tf.summary.scalar('total loss', total_loss)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        training_operation = optimizer.minimize(total_loss,var_list=updatable_variables) # ,var_list=updatable_variables
        return logits, training_operation, cross_entropy_loss,IoUop

#tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate, IoUop):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    correct_image = correct_label[:, :, :, 0]
    correct_image = tf.expand_dims(correct_image, axis=3)
    tf.summary.image('correct_Image', correct_image, max_outputs=1000)

    merged = tf.summary.merge_all()
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    writer = tf.summary.FileWriter('./TFlog', tf.get_default_graph())

    target = [train_op,
              cross_entropy_loss,
              merged,
              IoUop]
              #tf.get_default_graph().get_tensor_by_name('Optimization/logits:0'),
              #tf.get_default_graph().get_tensor_by_name('Optimization/lables:0')]
              #tf.get_default_graph().get_tensor_by_name('MyExtension/final_crop:0')
    idx = 0
    for i in range(epochs):
        batch_index = 0
        for batch_x, batch_y in get_batches_fn(batch_size):
            batch_index += 1
            feed_dict = {input_image: batch_x, correct_label: batch_y,
                         keep_prob: 1}
            out = sess.run(target, feed_dict = feed_dict)
            writer.add_summary(out[2],idx)
            idx = idx +1
            print("Epoch: {} batch: {} loss: {}"
                  .format(i+1,batch_index,out[1]))
        print("===================================================")

    pass

#tests.test_train_nn(train_nn)


def run():
    num_classes = 2
    image_shape = (160, 576) # (160, 576)
    data_dir = './data'
    runs_dir = './runs'
    #TF_log_dir = './TFlog'
    batch_size = 10
    num_epochs = 50
    learning_rate = 0.001
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/



    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function
        correct_label = tf.placeholder(tf.float32, [None, None, None, num_classes])
        #learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        layers_output = layers(input_image,vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        logits, train_op, cross_entropy_loss , IoUop= optimize(layers_output, correct_label, learning_rate, num_classes)
        # TODO: Train NN using the train_nn function
        train_nn(sess, num_epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
                 correct_label, keep_prob, learning_rate, IoUop)
        # TODO: Save inference data using helper.save_inference_samples
        saver = tf.train.Saver()
        saver.save(sess, './model/model.ckpt')
        print('model saved!')
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()
