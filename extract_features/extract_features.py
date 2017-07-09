import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf
from model.inception_resnet_v2 import inception_resnet_v2, inception_resnet_v2_arg_scope
from extract_features.harris_map import *
from data_providers.utils import *

slim = tf.contrib.slim

#Get the latest checkpoint file
checkpoint_file = tf.train.latest_checkpoint(cf.log_dir)

# The function extract features based on ResnetInception_v2 model pretrained in Imagenet. for images and harris attention maps.
# It extracts the Conv2d_7b_1x1 features from normalized images and harris attention maps and it concadenates them.
# Finally, it saves those features into hdf5 files for further training.

def run():

    # Construct the graph and build resnetInception model
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)  # Set the verbosity to INFO level

        # First create the dataset
        data_provider = dataset()
        data_provider_test = dataset(False)

        shape = (None, cf.image_shape[0], cf.image_shape[1], cf.image_shape[2])

        with tf.name_scope('inputs'):
            images = tf.placeholder(tf.float32, shape=shape, name='input_images')

        # Create the model inference
        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            logics, end_points = inception_resnet_v2(images, num_classes=data_provider.n_classes, is_training=True)

        # #get all the variables to restore from the checkpoint file and create the saver function to restore
        variables_to_restore = slim.get_variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        def restore_fn(sess):
            return saver.restore(sess, checkpoint_file)

        # Create the global step and an increment op for monitoring
        # global_step = get_or_create_global_step()
        # global_step_op = tf.assign(global_step,global_step + 1)  # no apply_gradient method so manually increasing the global_step
        # State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
        # predictions = tf.argmax(end_points['Predictions'], 1)
        # global_predictions = end_points['Predictions']
        # global_logics = end_points['Logits']
        pre_pool = end_points['Conv2d_7b_1x1']

        # Create a evaluation step function
        def extract_features_train(sess):
            # Check the time for each sess run
            num_examples = data_provider.train.num_examples
            data_provider.train.start_new_epoch()
            print 'Num of exampes test: ' + str(num_examples)
            for step in range(num_examples // cf.batch_size +1):
                offset = step*cf.batch_size
                batch_images, batch_images_mask, paths = data_provider.train.next_batch_two_images(offset)
                if images is None:
                    break

                feed_dict = {
                    images: batch_images,
                }

                fetches = [pre_pool]
                logics_batch = sess.run(fetches, feed_dict=feed_dict)

                feed_dict = {
                    images: batch_images_mask,
                }

                fetches = [pre_pool]
                logics_batch_mask = sess.run(fetches, feed_dict=feed_dict)

                features = np.concatenate([logics_batch[0], logics_batch_mask[0]], axis=3)

                print ('Extracting features. Step '+ str(step)+'/'+ str((num_examples // cf.batch_size + 1 )))
                for i in xrange(len(features)):
                    name_file = os.path.splitext(paths[i])[0] + '_features.h5'
                    h5f = h5py.File(name_file, 'w')
                    h5f.create_dataset('dataset_1', data=features[i])
                    h5f.close()

        def extract_features_validation(sess):
            # Check the time for each sess run
            num_examples = data_provider.validation.num_examples
            print 'Num of exampes test: ' + str(num_examples)
            for step in range(num_examples // cf.batch_size + 1):
                offset = step * cf.batch_size
                batch_images, batch_images_mask, paths = data_provider.validation.next_batch_two_images(offset)
                if images is None:
                    break

                feed_dict = {
                    images: batch_images,
                }

                fetches = [pre_pool]
                logics_batch = sess.run(fetches, feed_dict=feed_dict)

                feed_dict = {
                    images: batch_images_mask,
                }

                fetches = [pre_pool]
                logics_batch_mask = sess.run(fetches, feed_dict=feed_dict)

                features = np.concatenate([logics_batch[0], logics_batch_mask[0]], axis=3)

                print (
                'Extracting features. Step ' + str(step) + '/' + str((num_examples // cf.batch_size + 1)))
                for i in xrange(len(features)):
                    name_file = os.path.splitext(paths[i])[0] + '_features.h5'
                    h5f = h5py.File(name_file, 'w')
                    h5f.create_dataset('dataset_1', data=features[i])
                    h5f.close()

        def extract_features_test(sess):
            # Check the time for each sess run
            num_examples = data_provider_test.test.num_examples
            print 'Num of exampes test: ' + str(num_examples)
            for step in range(num_examples // cf.batch_size + 1):
                offset = step * cf.batch_size
                batch_images, batch_images_mask, paths = data_provider_test.test.next_batch_two_images(offset)
                if images is None:
                    break

                feed_dict = {
                    images: batch_images,
                }

                fetches = [pre_pool]
                logics_batch = sess.run(fetches, feed_dict=feed_dict)

                feed_dict = {
                    images: batch_images_mask,
                }

                fetches = [pre_pool]
                logics_batch_mask = sess.run(fetches, feed_dict=feed_dict)

                features = np.concatenate([logics_batch[0], logics_batch_mask[0]], axis=3)

                print (
                    'Extracting features. Step ' + str(step) + '/' + str((num_examples // cf.batch_size + 1)))
                for i in xrange(len(features)):
                    name_file = os.path.splitext(paths[i])[0] + '_features.h5'
                    h5f = h5py.File(name_file, 'w')
                    h5f.create_dataset('dataset_1', data=features[i])
                    h5f.close()

        # Define your supervisor for running a managed session. Do not run the summary_op automatically or else it will consume too much memory
        sv = tf.train.Supervisor(logdir=cf.log_dir, saver = None, summary_op=None, init_fn=restore_fn)

        # Run the managed session
        with sv.managed_session() as sess:

            extract_features_train(sess)

            extract_features_validation(sess)

            extract_features_test(sess)

            print 'Finished extract features!'

if __name__ == '__main__':
    run()
