import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import csv
import tensorflow as tf
from model.inception_resnet_v2 import inception_resnet_v2_arg_scope, fully_connected
from data_providers.utils import *

slim = tf.contrib.slim

#Get the latest checkpoint file
checkpoint_file = tf.train.latest_checkpoint(cf.log_fc_dir)

def run():

    # ======================= TESTING PROCESS =========================
    # Now we start to construct the graph and build our model
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)  # Set the verbosity to INFO level

        # First create the dataset and load one batch
        data_provider = dataset(False)

        shape = (None, 8, 8, 3072)

        with tf.name_scope('inputs'):
            features = tf.placeholder(tf.float32, shape=shape, name='input_images')

        # Create the model inference
        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            logits, end_points = fully_connected(features, num_classes=data_provider.n_classes, is_training=True)

        #get all the variables to restore from the checkpoint file and create the saver function to restore
        variables_to_restore = slim.get_variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        def restore_fn(sess):
            return saver.restore(sess, checkpoint_file)

        # Create the global step and an increment op for monitoring
        # global_step = get_or_create_global_step()
        # global_step_op = tf.assign(global_step,global_step + 1)  # no apply_gradient method so manually increasing the global_step
        # State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
        predictions = tf.argmax(end_points['Predictions'], 1)
        global_predictions = end_points['Predictions']

        # Create a evaluation step function
        def test_model(sess):
            # Check the time for each sess run
            num_examples = data_provider.test.num_examples
            print 'Num of exampes Test: ' + str(num_examples)
            for step in range(num_examples // cf.batch_size_test +1):
                offset = step*cf.batch_size_test
                batch_images, batch_id = data_provider.test.next_batch_features(offset)
                if batch_images is None:
                    break

                feed_dict = {
                    features: batch_images,
                }

                fetches = [global_predictions, predictions]
                global_predictions_b, predictions_b = sess.run(fetches, feed_dict=feed_dict)
                print ('Testing. Step '+ str(step)+'/'+ str((num_examples // cf.batch_size_test)))
                # Write predictions value to file
                for i in xrange(len(global_predictions_b)):
                    tmp = global_predictions_b[i]
                    out = tmp.argsort()[-5:][::-1]
                    test_writer.writerow([str(batch_id[i])+','+str(out[0])+' '+str(out[1])+' '+str(out[2])+' '+str(out[3])+' '+str(out[4])])

        # Define your supervisor for running a managed session. Do not run the summary_op automatically or else it will consume too much memory
        sv = tf.train.Supervisor(logdir=cf.log_fc_dir, summary_op=None, init_fn=restore_fn)

        with open('test_data.csv', 'wb') as csvfile:
            test_writer = csv.writer(csvfile, delimiter="|", quoting = csv.QUOTE_NONE, quotechar='')
            test_writer.writerow(['id,predicted'])
            # Run the managed session
            with sv.managed_session() as sess:
                # sess.run(sv.global_step)

                test_model(sess)

                print 'Finished testing!'

if __name__ == '__main__':
    run()