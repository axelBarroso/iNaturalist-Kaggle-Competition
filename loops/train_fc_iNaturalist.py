import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import time

import tensorflow as tf
from tensorflow.contrib.framework.python.ops.variables import get_or_create_global_step
from tensorflow.python.platform import tf_logging as logging

from model.inception_resnet_v2 import inception_resnet_v2_arg_scope, fully_connected

slim = tf.contrib.slim


from data_providers.utils import *

#Get the latest checkpoint file
checkpoint_file = tf.train.latest_checkpoint(cf.log_fc_dir)

def run():

    # Create the log directory here. Must be done here otherwise import will activate this unneededly.
    if not os.path.exists(cf.log_fc_dir):
        os.mkdir(cf.log_fc_dir)

    # ======================= TRAINING PROCESS =========================
    # Now we start to construct the graph and build our model
    with tf.Graph().as_default() as graph:
        tf.logging.set_verbosity(tf.logging.INFO)  # Set the verbosity to INFO level

        # First create the dataset and load one batch
        data_provider = dataset()

        # Know the number steps to take before decaying the learning rate and batches per epoch
        num_batches_per_epoch = data_provider.train.num_examples / cf.batch_size
        num_steps_per_epoch = num_batches_per_epoch  # Because one step is one batch processed
        decay_steps = int(cf.num_epochs_before_decay * num_steps_per_epoch)

        shape = (None, 8, 8, 3072)
        shape_labels = (None)

        with tf.name_scope('inputs'):
            features = tf.placeholder(tf.float32, shape=shape, name='input_images')
            labels = tf.placeholder(tf.int64, shape=shape_labels, name='input_labels')

        # Create the model inference
        with slim.arg_scope(inception_resnet_v2_arg_scope()):
            logits, end_points = fully_connected(features, num_classes=data_provider.n_classes, is_training=True)

        # Define the scopes that you want to exclude for restoration
        # exclude = ['InceptionResnetV2/Logits', 'InceptionResnetV2/AuxLogits']
        # variables_to_restore = slim.get_variables_to_restore(exclude=exclude)

        variables_to_restore = slim.get_variables_to_restore()

        # Perform one-hot-encoding of the labels (Try one-hot-encoding within the load_batch function!)
        one_hot_labels = slim.one_hot_encoding(labels, data_provider.n_classes)

        # Performs the equivalent to tf.nn.sparse_softmax_cross_entropy_with_logits but enhanced with checks
        loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=logits)
        total_loss = tf.losses.get_total_loss()  # obtain the regularization losses as well

        # Create the global step for monitoring the learning_rate and training.
        global_step = get_or_create_global_step()

        # Define your exponentially decaying learning rate
        lr = tf.train.exponential_decay(
            learning_rate = cf.initial_learning_rate,
            global_step = global_step,
            decay_steps = decay_steps,
            decay_rate = cf.learning_rate_decay_factor,
            staircase=True)

        # Now we can define the optimizer that takes on the learning rate
        optimizer = tf.train.AdamOptimizer(learning_rate=lr)

        # Create the train_op
        train_op = slim.learning.create_train_op(total_loss, optimizer)

        # State the metrics that you want to predict. We get a predictions that is not one_hot_encoded.
        predictions = tf.argmax(end_points['Predictions'], 1)
        probabilities = end_points['Predictions']
        accuracy, accuracy_update = tf.contrib.metrics.streaming_accuracy(predictions, labels)
        metrics_op = tf.group(accuracy_update, probabilities)

        # Now finally create all the summaries you need to monitor and group them into one summary op.
        tf.summary.scalar('losses/Total_Loss', total_loss)
        tf.summary.scalar('accuracy', accuracy)
        tf.summary.scalar('learning_rate', lr)
        my_summary_op = tf.summary.merge_all()

        # Now we need to create a training step function that runs both the train_op, metrics_op and updates the global_step concurrently.
        def train_epoch(sess, train_op, global_step):
            '''
            Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
            '''
            # Check the time for each sess run
            total_loss_avg = []
            accuracy_avg = []
            num_examples = data_provider.train.num_examples
            data_provider.train.start_new_epoch()
            fetches = [train_op, global_step, accuracy, metrics_op]
            for step in range(num_examples // cf.batch_size +1):
                start_time = time.time()
                offset = step * cf.batch_size
                batch_features, batch_labels = data_provider.train.next_batch_features(offset)
                if batch_features == []:
                    break

                feed_dict = {
                    features: batch_features,
                    labels: batch_labels,
                }

                loss, global_step_count, accuracy_step, _ = sess.run(fetches, feed_dict=feed_dict)
                time_elapsed = time.time() - start_time
                total_loss_avg.append(loss)
                accuracy_avg.append(accuracy_step)

                # Run the logging to print some results
                # logging.info('Train. Global step %s: loss: %.4f (%.2f sec/step)', global_step_count, loss, time_elapsed)
                logging.info('Train. Step %s/%s. loss: %.4f. accuracy: %.4f (%.2f sec/step)', step, (num_examples // cf.batch_size), loss, accuracy_step, time_elapsed)

                if (step+1) % 1000 == 0:
                    sv.saver.save(sess, sv.save_path, global_step=sv.global_step)

            mean_loss = np.mean(total_loss_avg)
            mean_accuracy = np.mean(accuracy_avg)

            return mean_loss, mean_accuracy, global_step_count

        # Create a evaluation step function
        def validate_model(sess, metrics_op):
            '''
            Simply runs a session for the three arguments provided and gives a logging on the time elapsed for each global step
            '''
            # Check the time for each sess run
            total_loss_avg = []
            accuracy_avg = []
            num_examples = data_provider.validation.num_examples
            for step in range(num_examples // cf.batch_size + 1):
                offset = step * cf.batch_size
                batch_images, batch_labels = data_provider.validation.next_batch(offset)
                if batch_images == []:
                    break

                feed_dict = {
                    features: batch_features,
                    labels: batch_labels,
                }

                fetches = [metrics_op, accuracy]
                _, accuracy_value = sess.run(fetches, feed_dict=feed_dict)
                # total_loss_avg.append(loss)
                accuracy_avg.append(accuracy_value)

                # Run the logging to print some results
                logging.info('Validation. Step %s/%s. Accuracy: %s', step, (num_examples // cf.batch_size), accuracy_value)

            mean_accuracy = np.mean(accuracy_avg)
            logging.info('Validation data. Accuracy: %s', mean_accuracy)

        # Now we create a saver function that actually restores the variables from a checkpoint file in a sess
        # saver = tf.train.Saver(variables_to_restore)
        saver = tf.train.Saver()

        def restore_fn(sess):
            # return saver.restore(sess, cf.checkpoint_file)
            return saver.restore(sess, checkpoint_file)

        # Define your supervisor for running a managed session. Do not run the summary_op automatically or else it will consume too much memory
        # sv = tf.train.Supervisor(logdir=cf.log_fc_dir, summary_op=None, init_fn=restore_fn)
        sv = tf.train.Supervisor(logdir=cf.log_fc_dir, summary_op=None)

        # Run the managed session
        with sv.managed_session() as sess:
            for epoch in xrange(cf.num_epochs):
                # for step in xrange(1):
                # At the start of every epoch, show the vital information:
                # if step % num_batches_per_epoch == 0:
                logging.info('Epoch %s/%s', epoch , cf.num_epochs)
                learning_rate_value, accuracy_value = sess.run([lr, accuracy])
                logging.info('Current Learning Rate: %s', learning_rate_value)
                logging.info('Current Streaming Accuracy: %s', accuracy_value)

                # optionally, print your logits and predictions for a sanity check that things are going fine.
                # logits_value, probabilities_value, predictions_value, labels_value = sess.run(
                #     [logits, probabilities, predictions, labels])
                # print 'logits: \n', logits_value
                # print 'Probabilities: \n', probabilities_value
                # print 'predictions: \n', predictions_value
                # print 'Labels:\n:', labels_value

                loss, accuracy_epoch, _ = train_epoch(sess, train_op, sv.global_step)
                # summaries = sess.run([my_summary_op])
                # sv.summary_computed(sess, summaries)
                # We log the final training loss and accuracy
                logging.info('Epoch %s . Loss: %s', epoch, loss)
                logging.info('Epoch %s . Accuracy: %s', epoch, accuracy_epoch)

                # if epoch % 2 == 0:
                #     validate_model(sess, metrics_op = metrics_op)

            # We log the final training loss and accuracy
            logging.info('Final Loss: %s', loss)
            logging.info('Final Accuracy: %s', sess.run(accuracy))

            # Once all the training has been done, save the log files and checkpoint model
            logging.info('Finished training! Saving model to disk now.')
            # saver.save(sess, "./flowers_model.ckpt")
            sv.saver.save(sess, sv.save_path, global_step=sv.global_step)

if __name__ == '__main__':
    run()