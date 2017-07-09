
# Baisc settings
is_debugging = False
is_training = True
create_hdf5_dataset = False
create_tfRecords_dataset = False
image_shape = (299, 299, 3)

# Paths
dataset_path = "/media/axel/store_data/datasets/Kaggle_iNaturalist/"

tfRecord_train_filename = "/media/axel/store_data/datasets/Kaggle_iNaturalist/train.tfrecords"
tfRecord_val_filename = "/media/axel/store_data/datasets/Kaggle_iNaturalist/val.tfrecords"
tfRecord_pattern_train = 'train.tfrecords'

json_train_path = '/media/axel/store_data/datasets/Kaggle_iNaturalist/json/train2017.json'
json_val_path = '/media/axel/store_data/datasets/Kaggle_iNaturalist/json/val2017.json'
json_test_path = '/media/axel/store_data/datasets/Kaggle_iNaturalist/json/test2017.json'

#State the number of classes to predict:
num_classes = 5089

#State where your log file is at. If it doesn't exist, create it.
log_dir = './log'
log_fc_dir = './log_fc'

#Create a new evaluation log directory to visualize the validation process
log_eval = './log_eval_test'

#State where your checkpoint file is
checkpoint_file = './inception_resnet_v2_2016_08_30.ckpt'

#State your batch size
batch_size = 16
batch_size_test = 32

# State the number of epochs
num_epochs = 10

#Learning rate information and configuration (Up to you to experiment)
initial_learning_rate = 0.0002
learning_rate_decay_factor = 0.7
num_epochs_before_decay = 2

