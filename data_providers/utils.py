import tensorflow as tf
import json
import os.path
from skimage import io
from skimage.transform import resize
import numpy as np
import configuration.model_configuration as cf
import h5py
from joblib import Parallel, delayed
from skimage.color import rgba2rgb, rgb2gray

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

class train(object):
    def __init__(self, image_shape, is_debugging):
        self.num_examples = 0
        self.images_info = []
        self.labels = []
        self.image_shape = image_shape
        self.shuffle_every_epoch = True
        self.epoch_images_info = []
        self.epoch_labels = []
        self.is_debugging = is_debugging

    def next_batch(self, offset):
        images = []
        labels = []

        for i in xrange(cf.batch_size):
            if len(self.images_info) == (offset + i):
                break
            # h5f = h5py.File(self.epoch_images_info[offset+i], 'r')
            # im = h5f['dataset_1'][:]
            im = io.imread(self.epoch_images_info[offset + i], as_grey=False)
            if len(im.shape) == 2:
                im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
            if im.shape[2] == 4:
                im = rgba2rgb(im)
            # im = im[..., np.newaxis]
            diff = im.shape[0] - im.shape[1]
            if diff > 0:
                im = im[diff / 2:im.shape[0] - diff / 2, :]
            elif diff < 0:
                im = im[:, -diff / 2:im.shape[1] + diff / 2]
            im = resize(im, (299, 299), mode='reflect')
            # RGB
            image_mean = [0.463, 0.479, 0.384]
            image_std = [0.126, 0.116, 0.15]
            for j in xrange(3):
                im[:, :, j] = (im[:, :, j] - image_mean[j]) / image_std[j]
            im.astype(np.float32)
            images.append(im)
            labels.append(self.epoch_labels[offset + i])
            # h5f.close()

        images = np.asarray(images)
        labels = np.asarray(labels)

        return images, labels

    def next_batch_features(self, offset):
        features = []
        labels = []

        for i in xrange(cf.batch_size):
            if len(self.images_info) == (offset + i):
                break
            name_file = os.path.splitext(self.epoch_images_info[offset+i])[0] + '_features.h5'
            if os.path.isfile(name_file):
                h5f = h5py.File(name_file, 'r')
                feature = h5f['dataset_1'][:]
                if len(feature) == 8:
                    features.append(feature)
                    labels.append(self.epoch_labels[offset + i])
                    h5f.close()
            else:
                continue

        features = np.asarray(features)
        labels = np.asarray(labels)

        return features, labels

    def next_batch_gray_images(self, offset):
        images = []
        paths = []

        for i in xrange(cf.batch_size):
            if len(self.images_info) == (offset + i):
                break
            # h5f = h5py.File(self.epoch_images_info[offset+i], 'r')
            name_file = os.path.splitext(self.images_info[offset + i])[0] + '.h5'
            if os.path.isfile(name_file):
                os.remove(name_file)
            # im = h5f['dataset_1'][:]
            im = io.imread(self.epoch_images_info[offset + i], as_grey=False)
            if len(im.shape) == 2:
                im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
            if im.shape[2] == 4:
                im = rgba2rgb(im)
            # im = im[..., np.newaxis]
            diff = im.shape[0] - im.shape[1]
            if diff > 0:
                im = im[diff / 2:im.shape[0] - diff / 2, :]
            elif diff < 0:
                im = im[:, -diff / 2:im.shape[1] + diff / 2]
            im = resize(im, (299, 299), mode='reflect')
            # RGB
            im = rgb2gray(im)
            im = im[..., np.newaxis]
            images.append(im)
            paths.append(self.epoch_images_info[offset + i])

        images = np.asarray(images)
        paths = np.asarray(paths)

        return images, paths

    def next_batch_two_images(self, offset):
        images = []
        images_mask = []
        paths = []

        for i in xrange(cf.batch_size):
            if len(self.images_info) == (offset + i):
                break
            name_file = os.path.splitext(self.epoch_images_info[offset + i])[0] + '_gray.h5'
            h5f = h5py.File(name_file, 'r')
            mask = h5f['dataset_1'][:]
            im = io.imread(self.epoch_images_info[offset + i], as_grey=False)
            if len(im.shape) == 2:
                im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
            if im.shape[2] == 4:
                im = rgba2rgb(im)
            # im = im[..., np.newaxis]
            diff = im.shape[0] - im.shape[1]
            if diff > 0:
                im = im[diff / 2:im.shape[0] - diff / 2, :]
            elif diff < 0:
                im = im[:, -diff / 2:im.shape[1] + diff / 2]
            im = resize(im, (299, 299), mode='reflect')
            im_mask = np.zeros((299,299,3))
            # RGB
            image_mean = [0.463, 0.479, 0.384]
            image_std = [0.126, 0.116, 0.15]
            for j in xrange(3):
                im[:, :, j] = (im[:, :, j] - image_mean[j]) / image_std[j]
                im_mask[:,:,j] = im[:, :, j] * mask
            im.astype(np.float32)
            images.append(im)
            images_mask.append(im_mask)
            paths.append(self.epoch_images_info[offset + i])
            # h5f.close()

        images = np.asarray(images)
        images_mask = np.asarray(images_mask)
        paths = np.asarray(paths)

        return images, images_mask, paths

    def next_batch_parallel(self, offset):
        if len(self.epoch_images_info) < (offset + cf.batch_size):
            return None, None

        images = Parallel(n_jobs= -1)(
                delayed(parallel_prepare_batch) (self.epoch_images_info[offset + i]) for i in xrange(cf.batch_size))

        images = np.asarray(images)
        labels = np.asarray(self.epoch_labels[offset:offset+cf.batch_size])
        return images, labels

    def start_new_epoch(self):
        if self.shuffle_every_epoch:
            images, labels = self.shuffle_images_and_labels(
                self.images_info, self.labels)
        else:
            images, labels = self.images_info, self.labels
        images = np.asarray(images)
        labels = np.asarray(labels)
        self.epoch_images_info = images
        self.epoch_labels = labels

    def shuffle_images_and_labels(self, images, labels):
        rand_indexes = np.random.permutation(len(images))
        # return images, labels
        return images[rand_indexes], labels[rand_indexes]

    def load_dataset(self, info, labels, n_classes):
        count = 0
        self.n_classes = n_classes
        for i in xrange(len(info)):
            base = cf.dataset_path + info[i]['file_name']
            if os.path.isfile(base):
                name_file = os.path.splitext(base)[0] + '.h5'
                self.images_info.append(base)
                self.labels.append(labels[i]['category_id'])
                count +=1
            else:
                print 'ERROR. File has not been found: ' + base
                continue

            if self.is_debugging:
                if i > 80:
                    break

        self.images_info = np.asarray(self.images_info)
        self.labels = np.asarray(self.labels)
        self.num_examples = count

    def create_hdf5_files(self, info):
        print 'Creating train hdf5 files . . .'
        print 'Dataset size of ' + str(len(info)) + ' images'
        Parallel(n_jobs= -1)(
            delayed(parallel_read) (cf.dataset_path + info[i]['file_name'], i) for i in xrange(len(info)))


    def create_tfRecord_files(self, info, labels):
        print 'Creating train tfRecord files . . .'
        print 'Dataset size of ' + str(len(info)) + ' images'
        writer = tf.python_io.TFRecordWriter(cf.tfRecord_train_filename)
        for i in range(len(info)):
            # Load the image
            img = load_image(cf.dataset_path + info[i]['file_name'])

            # Create a feature
            feature = {'train/label': _int64_feature(labels[i]['category_id']),
                       'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())

            if self.is_debugging:
                if i > 80:
                    break

        writer.close()

class validation(object):
    def __init__(self, image_shape, is_debugging):
        self.num_examples = 0
        self.images_info = []
        self.labels = []
        self.image_shape = image_shape
        self.is_debugging = is_debugging

    def next_batch(self, offset):
        images = []
        labels = []

        for i in xrange(cf.batch_size):
            if len(self.images_info) == (offset + i):
                break
            im = io.imread(self.images_info[offset + i], as_grey=False)
            if len(im.shape) == 2:
                im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
            if im.shape[2] == 4:
                im = rgba2rgb(im)
            diff = im.shape[0] - im.shape[1]
            if diff > 0:
                im = im[diff / 2:im.shape[0] - diff / 2, :]
            elif diff < 0:
                im = im[:, -diff / 2:im.shape[1] + diff / 2]
            im = resize(im, (299, 299), mode='reflect')
            # RGB
            image_mean = [0.463, 0.479, 0.384]
            image_std = [0.126, 0.116, 0.15]
            for j in xrange(3):
                im[:, :, j] = (im[:, :, j] - image_mean[j]) / image_std[j]
            im.astype(np.float32)
            images.append(im)
            labels.append(self.labels[offset + i])

        images = np.asarray(images)
        labels = np.asarray(labels)

        return images, labels

    def next_batch_features(self, offset):
        features = []
        labels = []

        for i in xrange(cf.batch_size):
            if len(self.images_info) == (offset + i):
                break
            name_file = os.path.splitext(self.images_info[offset+i])[0] + '_features.h5'
            if os.path.isfile(name_file):
                h5f = h5py.File(name_file, 'r')
                feature = h5f['dataset_1'][:]
                if len(feature) == 8:
                    features.append(feature)
                    labels.append(self.labels[offset + i])
                    h5f.close()
            else:
                continue

        features = np.asarray(features)
        labels = np.asarray(labels)

        return features, labels

    def next_batch_gray_images(self, offset):
        images = []
        paths = []

        for i in xrange(cf.batch_size):
            if len(self.images_info) == (offset + i):
                break
            name_file = os.path.splitext(self.images_info[offset + i])[0] + '.h5'
            if os.path.isfile(name_file):
                os.remove(name_file)
            im = io.imread(self.epoch_images_info[offset + i], as_grey=False)
            if len(im.shape) == 2:
                im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
            if im.shape[2] == 4:
                im = rgba2rgb(im)
            diff = im.shape[0] - im.shape[1]
            if diff > 0:
                im = im[diff / 2:im.shape[0] - diff / 2, :]
            elif diff < 0:
                im = im[:, -diff / 2:im.shape[1] + diff / 2]
            im = resize(im, (299, 299), mode='reflect')
            # RGB
            im = rgb2gray(im)
            im = im[..., np.newaxis]
            images.append(im)
            paths.append(self.images_info[offset + i])

        images = np.asarray(images)
        paths = np.asarray(paths)

        return images, paths

    def next_batch_two_images(self, offset):
        images = []
        images_mask = []
        paths = []

        for i in xrange(cf.batch_size):
            if len(self.images_info) == (offset + i):
                break
            name_file = os.path.splitext(self.images_info[offset + i])[0] + '_gray.h5'
            h5f = h5py.File(name_file, 'r')
            mask = h5f['dataset_1'][:]
            im = io.imread(self.images_info[offset + i], as_grey=False)
            if len(im.shape) == 2:
                im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
            if im.shape[2] == 4:
                im = rgba2rgb(im)
            diff = im.shape[0] - im.shape[1]
            if diff > 0:
                im = im[diff / 2:im.shape[0] - diff / 2, :]
            elif diff < 0:
                im = im[:, -diff / 2:im.shape[1] + diff / 2]
            im = resize(im, (299, 299), mode='reflect')
            im_mask = np.zeros((299,299,3))
            # RGB
            image_mean = [0.463, 0.479, 0.384]
            image_std = [0.126, 0.116, 0.15]
            for j in xrange(3):
                im[:, :, j] = (im[:, :, j] - image_mean[j]) / image_std[j]
                im_mask[:,:,j] = im[:, :, j] * mask
            im.astype(np.float32)
            images.append(im)
            images_mask.append(im_mask)
            paths.append(self.images_info[offset + i])

        images = np.asarray(images)
        images_mask = np.asarray(images_mask)
        paths = np.asarray(paths)

        return images, images_mask, paths

    def next_batch_parallel(self, offset):
        if len(self.images_info) < (offset + cf.batch_size):
            return None, None

        images = Parallel(n_jobs= -1)(
                delayed(parallel_prepare_batch) (self.images_info[offset + i]) for i in xrange(cf.batch_size))

        images = np.asarray(images)
        labels = np.asarray(self.labels[offset:offset+cf.batch_size])
        return images, labels

    def load_dataset(self, info, labels, n_classes):
        count = 0
        self.n_classes = n_classes
        for i in xrange(len(info)):
            base = cf.dataset_path + info[i]['file_name']
            if os.path.isfile(base):
                name_file = os.path.splitext(base)[0] + '.h5'
                self.images_info.append(base)
                self.labels.append(labels[i]['category_id'])
                count +=1
            else:
                print 'ERROR. File has not been found: ' + base
                continue

            if self.is_debugging:
                if i > 80:
                    break

        self.images_info = np.asarray(self.images_info)
        self.labels = np.asarray(self.labels)
        self.num_examples = count

    def create_hdf5_files(self, info):
        print 'Creating train hdf5 files . . .'
        print 'Dataset size of ' + str(len(info)) + ' images'
        Parallel(n_jobs= -1)(
            delayed(parallel_read) (cf.dataset_path + info[i]['file_name'], i) for i in xrange(len(info)))

    def create_tfRecord_files(self, info, labels):
        print 'Creating train tfRecord files . . .'
        print 'Dataset size of ' + str(len(info)) + ' images'
        writer = tf.python_io.TFRecordWriter(cf.tfRecord_train_filename)
        for i in range(len(info)):
            # Load the image
            img = load_image(cf.dataset_path + info[i]['file_name'])

            # Create a feature
            feature = {'train/label': _int64_feature(labels[i]['category_id']),
                       'train/image': _bytes_feature(tf.compat.as_bytes(img.tostring()))}
            # Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            # Serialize to string and write on the file
            writer.write(example.SerializeToString())

            if self.is_debugging:
                if i > 80:
                    break
        writer.close()

class test(object):
    def __init__(self, image_shape):
        self.num_examples = 0
        self.images_info = []
        self.id = []
        self.image_shape = image_shape
        self.shuffle_every_epoch = True

    def next_batch(self, offset):
        images = []
        ids = []

        for i in xrange(cf.batch_size_test):
            if len(self.images_info) == (offset + i):
                break
            im = io.imread(self.images_info[offset + i], as_grey=False)
            if len(im.shape) == 2:
                im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
            if im.shape[2] == 4:
                im = rgba2rgb(im)
            diff = im.shape[0] - im.shape[1]
            if diff > 0:
                im = im[diff / 2:im.shape[0] - diff / 2, :]
            elif diff < 0:
                im = im[:, -diff / 2:im.shape[1] + diff / 2]
            im = resize(im, (299, 299), mode='reflect')
            # RGB
            image_mean = [0.463, 0.479, 0.384]
            image_std = [0.126, 0.116, 0.15]
            for j in xrange(3):
                im[:, :, j] = (im[:, :, j] - image_mean[j]) / image_std[j]
            im.astype(np.float32)
            images.append(im)
            ids.append(self.id[offset + i])

        images = np.asarray(images)
        ids = np.asarray(ids)
        return images, ids

    def next_batch_features(self, offset):
        features = []
        ids = []

        for i in xrange(cf.batch_size_test):
            if len(self.images_info) == (offset + i):
                break
            name_file = os.path.splitext(self.images_info[offset+i])[0] + '_features.h5'
            h5f = h5py.File(name_file, 'r')
            feature = h5f['dataset_1'][:]
            features.append(feature)
            ids.append(self.id[offset + i])
            h5f.close()

        features = np.asarray(features)
        ids = np.asarray(ids)

        return features, ids

    def next_batch_two_images(self, offset):
        images = []
        images_mask = []
        paths = []

        for i in xrange(cf.batch_size):
            if len(self.images_info) == (offset + i):
                break
            name_file = os.path.splitext(self.images_info[offset + i])[0] + '_gray.h5'
            h5f = h5py.File(name_file, 'r')
            mask = h5f['dataset_1'][:]
            im = io.imread(self.images_info[offset + i], as_grey=False)
            if len(im.shape) == 2:
                im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
            if im.shape[2] == 4:
                im = rgba2rgb(im)
            # im = im[..., np.newaxis]
            diff = im.shape[0] - im.shape[1]
            if diff > 0:
                im = im[diff / 2:im.shape[0] - diff / 2, :]
            elif diff < 0:
                im = im[:, -diff / 2:im.shape[1] + diff / 2]
            im = resize(im, (299, 299), mode='reflect')
            im_mask = np.zeros((299,299,3))
            # RGB
            image_mean = [0.463, 0.479, 0.384]
            image_std = [0.126, 0.116, 0.15]
            for j in xrange(3):
                im[:, :, j] = (im[:, :, j] - image_mean[j]) / image_std[j]
                im_mask[:,:,j] = im[:, :, j] * mask
            im.astype(np.float32)
            images.append(im)
            images_mask.append(im_mask)
            paths.append(self.images_info[offset + i])

        images = np.asarray(images)
        images_mask = np.asarray(images_mask)
        paths = np.asarray(paths)

        return images, images_mask, paths

    def next_batch_gray_images(self, offset):
        images = []
        paths = []

        for i in xrange(cf.batch_size):
            if len(self.images_info) == (offset + i):
                break
            name_file = os.path.splitext(self.images_info[offset + i])[0] + '.h5'
            if os.path.isfile(name_file):
                os.remove(name_file)
            im = io.imread(self.images_info[offset + i], as_grey=False)
            if len(im.shape) == 2:
                im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
            if im.shape[2] == 4:
                im = rgba2rgb(im)
            # im = im[..., np.newaxis]
            diff = im.shape[0] - im.shape[1]
            if diff > 0:
                im = im[diff / 2:im.shape[0] - diff / 2, :]
            elif diff < 0:
                im = im[:, -diff / 2:im.shape[1] + diff / 2]
            im = resize(im, (299, 299), mode='reflect')
            # RGB
            im = rgb2gray(im)
            im = im[..., np.newaxis]
            images.append(im)
            paths.append(self.images_info[offset + i])

        images = np.asarray(images)
        paths = np.asarray(paths)

        return images, paths

    def next_batch_two_images(self, offset):
        images = []
        images_mask = []
        paths = []

        for i in xrange(cf.batch_size):
            if len(self.images_info) == (offset + i):
                break
            name_file = os.path.splitext(self.images_info[offset + i])[0] + '_gray.h5'
            h5f = h5py.File(name_file, 'r')
            mask = h5f['dataset_1'][:]
            im = io.imread(self.images_info[offset + i], as_grey=False)
            if len(im.shape) == 2:
                im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
            if im.shape[2] == 4:
                im = rgba2rgb(im)
            # im = im[..., np.newaxis]
            diff = im.shape[0] - im.shape[1]
            if diff > 0:
                im = im[diff / 2:im.shape[0] - diff / 2, :]
            elif diff < 0:
                im = im[:, -diff / 2:im.shape[1] + diff / 2]
            im = resize(im, (299, 299), mode='reflect')
            im_mask = np.zeros((299,299,3))
            # RGB
            image_mean = [0.463, 0.479, 0.384]
            image_std = [0.126, 0.116, 0.15]
            for j in xrange(3):
                im[:, :, j] = (im[:, :, j] - image_mean[j]) / image_std[j]
                im_mask[:,:,j] = im[:, :, j] * mask
            im.astype(np.float32)
            images.append(im)
            images_mask.append(im_mask)
            paths.append(self.images_info[offset + i])
            # h5f.close()

        images = np.asarray(images)
        images_mask = np.asarray(images_mask)
        paths = np.asarray(paths)

        return images, images_mask, paths

    def load_dataset(self, info, n_classes):
        count = 0
        self.n_classes = n_classes
        for i in xrange(len(info)):
            base = cf.dataset_path + 'test2017/' + info[i]['file_name']
            if os.path.isfile(base):
                name_file = os.path.splitext(base)[0] + '.h5'
                self.images_info.append(base)
                self.id.append(info[i]['id'])
                count += 1
            else:
                print 'ERROR. File has not been found: ' + base
                continue

        self.images_info = np.asarray(self.images_info)
        self.id = np.asarray(self.id)
        self.num_examples = count

    def create_hdf5_files(self, info):
        print 'Creating validation hdf5 files . . . '
        print 'Dataset size of ' + str(len(info)) + ' images'
        Parallel(n_jobs=-1)(
            delayed(parallel_read)(cf.dataset_path + info[i]['file_name'], i) for i in xrange(len(info)))

def parallel_prepare_batch(step_image_name):

    im = io.imread(step_image_name, as_grey=False)
    if len(im.shape) == 2:
        im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
    diff = im.shape[0] - im.shape[1]
    if diff > 0:
        im = im[diff / 2:im.shape[0] - diff / 2, :]
    elif diff < 0:
        im = im[:, -diff / 2:im.shape[1] + diff / 2]
    im = resize(im, (299, 299), mode='reflect')
    # RGB
    image_mean = [0.463, 0.479, 0.384]
    image_std = [0.126, 0.116, 0.15]
    for j in xrange(3):
        im[:, :, j] = (im[:, :, j] - image_mean[j]) / image_std[j]
    im.astype(np.float32)
    return im

def parallel_read(base, index):
    print str(index)

    if os.path.isfile(base):
        name_file = os.path.splitext(base)[0] + '.h5'
        if os.path.isfile(name_file):
            return
        # Save image in hdf5
        im = io.imread(base, as_grey=False)
        if len(im.shape) == 2:
            im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
        # im = im[..., np.newaxis]
        diff = im.shape[0] - im.shape[1]
        if diff > 0:
            im = im[diff / 2:im.shape[0] - diff / 2, :]
        elif diff < 0:
            im = im[:, -diff / 2:im.shape[1] + diff / 2]
        im = resize(im, (299, 299), mode='reflect')
        # RGB
        image_mean = [0.463, 0.479, 0.384]
        image_std = [0.126, 0.116, 0.15]
        for j in xrange(3):
            im[:, :, j] = (im[:, :, j] - image_mean[j]) / image_std[j]
        im.astype(np.float32)
        h5f = h5py.File(name_file, 'w')
        h5f.create_dataset('dataset_1', data=im)
        h5f.close()

def load_image(base):

    im = io.imread(base, as_grey=False)
    if len(im.shape) == 2:
        im = np.repeat(im[:, :, np.newaxis], 3, axis=2)
    # im = im[..., np.newaxis]
    diff = im.shape[0] - im.shape[1]
    if diff > 0:
        im = im[diff / 2:im.shape[0] - diff / 2, :]
    elif diff < 0:
        im = im[:, -diff / 2:im.shape[1] + diff / 2]
    im = resize(im, (299, 299), mode='reflect')
    # RGB
    image_mean = [0.463, 0.479, 0.384]
    image_std = [0.126, 0.116, 0.15]
    for j in xrange(3):
        im[:, :, j] = (im[:, :, j] - image_mean[j]) / image_std[j]
    return im.astype(np.float32)


class dataset(object):
    def __init__(self, is_training = True):
        self._base_dir = cf.dataset_path
        self.json_train_path = cf.json_train_path
        self.json_validation_path = cf.json_val_path
        self.json_test_path = cf.json_test_path
        self.train = train(cf.image_shape, cf.is_debugging)
        self.validation = validation(cf.image_shape, cf.is_debugging)
        self.test = test(cf.image_shape)
        self.n_classes = 0
        self._height = cf.image_shape[0]
        self._width = cf.image_shape[1]
        self._channels = cf.image_shape[2]
        self.data_shape = cf.image_shape
        self.is_debugging = cf.is_debugging
        self.is_training = is_training
        self._load_dataset()

    def _load_dataset(self):
        if self.is_training:
            with open(self.json_train_path) as json_data:
                d = json.load(json_data)
            self.n_classes = len(d['categories'])
            if cf.create_hdf5_dataset:
                self.train.create_hdf5_files(d['images'])
            if cf.create_tfRecords_dataset:
                self.train.create_tfRecord_files(d['images'], d['annotations'])
            self.train.load_dataset(d['images'], d['annotations'], self.n_classes)

            with open(self.json_validation_path) as json_data:
                d_v = json.load(json_data)
            if cf.create_hdf5_dataset:
                self.validation.create_hdf5_files(d_v['images'])
            if cf.create_tfRecords_dataset:
                self.validation.create_tfRecord_files(d_v['images'], d_v['annotations'])
            self.validation.load_dataset(d_v['images'], d_v['annotations'], self.n_classes)

        else:
            with open(self.json_test_path) as json_data:
                d = json.load(json_data)
            self.n_classes = len(d['categories'])
            self.test.load_dataset(d['images'], self.n_classes)