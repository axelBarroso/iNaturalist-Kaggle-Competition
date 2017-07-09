from extract_features.harris_map import *

def create_Harris_score_maps(datatest):
    num_examples = datatest.num_examples
    harris = harris_class()
    for step in range(num_examples // cf.batch_size + 1):
        print 'Step ' + str(step) + ' of ' + str(num_examples // cf.batch_size + 1)
        start_time = time.time()
        offset = step * cf.batch_size
        images_g, paths = datatest.next_batch_gray_images(offset)
        maps = harris.Harris_score_Map(images_g)
        for i in product(xrange(len(maps))):
            name_file = os.path.splitext(paths[i])[0] + '_gray.h5'
            map = maps[i] / maps[i].max()
            # Save image in hdf5
            h5f = h5py.File(name_file, 'w')
            h5f.create_dataset('dataset_1', data=map)
            h5f.close()
        time_elapsed = time.time() - start_time
        print time_elapsed


def run():
    # Load train and Validation data
    data_provider = dataset()
    create_Harris_score_maps(data_provider.train)
    create_Harris_score_maps(data_provider.validation)
    # Load test data
    data_provider = dataset(False)
    create_Harris_score_maps(data_provider.test)