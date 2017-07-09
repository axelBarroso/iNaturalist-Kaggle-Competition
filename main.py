from extract_features import extract_features, create_Harris_maps
from loops import train_iNaturalist, test_iNaturalist, train_fc_iNaturalist, test_fc_iNaturalist

def run():

    # Create Harris score maps
    create_Harris_maps.run()

    # Extract train & Validation features
    # It is possible to pretraining this model before extracting features with train_iNaturalist.py function
    # For testing performance use test_iNaturalist.run()
    # train_iNaturalist.run()
    extract_features.run()

    # Train last fully connected layer with images and Harris attention maps
    # Modify model_configuration for selecting desired parameters
    train_fc_iNaturalist.run()

    # Test final model. Extract the most likely 5 predictions in csv format
    test_fc_iNaturalist.run()


if __name__ == '__main__':
    run()