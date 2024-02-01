from tyreClassifier.components.data_preprocessing import *
from tyreClassifier.logging import logger

STAGE_NAME = "Data Preprocessing stage"

class DatapreprocessingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = DataPreprocessingConfig
        data_Preprocessing = DataPreprocessing(config=config)
        images, labels= data_Preprocessing.load_data()
        train_images, test_images, train_labels, test_labels, val_images, val_labels = data_Preprocessing.split_data(images, labels)
        split_data=[train_images, test_images, train_labels, test_labels, val_images, val_labels]
        data_Preprocessing.save_split_data(split_data)
        
        n_train = train_labels.shape[0]
        n_val = val_labels.shape[0]
        n_test = test_labels.shape[0]

        print("Number of training examples: {}".format(n_train))
        print("Number of validation examples: {}".format(n_val))
        print("Number of testing examples: {}".format(n_test))

        print("Training images are of shape: {}".format(train_images.shape))
        print("Training labels are of shape: {}".format(train_labels.shape))
        print("Validation images are of shape: {}".format(val_images.shape))
        print("Validation labels are of shape: {}".format(val_labels.shape))
        print("Test images are of shape: {}".format(test_images.shape))
        print("Test labels are of shape: {}".format(test_labels.shape))



if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DatapreprocessingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e