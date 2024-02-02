from tyreClassifier.components.model_training import *
from tyreClassifier.logging import logger

STAGE_NAME = "Model Training stage"

class ModelTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ModelTrainingConfig()
        model_training = ModelTraining(config=config)
        model_selction = ModelSelction(config=config)
        rf = model_selction.model_RandomForest()
        KNN = model_selction.model_KNN()
        vgg = model_selction.model_VGG16()
        resnet = model_selction.model_resnet_50()
        model_training.RandomForest_training(rf)
        model_training.KNN_training(KNN)
        model_training.vgg16_training(vgg)
        model_training.resnet_50_training(resnet)



if __name__ == '__main__':
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e