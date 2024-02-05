from tyreClassifier.utils.common import *
from tyreClassifier.logging import logger
from tyreClassifier.components.model_training import ModelTrainingConfig
import os
from keras.models import load_model
 




class ModelPrediction:
    def __init__(self,config:ModelTrainingConfig):
        self.config=config


    def load_models(self):
        try:
            models_path = r'models'
            models=os.listdir(Path(models_path))
            model_dic={}
            for model in models:
                model_path = os.path.join(models_path,model)
                if model == 'KNN.pkl':
                    model_dic[model.split('.')[0]] = pickle.load(open(model_path, 'rb'))
                elif model == 'RandomForest.pkl':
                    model_dic[model.split('.')[0]] = pickle.load(open(model_path, 'rb'))      
                elif model == 'resnet_50.h5':
                    model_dic[model.split('.')[0]] = load_model(model_path)    
                elif model == 'vgg16.h5':
                    model_dic[model.split('.')[0]] = load_model(model_path)
            logger.info(f"Loding models {model_dic.keys()}")

            return model_dic
        except Exception as e:
            raise e

    def model_prediction(self,model_dic):
        try:
            models={}
            for model_name in model_dic.keys():
                model = model_dic[model_name]
                accuracy_test, classification_report1 = evaluate_models(self.config.flat_test_images,
                            self.config.flat_test_labels,self.config.test_images,self.config.test_labels,model_name,model)
                models[model_name] = accuracy_test
                best_model = [key for key in models if models[key]== max(models.values())]

            names, counts = list(models.keys()),list(models.values())
            best_modelPlot(names, counts,best_model)
            logger.info("Making predictions and finding accuracy for all models ")
                               
        except Exception as e:
            raise e
