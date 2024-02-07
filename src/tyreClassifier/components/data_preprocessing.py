from tyreClassifier.logging import logger
import cv2
import tqdm
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import numpy as np
from pathlib import Path


@dataclass(frozen=True)
class DataPreprocessingConfig:
    dataSet = 'artifacts\data_ingestion\Digital images of defective and good condition tyres'
    preprocessed_data_path= 'artifacts\data_Preprocessing'
    class_names = os.listdir(dataSet)
    nb_classes = len(class_names)
    image_size = (224,224)
    

class DataPreprocessing:
    def __init__(self, config: DataPreprocessingConfig):
        self.config = config

    
    def load_data(self):
        try: 
            images = []
            labels = []
            flat_images = []
    
        # iterate through folders in each dataset
            for folder in os.listdir(self.config.dataSet):
                
                if folder in self.config.class_names[0]: label = 0
                elif folder in self.config.class_names[1]: label = 1
    
                folder_path=Path(os.path.join(self.config.dataSet, folder))
        
                # iterate through each image in folder
                for file in (os.listdir(folder_path)):
    
                    # get pathname of each image
                    img_path = os.path.join(folder_path, file)
                    
                    # Open and resize the| img
                    image = cv2.imread(img_path)
                    image = cv2.resize(image, self.config.image_size)
                    flat_image=(image.flatten())
    
                    # Append the image and its corresponding label to the output
                    images.append(image)
                    labels.append(label)
                    flat_images.append(flat_image)
                    print(len(flat_images))
            images = np.array(images)
            labels = np.array(labels, dtype = 'int32')
            flat_images= np.array(flat_images, dtype = 'float32')
            logger.info("loading image and Converting into vectors ")
            return images, labels, flat_images

        except Exception as e:
            raise e

    def split_data(self,images,labels,flat_images):
        try:
            images, labels,flat_images = shuffle(images, labels,flat_images, random_state=10)
            images = images / 255.0 
            flat_images = flat_images/255.0
    
            print(len(images),len(labels),len(flat_images))
            flat_train_images, flat_test_images, flat_train_labels, flat_test_labels = train_test_split(flat_images, labels, test_size = 0.2)
            train_images, test_images, train_labels, test_labels = train_test_split(images, labels, test_size = 0.2)
            test_images, val_images, test_labels, val_labels = train_test_split(test_images, test_labels, test_size = 0.5)
    
            split_list=[ train_images, test_images, train_labels, test_labels, val_images, val_labels]
            flat_split_list = [flat_train_images, flat_test_images, flat_train_labels, flat_test_labels]
            logger.info("Spliting data into test,train and validation ")
    
            return split_list, flat_split_list
        except Exception as e:
            raise e
    
    def save_split_data(self,split_list,flat_split_list):
        try:
            os.makedirs(self.config.preprocessed_data_path, exist_ok=True)
            split_list_path=Path(f"{self.config.preprocessed_data_path}/split_list.npy")
            flat_split_list_path=Path(f"{self.config.preprocessed_data_path}/flat_split_list.npy")
            np.save(split_list_path, np.array(split_list, dtype=object),allow_pickle=True)
            np.save(flat_split_list_path, np.array(flat_split_list, dtype=object),allow_pickle=True)
            logger.info(f"Saving test,train and validation as numpy array at {split_list_path} and {flat_split_list_path}")
        except Exception as e:
            raise e
        
        
    



