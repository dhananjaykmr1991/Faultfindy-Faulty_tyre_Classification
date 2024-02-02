import os
from pathlib import Path
from keras.models import Model
from keras.layers import Flatten, Dense,Dropout,GlobalAveragePooling2D
from keras.applications import VGG16,ResNet50
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
import numpy as np
from dataclasses import dataclass
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
import pickle 

@dataclass(frozen=True)
class ModelTrainingConfig:
    split_data_path = r'artifacts\data_Preprocessing\split_list.npy'
    flat_split_data_path = r'artifacts\data_Preprocessing\flat_split_list.npy'
    split_list = np.load(Path(split_data_path), allow_pickle=True)
    flat_split_list = np.load(Path(flat_split_data_path), allow_pickle=True)
    flat_train_images, flat_test_images, flat_train_labels, flat_test_labels = flat_split_list
    train_images, test_images, train_labels, test_labels, val_images, val_labels = split_list
    IMAGE_SIZE = [224, 224]
    num_classes = 2

class ModelSelction:
    def __init__(self,config:ModelTrainingConfig):
        self.config=config

    def model_VGG16(self):

        vgg = VGG16(input_shape = self.config.IMAGE_SIZE + [3], weights = 'imagenet', include_top = False)
        
        for layer in vgg.layers:
            layer.trainable = False

        x = Flatten()(vgg.output)
        x = Dense(self.config.num_classes, activation = 'softmax')(x)
        model = Model(inputs = vgg.input, outputs = x)
        model.compile(optimizer ='adam',loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        return model
    
    
    def model_resnet_50(self):

        resnet_50 = ResNet50(input_shape = self.config.IMAGE_SIZE + [3], weights = 'imagenet', include_top = False)
        
        for layer in resnet_50.layers:          
            layer.trainable = False

        num_classes=2
        x = resnet_50.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(512, activation='relu')(x) 
        x = Dropout(0.5)(x)
        x = Dense(256, activation='relu')(x) 
        x = Dropout(0.5)(x)
        x = Dense(128, activation='relu')(x) 
        x = Dropout(0.5)(x)
        x = Dense(64, activation='relu')(x) 
        x = Dropout(0.5)(x)
        predictions = Dense(self.config.num_classes, activation='softmax')(x)
        model = Model(inputs = resnet_50.input, outputs = predictions)
        model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

        return model
    
    def model_KNN(self):
        model = KNeighborsClassifier()

        kf=KFold(n_splits=10,shuffle=True,random_state=42)
        parameter={'n_neighbors': np.arange(30, 55, 2)}
        model=RandomizedSearchCV(estimator = model, param_distributions=parameter, cv=kf, verbose=1)
        return model
    
    def model_RandomForest(self):
        model = RandomForestClassifier(n_estimators=150) 
        param_dist = {"max_depth": [3,40], "max_features": [3, 50], "min_samples_split": [2, 15], 
                      "bootstrap": [True, False], "criterion": ["gini", "entropy"]} 
        model = RandomizedSearchCV(estimator = model, param_distributions = param_dist, n_iter = 100, 
                                   cv = 3, verbose=2, random_state=42, n_jobs = -1)
        
        return model



class ModelTraining:
    def __init__(self,config:ModelTrainingConfig):
        self.config=config

    def KNN_training(self,model):
        model.fit(self.config.flat_train_images,self.config.flat_train_labels)
        saved_model = pickle.dumps(model)  
        pickle.dump(model, open(Path('models\KNN.pkl'), 'wb'))

        return model
    
    def RandomForest_training(self,model):
        model.fit(self.config.flat_train_images,self.config.flat_train_labels)
        saved_model = pickle.dumps(model)  
        pickle.dump(model, open(Path('models\RandomForest.pkl'), 'wb'))

        return model



    def vgg16_training(self,model):
        checkpoint =ModelCheckpoint("models\vgg16.h5", save_best_only=True,save_weights_only=False)
        early_stopping =EarlyStopping(patience=5, restore_best_weights=True)
        learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_accuracy',patience = 2,verbose = 1,factor = 0.3, min_lr = 0.000001)
        model=model.fit(self.config.train_images,self.config.train_labels,
                                  epochs=10,validation_data=(self.config.val_images, self.config.val_labels),
                                 callbacks=[checkpoint,early_stopping,ReduceLROnPlateau]) 
     
        return model
    
    def resnet_50_training(self,model):
        checkpoint =ModelCheckpoint("models\resnet_50.h5", save_best_only=True,save_weights_only=False)
        early_stopping =EarlyStopping(patience=5, restore_best_weights=True)
        learning_rate_reduction = ReduceLROnPlateau(monitor = 'val_accuracy',patience = 2,verbose = 1,factor = 0.3, min_lr = 0.000001)
        model.fit(self.config.train_images,self.config.train_labels,epochs=10,
                  validation_data=(self.config.val_images, self.config.val_labels),
                  callbacks=[checkpoint,early_stopping,ReduceLROnPlateau])
        
        return model
    