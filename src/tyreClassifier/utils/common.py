import os
import pickle
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,ConfusionMatrixDisplay
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise e
    


def evaluate_models(flat_test_images,flat_test_labels,test_images,test_labels,model_name,model):
    print(model)
    try:
        if model_name in ('KNN', 'RandomForest'):
            predictions = model.predict(flat_test_images)
            accuracy_test = accuracy_score(flat_test_labels, predictions)
            classification_report1 = classification_report(flat_test_labels,predictions, target_names=['Defective(Class 0)', 'Good(Class 1)'])
            cm = confusion_matrix(flat_test_labels, predictions)

        else:
            predictions = model.predict(test_images)
            accuracy_test = accuracy_score(test_labels, np.argmax(predictions,axis=1))
            classification_report1 = classification_report(test_labels, np.argmax(predictions,axis=1), target_names=['Defective(Class 0)', 'Good(Class 1)'])
            cm = confusion_matrix(test_labels, np.argmax(predictions,axis=1))


        cm_display = ConfusionMatrixDisplay(confusion_matrix = cm, display_labels = ['Defective', 'Good'])

        cm_display.plot()

        '''plt.figure(figsize = (7,7))
        sns.heatmap(cm, cmap = 'Blues', linecolor = 'black', 
                    linewidth = 4, annot = True, fmt = '', xticklabels = 'Defective(Class 0)', yticklabels = 'Good(Class 1)')'''
        os.makedirs(Path("visuals/Confusion_Matrix"), exist_ok=True)
        img_path = f"visuals/Confusion_Matrix/{model_name}_ConfusionMatrix.png"
        plt.savefig(Path(img_path))
        plt.close()


        return accuracy_test, classification_report1

    except Exception as e:
        raise e

        
    

    
def plot_accuracy_loss_chart(history,model_name,epoch):
    try:
        epochs = [i for i in range(epoch)]
        fig , ax = plt.subplots(1,2)
        train_acc = history.history['accuracy']
        train_loss = history.history['loss']
        val_acc = history.history['val_accuracy']
        val_loss = history.history['val_loss']
        fig.set_size_inches(20,10)
        ax[0].plot(epochs , train_acc , 'go-' , label = 'Training Accuracy')
        ax[0].plot(epochs , val_acc , 'ro-' , label = 'Validation Accuracy')
        ax[0].set_title(f'Training & Validation Accuracy for {model_name}')
        ax[0].legend()
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("Accuracy")

        ax[1].plot(epochs , train_loss , 'g-o' , label = 'Training Loss')
        ax[1].plot(epochs , val_loss , 'r-o' , label = 'Validation Loss')
        ax[1].set_title('Training & Validation Loss')
        ax[1].legend()
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("Training & Validation Loss")
        title = ax[0].get_title()
        img_path = f"visuals/{title}.png"
        plt.savefig(Path(img_path))
        plt.close()
    except Exception as e:
        raise e
    



def best_modelPlot(Models,Accuracy,best_model):
    try:
        x = np.arange(len(Models)) # the label locations
        width = 0.35 # the width of the bars
        fig, ax = plt.subplots()
        ax.set_ylabel('Accuracy')
        ax.set_title('Models')
        ax.set_xticks(x)
        ax.set_xticklabels(Models)

        pps = ax.bar(x - width/2, Accuracy, width, label='Accuracy')
        for p in pps:
            height = round(p.get_height(),2)
            ax.annotate('{}'.format(height),
                xy=(p.get_x() + p.get_width() / 2, height),
                xytext=(0, 3), # 3 points vertical offset
                textcoords="offset points",
                ha='center', va='bottom')
        img_path = f"visuals/best_model-{best_model}.png"
        plt.savefig(Path(img_path))
        plt.close()

    except Exception as e:
        raise e