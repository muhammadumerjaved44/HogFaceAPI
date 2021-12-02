# -*- coding: utf-8 -*-
import os
import cv2
import glob
from matplotlib import pyplot as plt
from skimage import feature
from sklearn.svm import LinearSVC
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
import seaborn as sns
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import numpy as np
import pickle

from sklearn import preprocessing

import pathlib


def compute_hog(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (128, 256))

    hog_desc = feature.hog(image, orientations=9, pixels_per_cell=(8, 8),
        cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2-Hys')
    return hog_desc

def make_dataset(folder_path_list):
    images_list = []
    hog_list = []
    class_name_list = []
    for folder_path in folder_path_list:
        images_set = glob.glob(f'{folder_path}/*open.pgm')
        images_list.extend(images_set)
    for image_path in images_list:
        class_name_list.append(os.path.dirname(image_path).split('\\')[-1])
        hog_list.append(compute_hog(image_path))
    save_obj(class_name_list, "class_list")
    return hog_list, class_name_list


def train_model(hog_list, class_name_list):
    # hog_list = data_list
    label_encoder = preprocessing.LabelEncoder()
    features_df = pd.DataFrame(hog_list)
    class_df = pd.DataFrame(class_name_list, columns={'label'})
    class_df =  pd.DataFrame(label_encoder.fit_transform(class_df['label']), columns={'label'})
    svm_model = LinearSVC()
    x_train , x_test , y_train , y_test = train_test_split(features_df,
                                                       class_df,
                                                       random_state= 0,
                                                       test_size= 0.2,
                                                       stratify=class_df)
    svm_model.fit(x_train , y_train)
    save_obj(svm_model, "svm_model")
    return svm_model

def plot_roc(x_train , x_test , y_train , y_test, svm_model):
     ## roc code
    y_new = y_test.loc[(y_test['label'] == 4) | (y_test['label'] == 15)]
    y_new = np.where(y_new==4, 1, -1)
    x_new =  x_test.loc[(y_test['label'] == 4) | (y_test['label'] == 15)]

    lr_probs = svm_model._predict_proba_lr(x_new)

    lr_probs = lr_probs[:,1]
    ns_probs = [0 for _ in range(len(y_new))]

    ns_auc = roc_auc_score(y_new, ns_probs)
    lr_auc = roc_auc_score(y_new, lr_probs)
    # summarize scores
    print('No Skill: ROC AUC=%.3f' % (ns_auc))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))
    # calculate roc curves
    ns_fpr, ns_tpr, _ = roc_curve(y_new, ns_probs)
    lr_fpr, lr_tpr, _ = roc_curve(y_new, lr_probs)
    # plot the roc curve for the model
    plt.plot(ns_fpr, ns_tpr, linestyle='--', label='No Skill')
    plt.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    # axis labels
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    # show the legend
    plt.legend()
    # show the plot
    plt.show()


    y_pred = svm_model.predict(x_test)
    #print(metrics.accuracy_score(y_test,y_pred))

    cm = metrics.confusion_matrix(y_test, y_pred)
    sns.heatmap(cm , annot= True)

    return svm_model

def save_obj (obj, filename):
    filename = filename
    outfile =  open(filename,'wb')
    pickle.dump(obj,outfile)
    outfile.close()

def load_obj (filename):
    infile = open(filename,'rb')
    obj = pickle.load(infile)
    infile.close()

    return obj

def load_class_list(value):
    class_list = sorted(list(set(load_obj("class_list"))))
    class_dict = {f"{i}": x for i, x in enumerate(class_list)}
    return class_dict[f'{value}']


def single_image_predict(path):
    svm = load_obj("svm_model")
    path = pathlib.Path(path)
    if path.exists():
        x = compute_hog(str(path.absolute()))
        x = x.reshape(1 , -1)
        pridict = svm.predict(x)
        label = load_class_list(pridict.tolist()[0])
        return label






if __name__ == "__main__":

    # folder_path_list = glob.glob('faces/**/**')
    # hog_list, class_name_list = make_dataset(folder_path_list)

    # svm_model =train_model(hog_list, class_name_list)

    path = "faces/boland/C/boland_left_angry_open.pgm"
    result = single_image_predict(path)

