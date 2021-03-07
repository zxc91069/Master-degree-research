import random
import os
import cv2
import pandas as pd
import numpy as np
import keras
from keras.engine.saving import load_model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from tqdm import tqdm_notebook
from keras.utils import multi_gpu_model
from keras.models import Model
from keras.layers import Dense,Flatten
from keras.callbacks import ModelCheckpoint
from keras.layers import Dropout,Input
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
import tensorflow as tf
from xFunction.xPredict.xconfusion_matrix import plot_confusion_matrix
import time
from keras import backend as K
import sys


def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


def CreatCSVandWrite(CSVPathAndName,localtime, imgdataname, Epoch, Test_size, Images, lr, layer1, layer2, layer3,
                     layer4, layer5, layer6, layer7, layer8, layer9, Dense, dropout):
    #global localtime
    if os.path.exists(CSVPathAndName) == False:
        df = pd.DataFrame({"DATE" : [localtime],
                           "ImageDataName" : [imgdataname],
                           "Epoch": [Epoch],
                           "Test_size": [Test_size],
                           "Images": [Images],
                           "lr": [lr],
                           "Layer1": [layer1],
                           "Layer2": [layer2],
                           "Layer3": [layer3],
                           "Layer4": [layer4],
                           "Layer5": [layer5],
                           "Layer6": [layer6],
                           "Layer7": [layer7],
                           "Layer8": [layer8],
                           "Layer9": [layer9],
                           "Dense": [Dense],
                           "Dropout": [dropout],
                           "TrainingAcc": [""], #19
                           "TestingAcc":[""],
                           "AllDataAcc": [""],
                           "Predict_O_first": [""],
                           "Predict_RST_first": [""],
                           "Predict_N_first": [""],
                           "Predict_O_last": [""],
                           "Predict_RST_last": [""],
                           "Predict_N_last": [""],
                           "Predict_O_final": [""],
                           "Predict_RST_final": [""],
                           "Predict_N_final": [""]})
        df.to_csv(CSVPathAndName, index=False)
        print("新增CSV檔於" + CSVPathAndName)

    else:
        print("Write ACC to CSV")
        OpenCSV = pd.read_csv(CSVPathAndName)
        CSVrows = OpenCSV.shape[0]
        lis = [ localtime, IMAGEPATH.split("\\")[2], Epoch, Test_size, Images, lr,
                 str(Layer1_filter_each)+"(" + str(Layer1_filter_size) + "," + str(Layer1_filter_size) + ")" ,
                 str(Layer2_filter_each)+"(" + str(Layer2_filter_size) + "," + str(Layer2_filter_size) + ")" ,
                 "(" + str(Layer3_Maxpooling) + "," + str(Layer3_Maxpooling) + ")",
                 str(Layer4_filter_each)+"(" + str(Layer4_filter_size) + "," + str(Layer4_filter_size) + ")" ,
                 str(Layer5_filter_each)+"(" + str(Layer5_filter_size) + "," + str(Layer5_filter_size) + ")" ,
                 "(" + str(Layer6_Maxpooling) + "," + str(Layer6_Maxpooling) + ")",
                 str(Layer7_filter_each)+"(" + str(Layer7_filter_size) + "," + str(Layer7_filter_size) + ")" ,
                 str(Layer8_filter_each)+"(" + str(Layer8_filter_size) + "," + str(Layer8_filter_size) + ")" ,
                 "(" + str(Layer9_Maxpooling) + "," + str(Layer9_Maxpooling) + ")",
                 Dense,dropout,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan]

        print(OpenCSV.shape[0])
        OpenCSV.loc[len(OpenCSV)] = lis
        OpenCSV.to_csv(CSVPathAndName, index=False)


def load_image(path,id,img_size):
    image = cv2.imread(path+id)
    new_image=cv2.resize(image,(img_size,img_size))

    return new_image,id

def split_data(X,X_names,Y,randomcode=42):

    X_tr, X_tst, Y_tr, Y_tst = train_test_split(X, Y, test_size=testsize, random_state=randomcode)
    X_tr_names,X_tst_names,y_tr_name,y_tst_name = train_test_split(X_names, Y, test_size=testsize, random_state=randomcode)
    X_tr = X_tr.astype("float32")
    X_tst = X_tst.astype("float32")

    return X_tr, X_tst, Y_tr, Y_tst,X_tr_names,X_tst_names

def prepare_data(fileName, imagePath, img_size):
    train_df = pd.read_csv(fileName)
    ids = train_df['ID'].values

    X = []
    X_names = []
    Y = []
    for id in tqdm_notebook(ids):
        try:
            im,imgName = load_image(imagePath+"/", id,img_size)
            X.append(im)
            X_names.append(imgName)
            ads = train_df[train_df['ID'] == id]['Label'].values[0]
            Y.append(ads)
        except Exception as e:
            print(e)
            pass

    label_encoder = LabelEncoder()
    Y_origin = np.array(Y)
    integer_encoded = label_encoder.fit_transform(Y_origin)
    onehot_encoder = OneHotEncoder(sparse=False, categories='auto')
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)


    X = np.array(X)
    X_names = np.array(X_names)
    X = X.astype("float32")
    X /= 255


    return X,X_names,onehot_encoded,onehot_encoder,Y_origin


def build_model(nb_classes,lr):
    # cnn mode
    # The structure and parameters of Cnn can be changed to what you want.

    model = Sequential()

    model.add(Conv2D(16, (3, 3), padding='same',
                    input_shape=(img_size, img_size, 3), activation='relu'))#1

    model.add(Conv2D(8, (5, 5), activation='relu', padding='same')) #2

    model.add(MaxPooling2D(pool_size=(2, 2g))) #3

    model.add(Conv2D(4, (7, 7), padding='same', activation='relu')) #4

    model.add(Conv2D(8, (3, 3), activation='relu')) #5

    model.add(MaxPooling2D(pool_size=(2, 2)))#6

    model.add(Conv2D(8, (5, 5), activation='relu')) #7

    model.add(Conv2D(8, (7, 7), activation='relu')) #8

    model.add(MaxPooling2D(pool_size=(2, 2))) #9

    model.add(Flatten())

    model.add((Dense(512)))

    model.add(Dropout(0.4))

    model.add(Dense(nb_classes, activation='softmax'))

    opt = Adam(lr=lr)  # 0.0001

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['acc'])

    print(model.summary())

    return model

def model_predict(modelName,data,dataName,label,onehot_encoder,title,showPlot):
    opt = Adam(lr=lr) #0.0001
    classes = ['0','1','2','3','4','5']  # It depends on your dataset. If your dataset has 3 classes, then change classes to ['0','1','2']
    model = load_model(modelName)
    model.compile(loss='categorical_crossentropy',
                  optimizer=opt,
                  metrics=['acc'])

    result_p = model.predict(data)
    result_class = result_p.argmax(axis=1)
    label = onehot_encoder.inverse_transform(label)
    label = label.reshape(label.shape[0])

    sb = result_class - label
    ind = np.where(sb != 0)

    # failFiles = dataName[ind[0]]

    for i in range(len(ind[0])):
        print("file : "+dataName[ind[0][i]]+" distributed ")
        for pr in range(len(result_p[i])):
            print("Class " + classes[pr] + " :  " + str(result_p[i][pr]))
        print("========================")

    print("<!--             Wrongly predicted file : ",dataName[ind],"                --!>")

    acc_score = round(accuracy_score(label, result_class, normalize=True),3)
    print(acc_score)

    plot_confusion_matrix(y_true=label, y_pred=result_class, classes=classes, acc_score=acc_score, normalize=False,
                          title=title, plt_ornot=True)
    if showPlot == 1:
        plot_confusion_matrix(y_true=label,y_pred=result_class,classes=classes,acc_score=acc_score,normalize=False,
                              title=title,plt_ornot=True)
    else:
        print("Do Not Show Plot")

    return acc_score

def model_training(model,savedModelName,X_tr,Y_tr,X_tst,Y_tst,batch_size,nb_epoch):
    filepath = saveMODELpath + IMAGEPATH.split("\\")[2] + "_" +'tessize' + str(TESTSIZE_parameter).replace(".", "")
    firstmodel =  filepath +"_first.hdf5"
    lastmodel = filepath + "_last.hdf5"

    checkpointfirst = ModelCheckpoint(firstmodel, monitor='val_acc', verbose=1, save_best_only=True, mode='max_first')
    checkpointlast = ModelCheckpoint(lastmodel, monitor='val_acc', verbose=1, save_best_only=True, mode='max_last')
    #checkpoint2 = EarlyStopping(monitor='val_acc', mode='max', baseline=0.01, patience=10, verbose=2)

    callbacks_list = [checkpointfirst,checkpointlast]

    if load_weights is None or load_weights == "":
        print("Without pre-trained weights...")
    else:
        print("Load weights "+load_weights+"...")
        model.load_weights(load_weights)

    history = model.fit(X_tr, Y_tr, batch_size=batch_size,
                                           epochs=nb_epoch,
                                           validation_data=(X_tst, Y_tst),
                                           callbacks=callbacks_list)
    #
    model.save(filepath + '_final.hdf5')
    model.save_weights(filepath + '_final_weights.hdf5')
    plot_history(history)
    model_predict(opt,'M.hdf5',X_tr,X_tr_names,Y_tr,onehot_encoder)

    return history

def plot_history(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epoch")
    plt.legend(['train', 'validation'], loc="upper left")
    plt.show()