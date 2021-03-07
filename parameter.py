from Model_setting import *
import tensorflow as tf
import time


#parameter
img_size = 200
nb_classes = 6
batch_size = 32
nb_epoch= 1000
TESTSIZE_parameter = 0.7
load_weights = ""
lr = 0.0001
crossValCount = 10
ExecuteCode = 0
# 0 Not executed 1 Training model 2 Training and predict 3 only predict 4 CrossValidation  6 loop training 7 cross validation 8 CNN + SVM 9 CNN + RF

#Time
LOCALTIME = time.strftime("%Y.%m.%d.%H.%M", time.localtime())
local_day_time = str(time.strftime("%Y%m%d", time.localtime()))
local_hour_time = str(time.strftime("%H%M%S", time.localtime()))

#acccsv Name
CSVPATH = "D:/ALLAccuracy.csv"

# MODEL存檔

#ExecutCode = 6 迴圈訓練MODEL
IMAGEPATH = "D:/dataset1"
imgcsvname = "D:/dataset1.csv"
testsize = 0.1
savedModelName = 'dataset1.hdf5'
predict_modelName = "D:dataset1.hdf5"

#GPU setting
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8) # Control GPU VRAM consumption
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


X,X_names,Y,onehot_encoder,Y_origin= prepare_data(fileName=imgcsvname,imagePath=IMAGEPATH
                                        ,img_size=img_size)
X_tr, X_tst, Y_tr, Y_tst,X_tr_names,X_tst_names = split_data(X,X_names,Y)



if ExecuteCode == 0:
    print("不執行程式碼")
elif ExecuteCode == 1:
    print("只執行訓練階段")
    model = build_model(nb_classes=nb_classes,lr=lr)
    history = model_training(model=model, savedModelName=savedModelName
                                  , X_tr=X_tr, Y_tr=Y_tr, X_tst=X_tst, Y_tst=Y_tst
                                  , batch_size=batch_size, nb_epoch=nb_epoch)
elif ExecuteCode == 2:
    print("執行訓練與預測階段")
    #存MODEL
    createFolder(saveMODELpath)
    model = build_model(nb_classes=nb_classes,lr=lr)
    history = model_training(model=model, savedModelName=savedModelName
                                  , X_tr=X_tr, Y_tr=Y_tr, X_tst=X_tst, Y_tst=Y_tst
                                  , batch_size=batch_size, nb_epoch=nb_epoch)

    #修改  宣告參數
    TrainACC = model_predict(predict_modelName, X_tr, X_tr_names, Y_tr, onehot_encoder, title="Training Data", showPlot=0)
    TestACC = model_predict(predict_modelName, X_tst, X_tst_names, Y_tst, onehot_encoder, title="Testing Data", showPlot=0)
    ALLDATAACC = model_predict(predict_modelName, X, X_names, Y, onehot_encoder, title="All Data", showPlot=0)

    #各層寫入CSV
    CreatCSVandWrite(CSVPATH, LOCALTIME, IMAGEPATH.split("\\")[2], nb_epoch, TESTSIZE_parameter, Images= Images, lr=lr,
                     layer1=str(Layer1_filter_each) + "(" + str(Layer1_filter_size) + "," + str(
                         Layer1_filter_size) + ")",
                     layer2=str(Layer2_filter_each) + "(" + str(Layer2_filter_size) + "," + str(
                         Layer2_filter_size) + ")",
                     layer3="(" + str(Layer3_Maxpooling) + "," + str(Layer3_Maxpooling) + ")",
                     layer4=str(Layer4_filter_each) + "(" + str(Layer4_filter_size) + "," + str(
                         Layer4_filter_size) + ")",
                     layer5=str(Layer5_filter_each) + "(" + str(Layer5_filter_size) + "," + str(
                         Layer5_filter_size) + ")",
                     layer6="(" + str(Layer6_Maxpooling) + "," + str(Layer6_Maxpooling) + ")",
                     layer7=str(Layer7_filter_each) + "(" + str(Layer7_filter_size) + "," + str(
                         Layer7_filter_size) + ")",
                     layer8=str(Layer8_filter_each) + "(" + str(Layer8_filter_size) + "," + str(
                         Layer8_filter_size) + ")",
                     layer9="(" + str(Layer9_Maxpooling) + "," + str(Layer9_Maxpooling) + ")",
                     Dense=DENSE, dropout=Drop_Out
                     )

elif ExecuteCode ==3:
    print("執行預測階段")
    model_predict(predict_modelName, X_tr, X_tr_names, Y_tr, onehot_encoder,title="Training Data")
    model_predict(predict_modelName, X_tst, X_tst_names, Y_tst, onehot_encoder,title="Testing Data")
    model_predict(predict_modelName, X, X_names, Y, onehot_encoder,title="All Data")


elif ExecuteCode ==4:
    print("Cross validation 階段     執行 ",str(crossValCount)+" 次")

    clf = KerasClassifier(build_fn=build_model)
    p = {"nb_classes": nb_classes, "lr": lr, "epochs": nb_epoch,
         "batch_size": batch_size}
    clf.set_params(**p)

    kfold = StratifiedKFold(n_splits=crossValCount,  )
    label1D = Y_origin.ravel()
    results = cross_val_score(clf, X, label1D, cv=kfold, scoring="accuracy")
    print("")
    print("Validation accuracy")
    print("")
    print(results)
    print("Average Cross validation accuracy : ", results.mean())

elif ExecuteCode ==5:

    model_predict(predict_modelName_keyin, X, X_names, Y, onehot_encoder,title="Others Data",showPlot= 1 )

elif ExecuteCode ==6:
    #------------------請搭配Batch迴圈訓練.bat 檔案進行迴圈訓練---------------------------------
    #使用方法1.輸入檔案位置硬碟 2.輸入資料夾路徑 3.輸入訓練py檔名稱 4.輸入迴圈次數
    # i 值 串接由batch檔所產生1-?值 自行輸入
    i = int(sys.argv[1])
    #-------------------------------------------------------------------------------------------

    IMAGEPATH = str(parametercsv.iloc[i,0])
    TESTSIZE_parameter = parametercsv.iloc[i,2]
    testsize =  parametercsv.iloc[i,2]
    imgcsvname = str(parametercsv.iloc[i,1])
    #save model
    saveMODELpath = "./modelsave/" + local_day_time + "/" + IMAGEPATH.split("\\")[2] + "/"
    savedModelName = saveMODELpath + IMAGEPATH.split("\\")[2] + "_" + 'testzie' + str(TESTSIZE_parameter).replace(
        ".", "") + ".hdf5"  # 影像名稱
    predict_modelName = saveMODELpath + IMAGEPATH.split("\\")[2] + "_" + 'testzie' + str(
        TESTSIZE_parameter).replace(".", "") + "_final" + ".hdf5"
    predict_modelName_first = saveMODELpath + IMAGEPATH.split("\\")[2] + "_" + 'testzie' + str(
        TESTSIZE_parameter).replace(".", "") + "_first" + ".hdf5"
    predict_modelName_last = saveMODELpath + IMAGEPATH.split("\\")[2] + "_" + 'testzie' + str(
        TESTSIZE_parameter).replace(".", "") + "_last" + ".hdf5"
    predict_modelName_final = saveMODELpath + IMAGEPATH.split("\\")[2] + "_" + 'testzie' + str(
        TESTSIZE_parameter).replace(".", "") + "_final" + ".hdf5"

    X, X_names, Y, onehot_encoder, Y_origin = prepare_data(fileName=imgcsvname, imagePath=IMAGEPATH
                                                          , img_size=img_size)
    X_tr, X_tst, Y_tr, Y_tst, X_tr_names, X_tst_names = split_data(X, X_names, Y)

    createFolder(saveMODELpath)

    model = build_model(nb_classes=nb_classes, lr=lr)
    history = model_training(model=model, savedModelName=savedModelName
                              , X_tr=X_tr, Y_tr=Y_tr, X_tst=X_tst, Y_tst=Y_tst
                              , batch_size=batch_size, nb_epoch=nb_epoch)

    countcsvimage = pd.read_csv(imgcsvname)
    id_name = countcsvimage['ID'].values
    imagenamecount = id_name.tolist()



    #count image
    splitimgname = []
    for j in range(len(imagenamecount)):
        try:
            splitimgname.append(imagenamecount[j].split("_")[2])
        except IndexError:
            splitimgname = []


    RST = splitimgname.count("translation") + splitimgname.count("scale") + splitimgname.count("rotation")
    Noise = splitimgname.count("SaltAndPepper")
    Origin = len(imagenamecount) - RST - Noise

    #寫入CSV
    CreatCSVandWrite(CSVPATH, LOCALTIME, IMAGEPATH.split("\\")[2], nb_epoch, TESTSIZE_parameter, Origin=Origin, RST=RST,
                     Noise=Noise, lr=lr,
                     layer1=str(Layer1_filter_each) + "(" + str(Layer1_filter_size) + "," + str(
                         Layer1_filter_size) + ")",
                     layer2=str(Layer2_filter_each) + "(" + str(Layer2_filter_size) + "," + str(
                         Layer2_filter_size) + ")",
                     layer3="(" + str(Layer3_Maxpooling) + "," + str(Layer3_Maxpooling) + ")",
                     layer4=str(Layer4_filter_each) + "(" + str(Layer4_filter_size) + "," + str(
                         Layer4_filter_size) + ")",
                     layer5=str(Layer5_filter_each) + "(" + str(Layer5_filter_size) + "," + str(
                         Layer5_filter_size) + ")",
                     layer6="(" + str(Layer6_Maxpooling) + "," + str(Layer6_Maxpooling) + ")",
                     layer7=str(Layer7_filter_each) + "(" + str(Layer7_filter_size) + "," + str(
                         Layer7_filter_size) + ")",
                     layer8=str(Layer8_filter_each) + "(" + str(Layer8_filter_size) + "," + str(
                         Layer8_filter_size) + ")",
                     layer9="(" + str(Layer9_Maxpooling) + "," + str(Layer9_Maxpooling) + ")",
                     Dense=DENSE, dropout=Drop_Out
                     )
    ACCCSV = pd.read_csv(CSVPATH)

    ACCCSV.iat[i,19] = model_predict(predict_modelName, X_tr, X_tr_names, Y_tr, onehot_encoder, title="Training Data", showPlot=0)
    ACCCSV.iat[i,20] = model_predict(predict_modelName, X_tst, X_tst_names, Y_tst, onehot_encoder, title="Testing Data", showPlot=0)
    ACCCSV.iat[i,21] = model_predict(predict_modelName, X, X_names, Y, onehot_encoder, title="All Data", showPlot=0)
    K.clear_session()

    IMAGEPATH = str(parametercsv.iloc[i, 3])  #dataset1
    imgcsvname = str(parametercsv.iloc[i, 4])
    X, X_names, Y, onehot_encoder, Y_origin = prepare_data(fileName=imgcsvname, imagePath=IMAGEPATH
                                                           , img_size=img_size)
    X_tr, X_tst, Y_tr, Y_tst, X_tr_names, X_tst_names = split_data(X, X_names, Y)
    ACCCSV.iat[i, 22] = model_predict(predict_modelName_first, X, X_names, Y, onehot_encoder, title="All Data", showPlot=0)
    ACCCSV.iat[i, 25] = model_predict(predict_modelName_first, X, X_names, Y, onehot_encoder, title="All Data", showPlot=0)
    ACCCSV.iat[i, 28] = model_predict(predict_modelName_first, X, X_names, Y, onehot_encoder, title="All Data", showPlot=0)

    K.clear_session()

    IMAGEPATH = str(parametercsv.iloc[i, 5]) #dataset2
    imgcsvname = str(parametercsv.iloc[i, 6])
    X, X_names, Y, onehot_encoder, Y_origin = prepare_data(fileName=imgcsvname, imagePath=IMAGEPATH
                                                           , img_size=img_size)
    X_tr, X_tst, Y_tr, Y_tst, X_tr_names, X_tst_names = split_data(X, X_names, Y)
    ACCCSV.iat[i, 23] = model_predict(predict_modelName_last, X, X_names, Y, onehot_encoder, title="All Data", showPlot=0)
    ACCCSV.iat[i, 26] = model_predict(predict_modelName_last, X, X_names, Y, onehot_encoder, title="All Data", showPlot=0)
    ACCCSV.iat[i, 29] = model_predict(predict_modelName_last, X, X_names, Y, onehot_encoder, title="All Data", showPlot=0)

    IMAGEPATH = str(parametercsv.iloc[i, 7])  # dataset3
    imgcsvname = str(parametercsv.iloc[i, 8])

    X, X_names, Y, onehot_encoder, Y_origin = prepare_data(fileName=imgcsvname, imagePath=IMAGEPATH
                                                           , img_size=img_size)
    X_tr, X_tst, Y_tr, Y_tst, X_tr_names, X_tst_names = split_data(X, X_names, Y)
    ACCCSV.iat[i, 24] = model_predict(predict_modelName_final, X, X_names, Y, onehot_encoder, title="All Data", showPlot=0)
    ACCCSV.iat[i, 27] = model_predict(predict_modelName_final, X, X_names, Y, onehot_encoder, title="All Data", showPlot=0)
    ACCCSV.iat[i, 30] = model_predict(predict_modelName_final, X, X_names, Y, onehot_encoder, title="All Data", showPlot=0)

    ACCCSV.to_csv(CSVPATH,index=False)
    K.clear_session()


elif ExecuteCode == 8:                   #SVM

    opt = Adam(lr=lr)#0.0001

    classes = ['0','1','2','3','4','5']
    model = load_model(predict_modelName)

    m = Model(input = model.layers[0].input, output = model.layers[10].output)                   # get model Flatten Layer value
    model.summary()

    ConvX_tr = (m.predict(X_tr))
    ConvX_tst = (m.predict(X_tst))
    ConvY_trlabel = onehot_encoder.inverse_transform(Y_tr)
    ConvY_tstlabel = onehot_encoder.inverse_transform(Y_tst)

    ravel_ConvY_trlabel = ConvY_trlabel.ravel()
    ravel_ConvY_tstlabel = ConvY_tstlabel.ravel()


    model = SVC(C=1000, cache_size=200, class_weight=None, coef0=0.0,decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
                max_iter=-1, probability=False, random_state=None, shrinking=True,
                tol=0.001, verbose=False)

    # ---------------------------
    # grid serch parameter
    # ---------------------------
    param_grid = {'C': [0.1, 0.5, 1, 5, 10, 15, 20, 100, 150, 175, 200, 250, 1000],
                  'gamma': [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 3, 4, 5, 10], }
    clf = GridSearchCV(SVC(kernel='rbf'), param_grid)
    clf.fit(ConvX_tr, ravel_ConvY_trlabel)



    print('C =', best_C)
    print('gamma =', best_gamma)
    print('-' * 60)

    model = SVC(kernel='rbf', gamma=best_gamma, C=best_C)
    model.fit(ConvX_tr, ravel_ConvY_trlabel)


    predictions_tst = model.predict(ConvX_tst)
    predictions_tr = model.predict(ConvX_tr)

    print(confusion_matrix(ConvY_tstlabel, predictions_tst))
    print('\n')
    print(classification_report(ConvY_tstlabel, predictions_tst))

    print(confusion_matrix(ConvY_trlabel, predictions_tr))
    print('\n')
    print(classification_report(ConvY_trlabel, predictions_tr))


elif ExecuteCode == 9:      #RF
    opt = Adam(lr=lr)#0.0001

    classes = ['0','1','2','3','4','5']
    model = load_model(predict_modelName)

    m = Model(input = model.layers[0].input, output = model.layers[10].output)           # get model Flatten Layer value
    model.summary()

    ConvX_tr = (m.predict(X_tr))
    ConvX_tst = (m.predict(X_tst))
    ConvY_trlabel = onehot_encoder.inverse_transform(Y_tr)
    ConvY_tstlabel = onehot_encoder.inverse_transform(Y_tst)

    ravel_ConvY_trlabel = ConvY_trlabel.ravel()
    ravel_ConvY_tstlabel = ConvY_tstlabel.ravel()

    bag = RandomForestClassifier(criterion="entropy", n_estimators=100000, random_state=1)            # RF parameter

    bag.fit(ConvX_tr, ravel_ConvY_trlabel)

    Ypred = bag.predict(ConvX_tst)

    print(classification_report(Ypred, ravel_ConvY_tstlabel))







