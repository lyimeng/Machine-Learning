def train(info, BASE_CSV_PATH, TRAIN, TEST, MODEL_PATH, LE_PATH):
    # import the necessary packages·
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import classification_report
    from pyimagesearch import config
    import numpy as np
    import pickle
    import os


    def load_data_split(splitPath):
        # initialize the data and labels
        data = []
        labels = []
        
        # loop over the rows in the data split file
        for row in open(splitPath):
            count = len(data)
            # extract the class label and features from the row
            row = row.strip().split(",")
            label = row[0]
            features = np.array(row[1:], dtype="float16")

            # update the data and label lists
            data.append(features)
            labels.append(label)

        # convert the data and labels to NumPy arrays
        data = np.array(data)
        labels = np.array(labels)

        # np.save("data_test.npy", data)
        # np.save("labels_test.npy", labels)
     
        # return a tuple of the data and labels
        return (data, labels)



    # derive the paths to the training and testing CSV files
    trainingPath = os.path.sep.join([BASE_CSV_PATH,
        "{}.csv".format(TRAIN)])
    testingPath = os.path.sep.join([BASE_CSV_PATH,
        "{}.csv".format(TEST)])
     
    # load the data from disk
    print("[INFO] loading train data...")
    (trainX, trainY) = load_data_split(trainingPath)
    print("[INFO] loading test data...")
    (testX, testY) = load_data_split(testingPath)

    # load the label encoder from disk
    le = pickle.loads(open(LE_PATH, "rb").read())

    # train the model
    print("[INFO] training model...")
    model = LogisticRegression(solver="lbfgs", multi_class="auto")



    model.fit(trainX, trainY)

    # evaluate the model
    print("[INFO] evaluating...")
    preds = model.predict(testX)



    # print(classification_report(testY, preds, target_names=le.classes_))
    result = classification_report(testY, preds, target_names=le.classes_, digits= 4)
    text_file = open("output//result.txt", "w")
    text_file.write(str(info) + "\n")
    text_file.write(result)
    text_file.close()
    # serialize the model to disk
    # print("[INFO] saving model...")

    print(result)
    # np.save(BASE_CSV_PATH + '\\label.npy',val)
    # np.save(BASE_CSV_PATH+ '\\picture.npy', valX)
    with open(MODEL_PATH, "wb") as f:
        f.write(pickle.dumps(model))

# info = "320_ 25"
# BASE_CSV_PATH = "F:\\College\\MachineLearning\\ShortDistanceProfile\\images\\heat_map\\wl_320\\wl_320_ol_25\\output"
# TRAIN = "training"
# TEST = "evaluation"
# VAL = "validation"
# MODEL_PATH = "F:\\College\\MachineLearning\\ShortDistanceProfile\\images\\heat_map\\wl_320\\wl_320_ol_25\\output\\model.cpickle"
# LE_PATH = "F:\\College\\MachineLearning\\ShortDistanceProfile\\images\\heat_map\\wl_320\\wl_320_ol_25\\output\\le.cpickle"
# train(info, BASE_CSV_PATH, TRAIN, TEST, MODEL_PATH, LE_PATH)