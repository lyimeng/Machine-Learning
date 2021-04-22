def extract_features(TRAIN, TEST, BASE_PATH, BASE_CSV_PATH, BATCH_SIZE, LE_PATH):
    # import the necessary packages
    from sklearn.preprocessing import LabelEncoder
    from keras.applications import VGG16, ResNet50
    from keras.applications import imagenet_utils
    from keras.preprocessing.image import img_to_array
    from keras.preprocessing.image import load_img
    from pyimagesearch import config
    from imutils import paths
    import numpy as np
    import pickle
    import random
    import os
     
     # load the VGG16 network and initialize the label encoder
    print("[INFO] loading network...")
    model = VGG16(weights="imagenet", include_top=False)
    # model = ResNet50(weights="imagenet", include_top=False)
    le = None

    #loop over the data splits
    for split in (TRAIN, TEST):
    # for split in (TRAIN, VAL):
        # grab all image paths in the current split
        print("[INFO] processing '{} split'...".format(split))
        p = os.path.sep.join([BASE_PATH, split])
        imagePaths = list(paths.list_images(p))
     
        # randomly shuffle the image paths and then extract the class
        # labels from the file paths
        random.shuffle(imagePaths)
        labels = [p.split(os.path.sep)[-2] for p in imagePaths]
     
        # if the label encoder is None, create it
        if le is None:
            le = LabelEncoder()
            le.fit(labels)

        # open the output CSV file for writing
        csvPath = os.path.sep.join([BASE_CSV_PATH, "{}.csv".format(split)])

        # imageName = []
        with open(csvPath, "w") as csv:

            # loop over the images in batches
            for (b, i) in enumerate(range(0, len(imagePaths), BATCH_SIZE)):
                # extract the batch of images and labels, then initialize the
                # list of actual images that will be passed through the network
                # for feature extraction
                print("[INFO] processing batch {}/{}".format(b + 1, int(np.ceil(len(imagePaths) / float(BATCH_SIZE)))))
                batchPaths = imagePaths[i:i + BATCH_SIZE]
                batchLabels = le.transform(labels[i:i + BATCH_SIZE])
                batchImages = []

                # loop over the images and labels in the current batch
                for imagePath in batchPaths:
                    # load the input image using the Keras helper utility
                    # while ensuring the image is resized to 224x224 pixels
                    image = load_img(imagePath, target_size=(224, 224))
                    image = img_to_array(image)
                    # preprocess the image by (1) expanding the dimensions and
                    # (2) subtracting the mean RGB pixel intensity from the
                    # ImageNet dataset
                    image = np.expand_dims(image, axis=0)
                    image = imagenet_utils.preprocess_input(image)
        
                    # add the image to the batch
                    batchImages.append(image)

                    # imageName.append(imagePath)

                # pass the images through the network and use the outputs as
                # our actual features, then reshape the features into a
                # flattened volume
                batchImages = np.vstack(batchImages)
                features = model.predict(batchImages, batch_size=BATCH_SIZE)
                features = features.reshape((features.shape[0], 7 * 7 * 512))
                # features = features.reshape((features.shape[0], 7 * 7 * 2048))

                # loop over the class labels and extracted features
                for (label, vec) in zip(batchLabels, features):
                    # construct a row that exists of the class label and
                    # extracted features
                    vec = ",".join([str(v) for v in vec])
                    csv.write("{},{}\n".format(label, vec))

    # serialize the label encoder to disk
    with open(LE_PATH, "wb") as f:
        f.write(pickle.dumps(le))

    # with open(BASE_CSV_PATH + '\\image_name.txt', 'w') as filehandle:
    #     for listitem in imageName:
    #         filehandle.write('%s\n' % listitem)