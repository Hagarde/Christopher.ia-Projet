

def create_model(): # il faudrait fragmenter cette fonction avec model1, model 2 ,modelefinal 

    tf.random.set_seed(1234)
    ModelFinal = Sequential()
    # model1 est le premier CNN qui prend les images et les transforme en vecteur pour faire une matrice avec couche TimeDistributed
    model1 = Sequential()
    # Modèle post-transformation en vecteur 
    model2= Sequential()
    
    #https://www.tensorflow.org/api_docs/python/tf/keras/layers/DepthwiseConv2D

    model1.add(DepthwiseConv2D((3,3),activation="relu", strides=(1,1), padding = "same", input_shape=(TRAINING_IMAGE_SIZE[0],TRAINING_IMAGE_SIZE[0],NUMBER_OF_CHANNELS)))
    model1.add(Conv2D(64, (1,1), activation="relu", strides=(1,1), padding = "same"))
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    model1.add(DepthwiseConv2D((3,3),activation="relu", strides=(1,1), padding = "same"))
    model1.add(Conv2D(128, (1,1), activation="relu", strides=(1,1), padding = "same"))
    model1.add(MaxPool2D(pool_size=(2, 2)))

    model1.add(DepthwiseConv2D((3,3),activation="relu", strides=(1,1), padding = "same"))
    model1.add(Conv2D(256, (1,1), activation="relu", strides=(1,1), padding = "same"))
    model1.add(MaxPool2D(pool_size=(2, 2)))

    model1.add(Flatten())
    
    ModelFinal.add(tf.keras.layers.TimeDistributed(model))
    
    model2.add(Conv1D(64, 3, activation="relu", strides=(1,1), padding = "same"))
    model2.add(MaxPool1D(pool_size=2)

    model2.add(Conv1D(64, 3, activation="relu", strides=(1,1), padding = "same"))
    model2.add(MaxPool1D(pool_size=2)

    model2.add(Conv1D(64, 3, activation="relu", strides=(1,1), padding = "same"))
    model2.add(MaxPool1D(pool_size=2)

    model2.add(Conv1D(64, 3, activation="relu", strides=(1,1), padding = "same"))
    model2.add(MaxPool1D(pool_size=2)

    model2.add(tf.keras.layers.GlobalAveragePooling1D())

    model2.add(Dense(256,activation="relu"))

    model2.add(Dense(128,activation="relu"))

    model2.add(Dense(128,activation="relu"))

    model2.add(Dense(64,activation="relu"))

    model2.add(Dense(19,activation="softmax"))
    model2.summary()
    
    ModelFinal.add(model2)

    return ModelFinal    