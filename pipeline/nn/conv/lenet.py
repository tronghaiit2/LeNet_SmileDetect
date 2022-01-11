# import packages
from tensorflow.keras.models import Sequential
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, Activation, Flatten, Dense, Dropout

class LeNet:
    @staticmethod
    def build(width, height, depth, classes):
        # initialize the model
        model = Sequential()
        inputShape = (height, width, depth)

        # if we are using "channel_first", update the input shape
        if K.image_data_format() == "channel_first":
            inputShape = (depth, height, width)

        # first set of CONV => RELU
        model.add(Conv2D(4, (5, 5), padding="valid", input_shape=inputShape))
        model.add(Activation("relu"))
        #model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

         # second set of CONV => RELU => POOL layers
        model.add(Conv2D(8, (5, 5), padding="valid"))
        model.add(Activation("relu"))
        model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

        # third set of CONV => RELU => POOL layers
        model.add(Conv2D(16, (5, 5), padding="valid"))
        model.add(Activation("relu"))
        model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))

        # first set of FC => RELU layers
        model.add(Flatten())
        model.add(Dense(120))
        model.add(Activation("relu"))
        model.add(Dropout(rate=0.2))

        # second set of FC => RELU layers
        # model.add(Dense(84))
        # model.add(Activation("relu"))

        # softmax classififer
        model.add(Dense(classes))
        model.add(Activation("softmax"))

        # return the constructed network architecture
        return model