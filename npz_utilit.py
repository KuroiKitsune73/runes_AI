import numpy as np

def load_dataset():
    with np.load("") as f:  #please add here a name of npz dataset!

        #Convert from RGB -> Unit RGB
        #Unit (from 0 to 1, 0 is white, black is 1, middle is grey)
        x_train=f['x_train'].astype("float32")/255

        #reshape from (60000, 28, 28) into (60000,784)
        #784 neurons in input
        x_train=np.reshape(x_train, (x_train.shape[0],x_train.shape[1]))

        #labels
        y_train=f['y_train']

        #convert to output layer format for us
        #from list [1241217461...123] to matrix 60000x10
        #easy bias correction
        y_train=np.eye(10)[y_train]

        return x_train, y_train