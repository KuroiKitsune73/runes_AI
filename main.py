import numpy as np
import matplotlib.pyplot as plt

import npz_utilit as npz

images, labels=npz.load_dataset()
#Create I/O weights

weights_input_to_hidden=np.random.uniform(-0.5, 0.5, (20,784))
weights_hidden_to_output=np.random.uniform(-0.5, 0.5, (10,20))

#Create I/O bias

bias_input_to_hidden=np.zeros((20,1))
bias_hidden_to_output=np.zeros((10,1))

#Weight correction

epochs=3
#don't set more in case that AI can just memorize dataset:(

#count loss and accuracy
e_loss=0
e_correct=0

#Train it!
learning_rate=0.01

for epoch in range(epochs):
    print(f'Epoch {epoch}:')

    #drop images to multiplicate matrixes
    for image,label in zip(images,labels):
        image=np.reshape(image,(-1,1))
        label=np.reshape(label,(-1,1))

        #Forward propagation (to hidden layer)
        hidden_raw= bias_input_to_hidden + weights_input_to_hidden @image

        #Linear ReLU (sigmoid) to normalize matrix
        hidden=1/(1+np.exp(-hidden_raw))

        #Forward propagation (to output layer)
        output_raw=bias_hidden_to_output+weights_hidden_to_output @hidden
        output=1/(1+np.exp(-output_raw))
    
    #print(output)
        
        #Loss/Error calculation with MSE
        e_loss += 1 / len(output)* np.sum((output-label)**2, axis=0)
        e_correct += int(np.argmax(output)== np.argmax(label))

    #Backpropagation (training AI) OUTPUT
        
    delta_output= output-label
    weights_hidden_to_output += -learning_rate * delta_output @np.transpose(hidden)
    bias_hidden_to_output += -learning_rate * delta_output

    #Backpropagation (training AI) OUTPUT

    delta_hidden= np.transpose(weights_hidden_to_output) @delta_output * (hidden * (1 - hidden))
    weights_input_to_hidden += -learning_rate * delta_hidden @np.transpose(image)
    bias_input_to_hidden += -learning_rate * delta_hidden

    #DONE

    #debug between epochs
    print(f'Loss: {round(e_loss[0]/ images.shape[0]*100,3)}%')
    print(f'Accuracy: {round(e_correct[0]/ images.shape[0]*100,3)}%')
    e_loss=0
    e_correct=0

    #CHECK

import random

test_image = random.choice(images)

#Predict

image=np.reshape(image,(-1,1))

#Forward propagation (to hidden layer)
hidden_raw= bias_input_to_hidden + weights_input_to_hidden @image

#Linear ReLU (sigmoid) to normalize matrix
hidden=1/(1+np.exp(-hidden_raw))

#Forward propagation (to output layer)
output_raw=bias_hidden_to_output+weights_hidden_to_output @hidden
output=1/(1+np.exp(-output_raw))

#Plot generation

plt.imshow(test_image.reshape(28,28), cmap="Greys")
plt.title(f'NN sugests the rune is: {output.argmax()}')
plt.show()