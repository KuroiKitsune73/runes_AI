import numpy as np
import matplotlib.pyplot as plt

#Create I/O weights

weights_input_to_hidden=np.random.uniform(-0.5, 0.5, (4,5))
weights_hidden_to_output=np.random.uniform(-0.5, 0.5, (3,4))

#Create I/O bias

bias_input_to_hidden=np.zeros((4,1))
bias_hidden_to_output=np.zeros((3,1))