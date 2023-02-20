"""
Implementation notes:
* All values are real numbers
* The weigths and inputs may be implemented as 1-D vectors
*This way, the sum may be calculated in one operation z=w*x
"""
import numpy as np

class Perceptron:
    """ A single neuron with the sigmoid activate function
        Attributtes:
            inputs: The number of inputs in the perceptron, not cunting the bias
            bias: the bias term. By default it´s 1.0
    """
    
    def __init__(self, inputs, bias = 1.0):
        """Return a new Perceptron object with the specified number of inputs +1 (for the bias)"""
        self.weights = np.random.rand(inputs+1)*2 -1
        self.bias = bias
        
    def run(self,x):
        """Run the perceptron. x is a python list with the input values."""
        x_sum = np.dot(np.append(x,self.bias),self.weights)
        #this calculates the product point of the inputs and the weights
        return self.sigmoid(x_sum)
    
    def set_weights(self,w_init):
        """Set the weights. w_init is a python list with the weights"""
        self.weights = np.array(w_init)

    def sigmoid(self,x):
        """Evaluate the sigmoid function for thw floating point input x"""
        return 1/(1 + np.exp(-x))
