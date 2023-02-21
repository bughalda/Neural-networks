class MultiLayerPerceptron:
    """A multilayer perceptron class that uses the perceptron class above.
        Attributtes:
            layers: A python list with the number of elements per layer
            bias: The bias term. The same bias is used for all neurons
            eta: The learning rate"""
    def __init__(self,layers,bias=1.0):
        """Return a new MLP object with the specified parameters"""
        self.layers = np.array(layers,dtype=object)
        self.bias=bias
        self.network=[] #the list of all neurons
        self.values=[] #the list of all output values
        
        for i in range(len(self.layers)):
            self.values.append([])
            self.network.append([])
            self.values[i] = [0.0 for j in range(self.layers[i])]
            if i>0:
                for j in range(self.layers[i]):
                    self.network[i].append(Perceptron(inputs=self.layers[i-1],bias=self.bias))
    
        self.network = np.array([np.array(x) for x in self.network],dtype=object)
        self.values=np.array([np.array(x) for x in self.values],dtype=object)
        
    
    def set_weights(self,w_init):
        """set the weights.
            w_init is a list of lists with the weights for all, but the input layer"""
        for i in range(len(w_init)):
            for j in range(len(w_init[i])):
                self.network[i+1][j].set_weights(w_init[i][j])
                
    def printWeights(self):
        print()
        for i in range(1,len(self.network)):
            for j in range(self.layers[i]):
                print("Layer",i+1,"Neuron",j,self.network[i][j].weights)
            print()
            
    def run(self,x):
        """Feed a sample x into the multilayer perceptron"""
        x=np.array(x,dtype=object)
        self.values[0]=x
        for i in range(1,len(self.network)):
            for j in range(self.layers[i]):
                self.values[i][j]= self.network[i][j].run(self.values[i-1])
        return self.values[-1]
