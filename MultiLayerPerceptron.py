
class MultiLayerPerceptron:
    """A multilayer perceptron class that uses the perceptron class above.
        Attributtes:
            layers: A python list with the number of elements per layer
            bias: The bias term. The same bias is used for all neurons
            eta: The learning rate"""
    
    def __init__(self,layers,bias = 1.0, eta = 0.5):
        """Return a new MLP object with the specified parameters"""
        self.layers = np.array(layers,dtype=object)
        self.bias = bias
        self.eta = eta
        self.network = [] #the list of all neurons
        self.values = [] #the list of all output values
        self.d= []
        
        for i in range(len(self.layers)):
            self.values.append([])
            self.d.append([])
            self.network.append([])
            self.values[i] = [0.0 for j in range(self.layers[i])]
            self.d[i] = [0.0 for j in range(self.layers[i])]
            if i>0:
                for j in range(self.layers[i]):
                    self.network[i].append(Perceptron(inputs=self.layers[i-1],bias=self.bias))
    
        self.network = np.array([np.array(x) for x in self.network],dtype=object)
        self.values=np.array([np.array(x) for x in self.values],dtype=object)
        self.d = np.array([np.array(x) for x in self.d],dtype=object)
    
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
        
    #Backpropagation method
    def bp(self, x, y):
        """Run a single (x,y) pair with the backpropagation algorythm)."""
        x = np.array(x, dtype=object)
        y = np.array(y, dtype=object)

        #Step 1:  feed a sample to the network
        outputs = self.run(x)

        #Step 2: calculate the MSE
        error = (y-outputs)
        MSE = sum(error ** 2)/ self.layers[-1]

        #Step 3: Calculate the output error terms
        self.d[-1] = outputs * (1-outputs)*error

        #Step 4: Calculate the error term of each unit on each layer
        for i in reversed(range(1,len(self.network)-1)):
            for h in range(len(self.network[i])):
                fwd_error = 0.0
                for k in range(self.layers[i+1]):
                    fwd_error+= self.network[i+1][k].weights[h] * self.d[i+1][k]
                self.d[i][h]=self.values[i][h] * (1-self.values[i][h]) * fwd_error

        #step 5 & 6 : calculate the deltas and update the weights
        #iterates layers
        for i in range(1, len(self.network)):
            #iterates neurons
            for j in range(self.layers[i]):
                #iterates inputs
                for k in range(self.layers[i-1]+1):
                    if k == self.layers[i-1]:
                        delta = self.eta * self.d[i][j] *self.bias
                    else:
                        delta = self.eta * self.d[i][j] * self.values[i-1][k]
                    self.network[i][j].weights[k]+=delta

        return MSE
