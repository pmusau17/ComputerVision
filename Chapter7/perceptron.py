import numpy as np 


class Perceptron:
    #class constructor 
    def __init__(self,N,alpha):
        # initialize the weight matrix and store the learning rate
        #N is the number of columns in our dataset and for bitwise operations we will set it to two
        #The weight matrix will have N+1 the extra column is for the bias
        self.W = np.random.randn(N + 1) / np.sqrt(N) #fills our weight matrix with values obtained from a Gaussian distribution (zero mean, unit variance)
        self.alpha = alpha

    
    #define the step function for the perceptron
    def step(self,x):
        return 1 if x>0 else 0

    #To actually train the perceptron we will create a function called fit
    #this is common in python and implies "fit a model to the data"
    def fit(self,X,y,epochs=10):
        # Insert a column of 1s as the last entry
        # matrix -- this little trick allows us to treat the bias
        # as a trainable parameter within the weight matrix
        X = np.c_[X, np.ones((X.shape[0]))]

        # loop over the desired number of epochs
        for epoch in np.arange(0, epochs):
            #Loop over the desired data points
            #Zip is a function that Returns a single iterator object, having mapped values from all the containers
            #specified within the brackets
            for (x, target) in zip(X, y):
                # take the dot product between the input features
                # and the weight matrix, then pass this value
                # through the step function to obtain the prediction
                p = self.step(np.dot(x, self.W))

                # only perform the target does not match
                if p != target:
                    error = p -target

                    # update the weight matrix
                    self.W += -self.alpha * error * x

    #define the predict function
    def predict(self,X,addBias=True):
        #ensure our input is a matrix
        X = np.atleast_2d(X) #views inputs as at least two dimensional

        #We trained with a bias so we best use a bias in prediction
        X = np.c_[X, np.ones((X.shape[0]))]

        # take the dot product between the input features and the
        # weight matrix, then pass the value through the step
        # function
        return self.step(np.dot(X, self.W))

