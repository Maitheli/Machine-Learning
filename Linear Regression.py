import numpy

class LinearReg:
	def __init__(self, N):
         self.theta = numpy.random.rand( N+1, 1 )

	def gradient(self, X, Y):
            m = X.shape[0]
            predictions = self.predict(X)
            MSE = ( numpy.sum( predictions - Y ) ** 2 ) / m
            grad = 2 * ( numpy.dot( X.T, predictions - Y ) ) / m
            return grad, MSE

	def predict(self,X):
            X = numpy.concatenate( ( numpy.ones( ( X.shape[ 0 ], 1 ) ), X ), axis=1 )          
            predictions = numpy.dot( X, self.theta )
            return predictions
		#return self.theta.transpose() * X

	def train(self,X,Y, numberOfIterations, lr):
		#gradient descent code here
           #X = numpy.concatenate( ( numpy.ones( ( X.shape[ 0 ], 1 ) ), X ), axis=1 )
           for i in xrange(numberOfIterations):
               grad , MSE = self.gradient(X,Y)
               self.theta = self.theta - lr * grad

if __name__ == '__main__':
	#intialize data and perform train, validation, test splits
	#call train and predict
    N = 1000
    X = numpy.random.rand( N, 1 )
    Y = 5 * X + 7
    noise = numpy.random.rand( N, 1)
    noise = ( noise - 0.5 ) * 2
    Y = Y + noise
    
    boundaries = [ int(N*0.6), int(N*0.8) ]
    
    X_training_data = X[ :boundaries[ 0 ], : ]
    X_val_data = X[ boundaries[ 0 ]:boundaries[ 1 ], : ]
    X_test_data = X[ boundaries[ 1 ]:, : ]
    
    Y_training_data = Y[ :boundaries[ 0 ], : ]
    Y_val_data = Y[ boundaries[ 0 ]:boundaries[ 1 ], : ]
    Y_test_data = Y[ boundaries[ 1 ]:, : ]

    l = LinearReg( X_training_data.shape[ 1 ] )
    l.train( X_training_data, Y_training_data, 1000, 0.1 )

    print "The Final value of Parameters : ", l.theta
