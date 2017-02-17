import numpy as np
import pandas as pd
import math

print 'k-Nearest Neighbours'    
    
class kNN:
    
    def __init__(self):
        pass
        
    def normalisation(self, dataset):
        dataset.ix[ :, :-1 ] = (dataset.ix[ :, :-1 ] - dataset.ix[ :, :-1 ].mean())/(dataset.ix[ :, :-1 ].max() - dataset.ix[ :, :-1 ].min()) 
    
    def euclidean_dist(self, x1, x2):
		return math.sqrt( ( ( x1.subtract( x2 ) )**2 ).sum() )
               
    def manhattan_dist(self, x1, x2):
        return abs( x1.subtract( x2 ) ).sum()
        
    def predict_knn(self, k, training_data, c, type=0):
        df = pd.DataFrame()
        for index, row in training_data.iterrows():
            if type == 0:
                dist = self.euclidean_dist( row[ :-1 ], c[ :-1 ] )
            else:
                dist = self.manhattan_dist( row[ :-1 ], c[ :-1 ] )
            if index < k:
                df = df.append( { 'class': row[ 'class' ], 'dist': dist }, ignore_index=True )                
            else:
                if dist < df[ 'dist' ].max():
                    df.ix[ df[ 'dist' ].idxmax() ] = [ row[ 'class' ], dist ]
        return list( df[ 'class' ].value_counts().index )[ 0 ]
        
dataset_original = pd.read_csv('iris.data', names=['1','2','3','4', 'class'])
dataset = dataset_original.reindex(np.random.permutation(dataset_original.index)).reset_index(drop=True)
m = dataset.shape[ 0 ]

knn = kNN()
knn.normalisation( dataset )

boundaries = [ int(m*0.6) - 1, int(m*0.8) - 1 ]

training_data = dataset.ix[ :boundaries[ 0 ] ]
val_data = dataset.ix[ boundaries[ 0 ]+1:boundaries[ 1 ] ].reset_index(drop=True)
test_data = dataset.ix[ boundaries[ 1 ]+1: ].reset_index(drop=True)

def calc_accuracy( kk, data ):
    correct_predictions = 0
    for index, row in data.iterrows():
        if knn.predict_knn( k, training_data, row, 0 ) == row[ 'class' ]:
            correct_predictions = correct_predictions + 1
    return ( correct_predictions / ( data.shape[ 0 ]*1.0) ) * 100

accuracy = 0
#to get optimal k value using euclidean distance
for k in xrange(1, 10):
    _accuracy = calc_accuracy( k, val_data )    
    if _accuracy >= accuracy:
        accuracy = _accuracy
        final_k = k
print 'The optimal k is', final_k
        
correct_predictions = 0      
accuracy = 0  
#check accuracy on test data
accuracy = calc_accuracy( k, test_data )
print 'The accuracy of using the kNN on the test data is', accuracy, '%'
print ''
print ''
print 'Logistic Regression with Regularization'

class logisticRegression():
    
    def __init__(self):
        pass
    
    def sigmoid(self, z):
        return 1.0 / ( 1.0 + np.exp(-1.0 * z.astype( float )) )
        
    def cost_function(self, X, Y, theta, m, _lambda):
        return ( 1.0/m ) * ( -np.transpose( Y ).dot( np.log( self.sigmoid( X.dot( theta ) ) ) ) - np.transpose( 1-Y ).dot( np.log( 1-self.sigmoid( X.dot( theta ) ) ) ) ) + ( ( _lambda / ( 2.0 * m ) ) * ( theta[ 1: ].T.dot( theta[ 1: ] ) ) )
        
    def grad(self, X, Y, theta, m, _lambda):
        thetaReg = np.concatenate( ( np.zeros( ( 1, 1 ) ), theta[ 1: ] ) )
        return np.transpose( ( 1.0/m )*np.transpose( self.sigmoid( X.dot( theta ) ) - Y ).dot( X ) ) + ( ( _lambda/m ) * thetaReg )
        
    def train(self, X, Y, lr, iterations, theta, m, _lambda):
        for i in xrange( iterations ):
            theta = theta - lr * self.grad(X, Y, theta, m, _lambda)
        return theta, self.cost_function(X, Y, theta, m, _lambda)
        
    def predict(self, X, theta):
		return np.round( self.sigmoid( X.dot( theta ) ) ).astype( int )
            
dataset1 = dataset_original[ :100 ].values
np.random.shuffle( dataset1 )
X = dataset1[ :, :-1 ]
X = np.concatenate( ( np.ones( ( X.shape[ 0 ], 1 ) ), X ), axis=1 )
Y = np.unique( dataset1[ :, -1: ], return_inverse=True )[ 1 ]
Y = np.reshape( Y, ( np.shape( Y )[ 0 ], 1 ) )
m = X.shape[ 0 ]
boundaries = [ int(m*0.6), int(m*0.8) ]

X_training_data = X[ :boundaries[ 0 ] ]
X_val_data = X[ boundaries[ 0 ]:boundaries[ 1 ] ]
X_test_data = X[ boundaries[ 1 ]: ]

Y_training_data = Y[ :boundaries[ 0 ] ]
Y_val_data = Y[ boundaries[ 0 ]:boundaries[ 1 ] ]
Y_test_data = Y[ boundaries[ 1 ]: ]

theta = np.zeros( ( X.shape[ 1 ], 1 ) )
_lambda = 1.0

lreg = logisticRegression()
theta, cost_function = lreg.train( X_training_data, Y_training_data, 1, 1000, theta, m, _lambda )
print "The parameter values are", theta
predicted_data = lreg.predict( X_test_data, theta )

accuracy = ( ( (Y_test_data == predicted_data).sum() ) / ( 1.0 * Y_test_data.shape[ 0 ] ) ) * 100

print "The accuracy of prediction", accuracy, "%"



    



    

        

    
        



