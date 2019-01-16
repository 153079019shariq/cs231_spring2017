import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  n_train = X.shape[0]
  n_class = W.shape[1]
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  output =np.dot(X,W)
  soft    =np.zeros((n_train,n_class))
  act_out =np.zeros((n_train,n_class))
  act_out[range(n_train),y]=1
  loss_class =np.zeros((output.shape))
  #Softmax_output############################################
  for i in range(n_train):
    soft[i,:] = np.exp(output[i,:])/np.sum(np.exp(output[i,:]))
    loss_class[i,:]=-(np.multiply(act_out[i,:],np.log(soft[i,:])))
    #print(X[i,:].shape,soft[i,:].shape,act_out[i,:].shape)
  dW= np.dot(X.T,(soft -  act_out))/n_train +2*reg*W
  loss=1/n_train*np.sum(loss_class) +np.sum(W*W)


  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  '''
  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  '''
  
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)
  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  n_train =X.shape[0]
  n_class =W.shape[1] 
  output =np.dot(X,W)
  softmax_out =np.exp(output)/(np.sum(np.exp(output),axis=1).reshape(-1,1))
  act_out =np.zeros((n_train,n_class))
  act_out[list(range(n_train)),y]=1

  
  loss =-1/n_train*np.sum(act_out*np.log(softmax_out))+reg*np.sum(W*W)
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################
  dW= np.dot(X.T,(softmax_out -  act_out))/n_train +2*reg*W
  return loss, dW

