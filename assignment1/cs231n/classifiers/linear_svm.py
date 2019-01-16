import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
    """
  Structured SVM loss function, naive implementation (with loops).

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
    dW = np.zeros(W.shape)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class_score = scores[y[i]]
        loss_contributors_count = 0
        for j in xrange(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                # incorrect class gradient part
                dW[:, j] += X[i]
                # count contributor terms to loss function
                loss_contributors_count += 1
        # correct class gradient part
        dW[:, y[i]] += (-1) * loss_contributors_count * X[i]

    # Right now the loss is a sum over all training examples, but we want it
    # to be an average instead so we divide by num_train.
    loss /= num_train
    dW /= num_train

    # Add regularization to the loss.
    loss += reg * np.sum(W * W)
    # Add regularization to the gradient
    dW += 2 * reg * W

    #############################################################################
    # TODO:                                                                     #
    # Compute the gradient of the loss function and store it dW.                #
    # Rather that first computing the loss and then computing the derivative,   #
    # it may be simpler to compute the derivative at the same time that the     #
    # loss is being computed. As a result you may need to modify some of the    #
    # code above to compute the gradient.                                       #
    #############################################################################


    return loss, dW



def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  #print(X.shape,W.shape)
  s_j  = np.dot(X,W)                                            #Calculate the dot of X,W
  s_mask = np.zeros(s_j.shape)                                  #For s_j calculation
  s_mask[list(range(y.shape[0])),y] =1                          #Make those element 1 corresponding to y 
  temp2 =np.sum(np.multiply(s_j,s_mask),axis=1).reshape(-1,1)   #Create s_j of shape (N,1) #if just multiplied then shape(N,C)
  temp =s_j - temp2 +1
  temp[list(range(y.shape[0])),y] =0                            #Make the s_j=s_yi the loss =0
  max_mask =(temp>0).astype(float)*temp                        #Multiply(max(0,temp))
  loss   =np.sum(max_mask)/(X.shape[0]) +reg*np.sum(W*W)

  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################

  g_mask = np.zeros((temp.shape))
  g_mask[temp>0] =1
  g_mask[list(range(y.shape[0])),y]=-np.sum(g_mask,axis=1)
  dW = np.dot(X.T,g_mask) 
  dW /=X.shape[0]
  dW +=2 * reg * W
  #print(X.T.shape,g_mask.shape,dW.shape)
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
