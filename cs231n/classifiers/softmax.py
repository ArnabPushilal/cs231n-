from builtins import range
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
    num_train=X.shape[0]
    num_classes=W.shape[1]
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    for i in range(num_train):
      score=X[i].dot(W)
      score= score - np.max(score)  # using factor to avoid numerical instability
     
      correct_class_score= score[y[i]] 
      correct_ex = np.exp(score[y[i]])
      exp_term =np.sum(np.exp(score))
      
      loss += -correct_class_score + np.log(exp_term) # Loss function expression
      dW[:, y[i]] += (correct_ex/ exp_term-1) * X[i] # correct class gradient ( derived from DL/Dw https://madalinabuzau.github.io/2016/11/29/gradient-descent-on-a-softmax-cross-entropy-cost-function.html)
      for j in range(num_classes):
        if j == y[i]:
            continue
          # for incorrect classes
        dW[:, j] += np.exp(score[j]) / exp_term * X[i] # Check derivation here https://madalinabuzau.github.io/2016/11/29/gradient-descent-on-a-softmax-cross-entropy-cost-function.html
        
    loss /= num_train
    loss += reg * np.sum(W * W)
    dW /= num_train
    dW += 2 * reg * W  


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    num_train=X.shape[0]
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    score=X.dot(W)
    score=score-np.max(score)
    correct_ex=np.exp(score[range(num_train),y])
    exp_term =np.sum(np.exp(score),axis=1)
    loss = -1 *np.log(correct_ex/exp_term)
    loss=np.sum(loss)
    loss /= num_train
    loss += reg * np.sum(W * W)
    m=np.divide(np.exp(score), exp_term.reshape(num_train, 1)) 
    m[range(num_train),y] += -1   # correct/total -1 * X is the gradient
    dW= X.T.dot(m) 
    dW /= num_train
    dW += 2 * reg * W  

    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
