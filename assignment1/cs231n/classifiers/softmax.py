import numpy as np
from random import shuffle

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

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  n = X.shape[0]
  c = W.shape[1]
  score = X.dot(W) # N*C
  # Fix for numeraical stability by suntracting max from score vector
  subtract = np.reshape(np.max(score, axis=1), (n, 1)) # N*1
  score = score - subtract
  for i in range(n):
    curr_score = score[i]
    loss += -np.log(np.exp(curr_score[y[i]])/np.sum(np.exp(curr_score)))
    for j in range(c):
      sub_grad = np.exp(curr_score[j]) / np.sum(np.exp(curr_score))
      if j == y[i]:
        dW[:, j] += (sub_grad - 1) * X[i]
      else:
        dW[:, j] += sub_grad * X[i]

  loss = loss/n + reg * np.sum(W * W)
  dW = dW/n + 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  n = X.shape[0]
  c = W.shape[1]
  score = X.dot(W) # N*C
  # Fix for numeraical stability by suntracting max from score vector
  subtract = np.reshape(np.max(score, axis=1), (n, 1)) # N*1
  score = score - subtract # N*C  
  temp = np.reshape(np.sum(np.exp(score), axis=1), (n, 1))
  softmax = np.exp(score) / temp # N*C
  loss = -np.sum(np.log(softmax[range(n), y]))
  loss = loss/n + reg * np.sum(W * W)
  
  # sub_grad = np.exp(score) / temp # N*C
  softmax[range(n), y] -= 1
  dW = (X.T).dot(softmax)
  dW = dW/n + 2 * reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

