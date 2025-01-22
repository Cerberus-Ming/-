from __future__ import division
from __future__ import print_function

from IPython import get_ipython

# get_ipython().system('pip install git+https://github.com/deepvision-class/starter-code')

# Setup code

import math
import random
import time

import coutils
import matplotlib.pyplot as plt
import torch

# get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams['figure.figsize'] = (10.0, 8.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

if torch.cuda.is_available:
    print('Good to go!')
else:
    print('Please set GPU via Edit -> Notebook Settings.')


# load CIFAR10 dataset with normalization

def get_CIFAR10_data(validation_ratio=0.02):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the linear classifier.
    """
    X_train, y_train, X_test, y_test = coutils.data.cifar10()

    # Move all the data to the GPU
    X_train = X_train.cuda()
    y_train = y_train.cuda()
    X_test = X_test.cuda()
    y_test = y_test.cuda()

    # 0. Visualize some examples from the dataset.
    class_names = [
        'plane', 'car', 'bird', 'cat', 'deer',
        'dog', 'frog', 'horse', 'ship', 'truck'
    ]
    img = coutils.utils.visualize_dataset(X_train, y_train, 12, class_names)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

    # 1. Normalize the data: subtract the mean RGB (zero mean)
    mean_image = X_train.mean(dim=0, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True)
    X_train -= mean_image
    X_test -= mean_image

    # 2. Reshape the image data into rows
    X_train = X_train.reshape(X_train.shape[0], -1)
    X_test = X_test.reshape(X_test.shape[0], -1)

    # 3. Add bias dimension and transform into columns
    ones_train = torch.ones(X_train.shape[0], 1, device=X_train.device)
    X_train = torch.cat([X_train, ones_train], dim=1)
    ones_test = torch.ones(X_test.shape[0], 1, device=X_test.device)
    X_test = torch.cat([X_test, ones_test], dim=1)

    # 4. Carve out part of the training set to use for validation.
    num_training = int(X_train.shape[0] * (1.0 - validation_ratio))
    num_validation = X_train.shape[0] - num_training

    # Return the dataset as a dictionary
    data_dict = {}
    data_dict['X_val'] = X_train[num_training:num_training + num_validation]
    data_dict['y_val'] = y_train[num_training:num_training + num_validation]
    data_dict['X_train'] = X_train[0:num_training]
    data_dict['y_train'] = y_train[0:num_training]

    data_dict['X_test'] = X_test
    data_dict['y_test'] = y_test
    return data_dict


# Invoke the above function to get our data.
data_dict = get_CIFAR10_data()
print('Train data shape: ', data_dict['X_train'].shape)
print('Train labels shape: ', data_dict['y_train'].shape)
print('Validation data shape: ', data_dict['X_val'].shape)
print('Validation labels shape: ', data_dict['y_val'].shape)
print('Test data shape: ', data_dict['X_test'].shape)
print('Test labels shape: ', data_dict['y_test'].shape)


def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-7):
    """
    Utility function to perform numeric gradient checking. We use the centered
    difference formula to compute a numeric derivative:

    f'(x) =~ (f(x + h) - f(x - h)) / (2h)

    Rather than computing a full numeric gradient, we sparsely sample a few
    dimensions along which to compute numeric derivatives.

    Inputs:
    - f: A function that inputs a torch tensor and returns a torch scalar
    - x: A torch tensor giving the point at which to evaluate the numeric gradient
    - analytic_grad: A torch tensor giving the analytic gradient of f at x
    - num_checks: The number of dimensions along which to check
    - h: Step size for computing numeric derivatives
    """
    # fix random seed for
    coutils.utils.fix_random_seed()

    for i in range(num_checks):
        ix = tuple([random.randrange(m) for m in x.shape])

        oldval = x[ix].item()
        x[ix] = oldval + h  # increment by h
        fxph = f(x).item()  # evaluate f(x + h)
        x[ix] = oldval - h  # decrement by h
        fxmh = f(x).item()  # evaluate f(x - h)
        x[ix] = oldval  # reset

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error_top = abs(grad_numerical - grad_analytic)
        rel_error_bot = (abs(grad_numerical) + abs(grad_analytic) + 1e-12)
        rel_error = rel_error_top / rel_error_bot
        msg = 'numerical: %f analytic: %f, relative error: %e'
        print(msg % (grad_numerical, grad_analytic, rel_error))


# ## SVM Classifier
#     
# - implement a fully-vectorized **loss function** for the SVM
# - implement the fully-vectorized expression for its **analytic gradient**
# - **check your implementation** using numerical gradient
# - use a validation set to **tune the learning rate and regularization** strength
# - **optimize** the loss function with **SGD**
# - **visualize** the final learned weights
# 

def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples. When you implment the regularization over W, please DO NOT
    multiply the regularization term by 1/2 (no coefficient).

    Inputs:
    - W: A PyTorch tensor of shape (D, C) containing weights.
    - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as torch scalar
    - gradient of loss with respect to weights W; a tensor of same shape as W
    """
    dW = torch.zeros_like(W)  # initialize the gradient as zero

    # compute the loss and the gradient
    num_classes = W.shape[1]
    num_train = X.shape[0]
    loss = 0.0
    for i in range(num_train):
        scores = W.t().mv(X[i])  # shape (c,)
        correct_class_score = scores[y[i]]
        for j in range(num_classes):
            if j == y[i]:
                continue
            margin = scores[j] - correct_class_score + 1  # note delta = 1
            if margin > 0:
                loss += margin
                dW[:, y[i]] -= X[i]
                dW[:, j] += X[i]

    loss /= num_train

    # Add regularization to the loss.
    loss += reg * torch.sum(W * W)
    dW /= num_train
    dW += reg * 2 * W

    return loss, dW


# generate a random SVM weight tensor of small numbers
coutils.utils.fix_random_seed()
W = torch.randn(3073, 10, device=data_dict['X_val'].device) * 0.0001

loss, grad = svm_loss_naive(W, data_dict['X_val'], data_dict['y_val'], 0.000005)
print('loss: %f' % (loss,))

# The `grad` returned from the function above is right now all zero. 
# Derive and implement the gradient for the SVM cost function and implement it inline inside the function `svm_loss_naive`. 

# Once you've implemented the gradient, recompute it with the code below
# and gradient check it with the function we provided for you

# Use a random W and a minibatch of data from the val set for gradient checking
coutils.utils.fix_random_seed()
W = 0.0001 * torch.randn(3073, 10, device=data_dict['X_val'].device).double()
batch_size = 64
X_batch = data_dict['X_val'][:64].double()
y_batch = data_dict['y_val'][:64]

# Compute the loss and its gradient at W.
loss, grad = svm_loss_naive(W.double(), X_batch, y_batch, reg=0.0)

# Numerically compute the gradient along several randomly chosen dimensions, and
# compare them with your analytically computed gradient. The numbers should
# match almost exactly along all dimensions.
f = lambda w: svm_loss_naive(w, X_batch, y_batch, reg=0.0)[0]
grad_numerical = grad_check_sparse(f, W.double(), grad)

# Do the gradient check once again with regularization turned on.

# Use a minibatch of data from the val set for gradient checking
coutils.utils.fix_random_seed()
W = 0.0001 * torch.randn(3073, 10, device=data_dict['X_val'].device).double()
batch_size = 64
X_batch = data_dict['X_val'][:64].double()
y_batch = data_dict['y_val'][:64]

# Compute the loss and its gradient at W.
loss, grad = svm_loss_naive(W.double(), X_batch, y_batch, reg=1e3)

# Numerically compute the gradient along several randomly chosen dimensions, and
# compare them with your analytically computed gradient. The numbers should
# match almost exactly along all dimensions.
f = lambda w: svm_loss_naive(w, X_batch, y_batch, reg=1e3)[0]
grad_numerical = grad_check_sparse(f, W.double(), grad)


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation. When you implment
    the regularization over W, please DO NOT multiply the regularization term by
    1/2 (no coefficient).

    Inputs and outputs are the same as svm_loss_naive.

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples. When you implment the regularization over W, please DO NOT
    multiply the regularization term by 1/2 (no coefficient).

    Inputs:
    - W: A PyTorch tensor of shape (D, C) containing weights.
    - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as torch scalar
    - gradient of loss with respect to weights W; a tensor of same shape as W
    """
    loss = 0.0
    dW = torch.zeros_like(W)  # initialize the gradient as zero

    D, C = W.shape
    M = X.shape[0]
    idx0 = torch.arange(M)

    scores = X.mm(W)
    assert scores.shape == (M, C)

    correct_class_scores = scores[idx0, y].reshape(-1, 1)
    assert correct_class_scores.shape == (M, 1)

    margin = scores - correct_class_scores + 1

    assert margin.shape == (M, C)

    margin[margin < 0] = 0  # max(0, margin) operation
    margin[idx0, y] = 0  # correct prediction loss are eliminated

    loss = margin.sum() + reg * (W ** 2).sum()
    loss /= M

    # Compute gradient
    margin[margin > 0] = 1
    valid_margin_count = margin.sum(dim=1)

    # Subtract in correct class (-s_y)
    margin[idx0, y] -= valid_margin_count
    dW = (X.t()).matmul(margin) / M

    # Regularization gradient
    dW = dW + (reg * 2 * W)

    return loss, dW


# Let's first check the speed and performance bewteen the non-vectorized and the vectorized version. You should see a speedup of more than 100x.

# In[ ]:

# Next implement the function svm_loss_vectorized; for now only compute the loss;
# we will implement the gradient in a moment.

# Use random weights and a minibatch of val data for gradient checking
coutils.utils.fix_random_seed()
W = 0.0001 * torch.randn(3073, 10, device=data_dict['X_val'].device).double()
X_batch = data_dict['X_val'][:128].double()
y_batch = data_dict['y_val'][:128]
reg = 0.000005

# Run and time the naive version
torch.cuda.synchronize()
tic = time.time()
loss_naive, grad_naive = svm_loss_naive(W, X_batch, y_batch, reg)
torch.cuda.synchronize()
toc = time.time()
ms_naive = 1000.0 * (toc - tic)
print('Naive loss: %e computed in %.2fms' % (loss_naive, ms_naive))

# Run and time the vectorized version
torch.cuda.synchronize()
tic = time.time()
loss_vec, _ = svm_loss_vectorized(W, X_batch, y_batch, reg)
torch.cuda.synchronize()
toc = time.time()
ms_vec = 1000.0 * (toc - tic)
print('Vectorized loss: %e computed in %.2fms' % (loss_vec, ms_vec))

# The losses should match but your vectorized implementation should be much faster.
print('Difference: %.2e' % (loss_naive - loss_vec))
print('Speedup: %.2fX' % (ms_naive / ms_vec))

# Then, let's compute the gradient of the loss function. We can check the difference of gradient as well. (The error should be less than 1e-6)
# 
# Now implement a vectorized version of the gradient computation in `svm_loss_vectorize` above. Run the cell below to compare the gradient of your naive and vectorized implementations. The difference between the gradients should be less than `1e-6`, and the vectorized version should run at least 100x faster.
# 

# In[ ]:

# The naive implementation and the vectorized implementation should match, but
# the vectorized version should still be much faster.

# Use random weights and a minibatch of val data for gradient checking
coutils.utils.fix_random_seed()
W = 0.0001 * torch.randn(3073, 10, device=data_dict['X_val'].device).double()
X_batch = data_dict['X_val'][:128].double()
y_batch = data_dict['y_val'][:128]
reg = 0.000005

# Run and time the naive version
torch.cuda.synchronize()
tic = time.time()
_, grad_naive = svm_loss_naive(W, X_batch, y_batch, 0.000005)
torch.cuda.synchronize()
toc = time.time()
ms_naive = 1000.0 * (toc - tic)
print('Naive loss and gradient: computed in %.2fms' % ms_naive)

# Run and time the vectorized version
torch.cuda.synchronize()
tic = time.time()
_, grad_vec = svm_loss_vectorized(W, X_batch, y_batch, 0.000005)
torch.cuda.synchronize()
toc = time.time()
ms_vec = 1000.0 * (toc - tic)
print('Vectorized loss and gradient: computed in %.2fms' % ms_vec)

# The loss is a single number, so it is easy to compare the values computed
# by the two implementations. The gradient on the other hand is a tensor, so
# we use the Frobenius norm to compare them.
grad_difference = torch.norm(grad_naive - grad_vec, p='fro')
print('Gradient difference: %.2e' % grad_difference)
print('Speedup: %.2fX' % (ms_naive / ms_vec))


# Now that we have an efficient vectorized implementation of the SVM loss and its gradient, we can implement a training pipeline for linear classifiers.
# 
# Complete the implementation of the following function:
# 

# In[ ]:


def train_linear_classifier(loss_func, W, X, y, learning_rate=1e-3,
                            reg=1e-5, num_iters=100, batch_size=200, verbose=False):
    """
    Train this linear classifier using stochastic gradient descent.

    Inputs:
    - loss_func: loss function to use when training. It should take W, X, y
      and reg as input, and output a tuple of (loss, dW)
    - W: A PyTorch tensor of shape (D, C) giving the initial weights of the
      classifier. If W is None then it will be initialized here.
    - X: A PyTorch tensor of shape (N, D) containing training data; there are N
      training samples each of dimension D.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c
      means that X[i] has label 0 <= c < C for C classes.
    - learning_rate: (float) learning rate for optimization.
    - reg: (float) regularization strength.
    - num_iters: (integer) number of steps to take when optimizing
    - batch_size: (integer) number of training examples to use at each step.
    - verbose: (boolean) If true, print progress during optimization.

    Returns: A tuple of:
    - W: The final value of the weight matrix and the end of optimization
    - loss_history: A list of Python scalars giving the values of the loss at each
      training iteration.
    """
    # assume y takes values 0...K-1 where K is number of classes
    num_classes = torch.max(y) + 1
    num_train, dim = X.shape
    if W is None:
        # lazily initialize W
        W = 0.000001 * torch.randn(dim, num_classes, device=X.device, dtype=X.dtype)

    # Run stochastic gradient descent to optimize W
    loss_history = []
    for it in range(num_iters):
        X_batch = None
        y_batch = None

        idx = torch.randint(num_train, size=(batch_size,))
        X_batch = X[idx, :]
        y_batch = y[idx]

        # evaluate loss and gradient
        loss, grad = loss_func(W, X_batch, y_batch, reg)
        loss_history.append(loss.item())

        W -= learning_rate * grad

        if verbose and it % 100 == 0:
            print('iteration %d / %d: loss %f' % (it, num_iters, loss))

    return W, loss_history


# Once you have implemented the training function, run the following cell to train a linear classifier using some default hyperparameters:

# In[ ]:


# fix random seed before we perform this operation
coutils.utils.fix_random_seed()

torch.cuda.synchronize()
tic = time.time()

W, loss_hist = train_linear_classifier(svm_loss_vectorized, None,
                                       data_dict['X_train'],
                                       data_dict['y_train'],
                                       learning_rate=3e-11, reg=2.5e4,
                                       num_iters=1500, verbose=True)

torch.cuda.synchronize()
toc = time.time()
print('That took %fs' % (toc - tic))

# A useful debugging strategy is to plot the loss as a function of iteration number. In this case it seems our hyperparameters are not good, since the training loss is not decreasing very fast.
# 
# 

# In[ ]:


plt.plot(loss_hist, 'o')
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.show()


# Let's move on to the prediction stage:

# In[ ]:


def predict_linear_classifier(W, X):
    """
    Use the trained weights of this linear classifier to predict labels for
    data points.

    Inputs:
    - W: A PyTorch tensor of shape (D, C), containing weights of a model
    - X: A PyTorch tensor of shape (N, D) containing training data; there are N
      training samples each of dimension D.

    Returns:
    - y_pred: PyTorch int64 tensor of shape (N,) giving predicted labels for each
      elemment of X. Each element of y_pred should be between 0 and C - 1.
    """
    y_pred = torch.zeros(X.shape[0])

    y_pred = torch.argmax(X.matmul(W), dim=1)

    return y_pred


# Then, let's evaluate the performance our trained model on both the training and validation set. You should see validation accuracy less than 10%.

# In[ ]:


# evaluate the performance on both the training and validation set
y_train_pred = predict_linear_classifier(W, data_dict['X_train'])
train_acc = 100.0 * (data_dict['y_train'] == y_train_pred).float().mean().item()
print('Training accuracy: %.2f%%' % train_acc)
y_val_pred = predict_linear_classifier(W, data_dict['X_val'])
val_acc = 100.0 * (data_dict['y_val'] == y_val_pred).float().mean().item()
print('Validation accuracy: %.2f%%' % val_acc)


# Unfortunately, the performance of our initial model is quite bad. To find a better hyperparamters, let's first modulize the functions that we've implemented.

# In[ ]:


# Note: We will re-use `LinearClassifier' in Softmax section
class LinearClassifier(object):

    def __init__(self):
        self.W = None

    def train(self, X_train, y_train, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        train_args = (self.loss, self.W, X_train, y_train, learning_rate, reg,
                      num_iters, batch_size, verbose)
        self.W, loss_history = train_linear_classifier(*train_args)
        return loss_history

    def predict(self, X):
        return predict_linear_classifier(self.W, X)

    def loss(self, W, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - W: A PyTorch tensor of shape (D, C) containing (trained) weight of a model.
        - X_batch: A PyTorch tensor of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A PyTorch tensor of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an tensor of the same shape as W
        """
        pass

    def _loss(self, X_batch, y_batch, reg):
        self.loss(self.W, X_batch, y_batch, reg)


class LinearSVM(LinearClassifier):
    """ A subclass that uses the Multiclass SVM loss function """

    def loss(self, W, X_batch, y_batch, reg):
        return svm_loss_vectorized(W, X_batch, y_batch, reg)


# Now, please use the validation set to tune hyperparameters (regularization strength and learning rate). You should experiment with different ranges for the learning rates and regularization strengths.
# 
# To get full credit for the assignment your best model found through cross-validation should achieve an accuracy of at least 37% on the validation set.
# 

# results is dictionary mapping tuples of the form
# (learning_rate, regularization_strength) to tuples of the form
# (training_accuracy, validation_accuracy). The accuracy is simply the fraction
# of data points that are correctly classified.
results = {}
best_val = -1  # The highest validation accuracy that we have seen so far.
best_svm = None  # The LinearSVM object that achieved the highest validation rate.
learning_rates = []  # learning rate candidates, e.g. [1e-3, 1e-2, ...]
regularization_strengths = []  # regularization strengths candidates e.g. [1e0, 1e1, ...]
from random import uniform

num_iters = 2000
num_model = 5
for i in range(num_model):
    lr = 10 ** uniform(-2, -2.5)
    reg = 10 ** uniform(-3, -3.5)

    train_args = (data_dict['X_train'], data_dict['y_train'], lr, reg,
                  num_iters, 128, False)

    model = LinearSVM()
    model.train(*train_args)

    y_train_pred = model.predict(data_dict['X_train'])
    y_val_pred = model.predict(data_dict['X_val'])

    train_acc = 100.0 * (data_dict['y_train'] == y_train_pred).float().mean().item()
    val_acc = 100.0 * (data_dict['y_val'] == y_val_pred).float().mean().item()
    results[(lr, reg)] = (train_acc, val_acc)

    if val_acc > best_val:
        best_val = val_acc
        best_svm = model

# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %.3e reg %.3e train accuracy: %.3f val accuracy: %.3f' % (
        lr, reg, train_accuracy, val_accuracy))

print('best validation accuracy achieved during cross-validation: %f' % best_val)

# Visualize the cross-validation results. You can use this as a debugging tool -- after examining the cross-validation results here, you may want to go back and rerun your cross-validation from above.

# In[ ]:


x_scatter = [math.log10(x[0]) for x in results]
y_scatter = [math.log10(x[1]) for x in results]

# plot training accuracy
marker_size = 100
colors = [results[x][0] for x in results]
plt.scatter(x_scatter, y_scatter, marker_size, c=colors, cmap='viridis')
plt.colorbar()
plt.xlabel('log learning rate')
plt.ylabel('log regularization strength')
plt.title('CIFAR-10 training accuracy')
plt.gcf().set_size_inches(8, 5)
plt.show()

# plot validation accuracy
colors = [results[x][1] for x in results]  # default size of markers is 20
plt.scatter(x_scatter, y_scatter, marker_size, c=colors, cmap='viridis')
plt.colorbar()
plt.xlabel('log learning rate')
plt.ylabel('log regularization strength')
plt.title('CIFAR-10 validation accuracy')
plt.gcf().set_size_inches(8, 5)
plt.show()

# Evaluate the best svm on test set. To get full credit for the assignment you should achieve a test-set accuracy above 35%.

# In[ ]:


y_test_pred = best_svm.predict(data_dict['X_test'])
test_accuracy = torch.mean((data_dict['y_test'] == y_test_pred).float())
print('linear SVM on raw pixels final test set accuracy: %f' % test_accuracy)

# Visualize the learned weights for each class. Depending on your choice of learning rate and regularization strength, these may or may not be nice to look at.

# In[ ]:


w = best_svm.W[:-1, :]  # strip out the bias
w = w.reshape(3, 32, 32, 10)
w = w.transpose(0, 2).transpose(1, 0)

w_min, w_max = torch.min(w), torch.max(w)
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in range(10):
    plt.subplot(2, 5, i + 1)

    # Rescale the weights to be between 0 and 255
    wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
    plt.imshow(wimg.type(torch.uint8).cpu())
    plt.axis('off')
    plt.title(classes[i])


# ## Softmax Classifier
# 
# Similar to the SVM, you will:
# 
# - implement a fully-vectorized **loss function** for the Softmax classifier
# - implement the fully-vectorized expression for its **analytic gradient**
# - **check your implementation** with numerical gradient
# - use a validation set to **tune the learning rate and regularization** strength
# - **optimize** the loss function with **SGD**
# - **visualize** the final learned weights

# First, let's start from implementing the naive softmax loss function with nested loops.


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops).  When you implment
    the regularization over W, please DO NOT multiply the regularization term by
    1/2 (no coefficient).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A PyTorch tensor of shape (D, C) containing weights.
    - X: A PyTorch tensor of shape (N, D) containing a minibatch of data.
    - y: A PyTorch tensor of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an tensor of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = torch.zeros_like(W)

    M = X.shape[0]
    C = W.shape[1]

    for i in range(M):
        label = y[i]
        y_hat = (X[i, :].matmul(W)).reshape(1, -1)

        y_hat -= torch.max(y_hat)
        y_hat = torch.exp(y_hat) / (torch.exp(y_hat).sum())

        assert y_hat.shape == (1, C)

        loss += -torch.log(y_hat[:, label])

        y_hat[:, label] -= 1
        dW += (X[i, :].reshape(-1, 1)).matmul(y_hat)

    dW /= M
    loss /= M

    dW += reg * 2 * W
    loss += reg * (W ** 2).sum()

    return loss, dW


# As a sanity check to see whether we have implemented the loss correctly, run the softmax classifier with a small random weight matrix and no regularization.

# Generate a random softmax weight tensor and use it to compute the loss.
coutils.utils.fix_random_seed()
W = 0.0001 * torch.randn(3073, 10, device=data_dict['X_val'].device).double()

X_batch = data_dict['X_val'][:128].double()
y_batch = data_dict['y_val'][:128]

# Complete the implementation of softmax_loss_naive and implement a (naive)
# version of the gradient that uses nested loops.
loss, grad = softmax_loss_naive(W, X_batch, y_batch, reg=0.0)

# As a rough sanity check, our loss should be something close to log(10.0).
print('loss: %f' % loss)
print('sanity check: %f' % (math.log(10.0)))

# Next, we use gradient checking to debug the analytic gradient of our naive softmax loss function. If you've implemented the gradient correctly, you should see relative errors less than `1e-6`.
# 

# In[ ]:


coutils.utils.fix_random_seed()
W = 0.0001 * torch.randn(3073, 10, device=data_dict['X_val'].device).double()
X_batch = data_dict['X_val'][:128].double()
y_batch = data_dict['y_val'][:128]

loss, grad = softmax_loss_naive(W, X_batch, y_batch, reg=0.0)

f = lambda w: softmax_loss_naive(w, X_batch, y_batch, reg=0.0)[0]
grad_check_sparse(f, W, grad, 10)

# Let's perform another gradient check with regularization enabled. Again you should see relative errors less than `1e-6`.

# In[ ]:


coutils.utils.fix_random_seed()
W = 0.0001 * torch.randn(3073, 10, device=data_dict['X_val'].device).double()
reg = 10.0

X_batch = data_dict['X_val'][:128].double()
y_batch = data_dict['y_val'][:128]

loss, grad = softmax_loss_naive(W, X_batch, y_batch, reg)

f = lambda w: softmax_loss_naive(w, X_batch, y_batch, reg)[0]
grad_check_sparse(f, W, grad, 10)


# Then, let's move on to the vectorized form

# In[ ]:


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.  When you implment the
    regularization over W, please DO NOT multiply the regularization term by 1/2
    (no coefficient).

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = torch.zeros_like(W)

    softmax = lambda Z: torch.exp(Z) / torch.exp(Z).sum(dim=1, keepdim=True)

    C = W.shape[1]
    M = X.shape[0]
    idx0 = range(M)

    Z = X.matmul(W)
    assert Z.shape == (M, C)

    Z -= torch.max(Z, dim=1, keepdim=True)[0]
    y_hat = softmax(Z)

    loss = -torch.log(y_hat[idx0, y]).sum()

    loss /= M
    loss += reg * (W ** 2).sum()

    y_hat[idx0, y] -= 1
    dW = X.t().matmul(y_hat)

    dW /= M
    dW += reg * 2 * W

    return loss, dW


# Now that we have a naive implementation of the softmax loss function and its gradient, implement a vectorized version in softmax_loss_vectorized. The two versions should compute the same results, but the vectorized version should be much faster.
# 
# The differences between the naive and vectorized losses and gradients should both be less than `1e-6`, and your vectorized implementation should be at least 100x faster than the naive implementation.

# In[ ]:


coutils.utils.fix_random_seed()
W = 0.0001 * torch.randn(3073, 10, device=data_dict['X_val'].device)
reg = 0.05

X_batch = data_dict['X_val'][:128]
y_batch = data_dict['y_val'][:128]

# Run and time the naive version
torch.cuda.synchronize()
tic = time.time()
loss_naive, grad_naive = softmax_loss_naive(W, X_batch, y_batch, reg)
torch.cuda.synchronize()
toc = time.time()
ms_naive = 1000.0 * (toc - tic)
print('naive loss: %e computed in %fs' % (loss_naive, ms_naive))

# Run and time the vectorized version
torch.cuda.synchronize()
tic = time.time()
loss_vec, grad_vec = softmax_loss_vectorized(W, X_batch, y_batch, reg)
torch.cuda.synchronize()
toc = time.time()
ms_vec = 1000.0 * (toc - tic)
print('vectorized loss: %e computed in %fs' % (loss_vec, ms_vec))

# we use the Frobenius norm to compare the two versions of the gradient.
loss_diff = (loss_naive - loss_vec).abs().item()
grad_diff = torch.norm(grad_naive - grad_vec, p='fro')
print('Loss difference: %.2e' % loss_diff)
print('Gradient difference: %.2e' % grad_diff)
print('Speedup: %.2fX' % (ms_naive / ms_vec))

# Let's check that your implementation of the softmax loss is numerically stable.
# 
# If either of the following print `nan` then you should double-check the numeric stability of your implementations.

device = data_dict['X_train'].device
dtype = torch.float32
D = data_dict['X_train'].shape[1]
C = 10

W_ones = torch.ones(D, C, device=device, dtype=dtype)
W, loss_hist = train_linear_classifier(softmax_loss_naive, W_ones,
                                       data_dict['X_train'],
                                       data_dict['y_train'],
                                       learning_rate=1e-8, reg=2.5e4,
                                       num_iters=1, verbose=True)

W_ones = torch.ones(D, C, device=device, dtype=dtype)
W, loss_hist = train_linear_classifier(softmax_loss_vectorized, W_ones,
                                       data_dict['X_train'],
                                       data_dict['y_train'],
                                       learning_rate=1e-8, reg=2.5e4,
                                       num_iters=1, verbose=True)

# Now lets train a softmax classifier with some default hyperparameters:

# fix random seed before we perform this operation
coutils.utils.fix_random_seed(10)

torch.cuda.synchronize()
tic = time.time()

W, loss_hist = train_linear_classifier(softmax_loss_vectorized, None,
                                       data_dict['X_train'],
                                       data_dict['y_train'],
                                       learning_rate=1e-10, reg=2.5e4,
                                       num_iters=1500, verbose=True)

torch.cuda.synchronize()
toc = time.time()
print('That took %fs' % (toc - tic))

# Plot the loss curve:

plt.plot(loss_hist, 'o')
plt.xlabel('Iteration number')
plt.ylabel('Loss value')
plt.show()

# Let's compute the accuracy of current model. It should be less than 10%.


# evaluate the performance on both the training and validation set
y_train_pred = predict_linear_classifier(W, data_dict['X_train'])
train_acc = 100.0 * (data_dict['y_train'] == y_train_pred).float().mean().item()
print('training accuracy: %.2f%%' % train_acc)
y_val_pred = predict_linear_classifier(W, data_dict['X_val'])
val_acc = 100.0 * (data_dict['y_val'] == y_val_pred).float().mean().item()
print('validation accuracy: %.2f%%' % val_acc)


# Now use the validation set to tune hyperparameters (regularization strength and learning rate). You should experiment with different ranges for the learning rates and regularization strengths.

class Softmax(LinearClassifier):
    """ A subclass that uses the Softmax + Cross-entropy loss function """

    def loss(self, W, X_batch, y_batch, reg):
        return softmax_loss_vectorized(W, X_batch, y_batch, reg)


results = {}
best_val = -1
best_softmax = None

learning_rates = []  # learning rate candidates
regularization_strengths = []  # regularization strengths candidates

# As before, store your cross-validation results in this dictionary.
# The keys should be tuples of (learning_rate, regularization_strength) and
# the values should be tuples (train_accuracy, val_accuracy)
results = {}

num_iters = 2000
num_model = 5
for i in range(num_model):
    lr = 10 ** uniform(-.5, -1.5)
    reg = 10 ** uniform(-4, -5)

    train_args = (data_dict['X_train'], data_dict['y_train'], lr, reg,
                  num_iters, 128, False)

    model = Softmax()
    model.train(*train_args)

    y_train_pred = model.predict(data_dict['X_train'])
    y_val_pred = model.predict(data_dict['X_val'])

    train_acc = 100.0 * (data_dict['y_train'] == y_train_pred).float().mean().item()
    val_acc = 100.0 * (data_dict['y_val'] == y_val_pred).float().mean().item()
    results[(lr, reg)] = (train_acc, val_acc)

    if val_acc > best_val:
        best_val = val_acc
        best_softmax = model

# Print out results.
for lr, reg in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg)]
    print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
        lr, reg, train_accuracy, val_accuracy))

print('best validation accuracy achieved during cross-validation: %f' % best_val)

# Run the following to visualize your cross-validation results:


x_scatter = [math.log10(x[0]) for x in results]
y_scatter = [math.log10(x[1]) for x in results]

# plot training accuracy
marker_size = 100
colors = [results[x][0] for x in results]
plt.scatter(x_scatter, y_scatter, marker_size, c=colors, cmap='viridis')
plt.colorbar()
plt.xlabel('log learning rate')
plt.ylabel('log regularization strength')
plt.title('CIFAR-10 training accuracy')
plt.gcf().set_size_inches(8, 5)
plt.show()

# plot validation accuracy
colors = [results[x][1] for x in results]  # default size of markers is 20
plt.scatter(x_scatter, y_scatter, marker_size, c=colors, cmap='viridis')
plt.colorbar()
plt.xlabel('log learning rate')
plt.ylabel('log regularization strength')
plt.title('CIFAR-10 validation accuracy')
plt.gcf().set_size_inches(8, 5)
plt.show()

# Them, evaluate the performance of your best model on test set. To get full credit for this assignment you should achieve a test-set accuracy above 0.36.


y_test_pred = best_softmax.predict(data_dict['X_test'])
test_accuracy = torch.mean((data_dict['y_test'] == y_test_pred).float())
print('softmax on raw pixels final test set accuracy: %f' % (test_accuracy,))

# Finally, let's visualize the learned weights for each class


w = best_softmax.W[:-1, :]  # strip out the bias
w = w.reshape(3, 32, 32, 10)
w = w.transpose(0, 2).transpose(1, 0)

w_min, w_max = torch.min(w), torch.max(w)

classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
for i in range(10):
    plt.subplot(2, 5, i + 1)

    # Rescale the weights to be between 0 and 255
    wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
    plt.imshow(wimg.type(torch.uint8).cpu())
    plt.axis('off')
    plt.title(classes[i])
