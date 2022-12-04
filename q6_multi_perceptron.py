
from itertools import combinations # import the combinations library.

import numpy as np # import the numpy library

MIN_EPOCHS = 2
MAX_EPOCHS = 50

def polynomial_kernel(x, y, p):

    """
    
    This function computes the polynomial kernel of degree p between two vectors x and y.
    @param x: a vector
    @param y: a vector
    @return: the polynomial kernel of degree p between x and y
    
    """

    x = np.array(x)
    y = np.array(y)
    return (np.dot(x, y.T)) ** p

class MultiClassPerceptronOvO:
  """
  One-versus-one (OvO) kernel perceptron
  """

  def __init__(self, num_classes):

    """

    This function initialises the MultiClassPerceptronOvO class.
    
    @param: num_classes: number of classes for the custom kernel peceptron algorithm. 
    
    """
    self.num_classes = num_classes
    self.num_classifiers = int(num_classes * (num_classes - 1) / 2)

    self.class_combinations = list(combinations(range(num_classes), 2))
    self.class_combinations_to_alpha_index = dict(zip(self.class_combinations, range(self.num_classifiers)))

    # Initialised by set_polynomial_kernel method
    self.kernel = None

    # Initialised by train method
    self.X_train = None
    self.alpha = None

  def __get_alpha_index(self, i, j):
    """
    Get the row index for two classes i, j of a classifier in the alpha matrix
    """
    key = tuple(sorted([i, j]))
    return self.class_combinations_to_alpha_index[key]
  
  def polynomial_fitting(self, degree):

        """
        
        @param: degree is the degree of the polynomial to be fitted.
        """
        self.kernel = lambda a, b: polynomial_kernel(a, b, degree) # set the kernel to the polynomial kernel.

  def __predict(self, K_matrix, example):
    """
    Predict the class of an example as the one with the most votes over all
    classifiers
    """
    confidences = np.dot(self.alpha, K_matrix[:, example])
    binary_predictions = np.sign(confidences)
    predicted_class_indices = np.clip(binary_predictions, a_min=0, a_max=None).astype(int)

    multiclass_predictions = np.empty((self.num_classifiers)).astype(int)

    # Track the prediction of each classifier
    for classifier, predicted_class_index in enumerate(predicted_class_indices.tolist()):
      multiclass_predictions[classifier] = self.class_combinations[classifier][predicted_class_index]

    # Choose the class with the most votes
    y_hat = np.bincount(multiclass_predictions).argmax()

    return y_hat, multiclass_predictions

  def train(self, X_train, y_train):

    X_train = np.array(X_train)

    y_train = np.array(y_train)

    # Initialisation
    num_examples = X_train.shape[0]
    self.X_train = X_train
    self.alpha = np.zeros((self.num_classifiers, num_examples))

    # Compute Gram matrix, K
    gram_matrix = self.kernel(X_train, X_train)

    # Training loop
    running_mistakes = 0
    train_accuracy_list = []

    converged = False
    epoch = 0

    while (converged == False) and (epoch <= MAX_EPOCHS):

      running_mistakes = 0

      # Shuffle the training data
      shuffled_indices = [*range(num_examples)]
      np.random.shuffle(shuffled_indices)

      for example in shuffled_indices:
        x = X_train[example]
        y = y_train[example]
        
        # Predict (most votes)
        y_hat, multiclass_predictions = self.__predict(gram_matrix, example)

        # Check if prediction is incorrect
        if (y_hat != y):
          running_mistakes += 1

        # Update relevant (i.e. binary prediction including label y) incorrect classifiers;
        # skip if classifier is correct or does not make prediction including label y
        for index, (i, j) in enumerate(self.class_combinations_to_alpha_index.keys()):
          prediction = multiclass_predictions[index]

          if y == i and prediction != y: # If so, y is the negative class for an incorrect classifier 
            self.alpha[index, example] -= 1
          elif y == j and prediction != y: # If so, y is the positive class for an incorrect classifier
            self.alpha[index, example] += 1
        
      # Calculate training accuracy
      train_accuracy = (num_examples - running_mistakes) / float(num_examples)
      train_accuracy_list.append(train_accuracy)

      # Early stopping
      if epoch >= MIN_EPOCHS:
        if (np.mean(train_accuracy_list[-2:]) - np.mean(train_accuracy_list[-4:-2])) < 0.01:
          converged = True

          # Count number of non-zero alpha vector elements per class
          print(f"Number of non-zero alpha vector elements per class: {np.count_nonzero(self.alpha, axis=1)}")
      
      epoch += 1
                
    return train_accuracy_list

  def test(self, X_test, y_test):

    X_test = np.array(X_test)

    y_test = np.array(y_test)

    num_examples = X_test.shape[0]

    # Compute kernel matrix, K
    K_matrix = self.kernel(self.X_train, X_test)

    running_mistakes = 0
    confusion_matrix = np.zeros((self.num_classes, self.num_classes))

    # Make a prediction for each example
    for example in range(num_examples):
      y = y_test[example]

      # Predict (most votes)
      y_hat, _ = self.__predict(K_matrix, example)

      # Track mistakes
      if (y_hat != y):
        running_mistakes += 1
        confusion_matrix[y][y_hat] += 1
    
    # Calculate test accuracy
    test_accuracy = (num_examples - running_mistakes) / float(num_examples)
    
    return test_accuracy, confusion_matrix