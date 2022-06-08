""" Python code submission file.

IMPORTANT:
- Do not include any additional python packages.
- Do not change the interface and return values of the task functions.
- Only insert your code between the Start/Stop of your code tags.
- Prior to your submission, check that the pdf showing your plots is generated.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime
from matplotlib.backends.backend_pdf import PdfPages

def load_data():
    """ General utility function to load provided .npy data

        For datasets A and B:
        - load train/test data (N/N_t samples)
        - the first 2 columns are input x and the last column is target y for all N/N_t samples
        - include transform to homogeneous input data
    """


    """data_without_targets = [
        {'a_test': data_a_test[:, :2]},
        {'a_train': data_a_train[:, :2]},
        {'b_train': data_b_train[:, :2]},
        {'b_test': data_a_test[:, :2]}
    ]"""
    """ Start of your code 
    """
    data_a_train, data_a_test = np.load('./data/data_a_train.npy'), np.load('./data/data_a_test.npy')
    data_b_train, data_b_test = np.load('./data/data_b_train.npy'), np.load('./data/data_b_test.npy')

    #homogeneous transformation
    data_a_test = np.insert(data_a_test.T, 0, [np.ones(data_a_test.shape[0])], axis=0).T
    data_a_train = np.insert(data_a_train.T, 0, [np.ones(data_a_train.shape[0])], axis=0).T
    data_b_train = np.insert(data_b_train.T, 0, [np.ones(data_b_train.shape[0])], axis=0).T
    data_b_test = np.insert(data_b_test.T, 0, [np.ones(data_b_test.shape[0])], axis=0).T
    """ End of your code
    """

    return data_a_train, data_a_test, data_b_train, data_b_test


def qr_inv(A, b):
    Q,R = np.linalg.qr(A)
    z = Q.T@b
    return np.linalg.solve(R,z)


def plot(ax, x, y):
    ax.scatter(x[y == -1][:, 1], x[y == -1][:, 2], color='r', label='false')
    ax.scatter(x[y == 1][:, 1], x[y == 1][:, 2], color='b', label='true')
    
def quadratic():
    """ Subtask 1: Quadratic Loss as Convex Surrogate in Binary Classification

        Requirements for the plot:
        - plot each of the groundtruth test data A and predicted test data A in a scatterplot into one subplot
        - indicate the two classes with 2 different colors for both subplots
        - use a legend
    """
    fig, ax = plt.subplots(1, 2, figsize=(10,5))
    plt.suptitle('Task 1 - Quadratic Loss as Convex Surrogate in Binary Classification', fontsize=12)
    ax[0].set_title('Test Data A')
    ax[1].set_title('Test Predictions A')

    """ Start of your code 
    """

    a_train, a_test, _, _ = load_data()
    x_train = a_train[:, :3]
    x_test = a_test[:, :3]
    y_train = np.squeeze(a_train[:, 3:])
    y_test = np.squeeze(a_test[:, 3:])

    w = qr_inv(x_train.T @ x_train, x_train.T @ y_train)

    train_prediction, test_prediction = np.sign(x_train @ w), np.sign(x_test @ w)
    accuracy_train = np.mean([True if y_train[i] == train_prediction[i] else False for i in range(0,train_prediction.shape[0])])
    accuracy_test = np.mean([True if y_test[i] == test_prediction[i] else False for i in range(0,test_prediction.shape[0])])

    ax[0].scatter(x_test[y_test == 1][:, 1], x_test[y_test == 1][:, 2], color='orange', label='True')
    ax[0].scatter(x_test[y_test == -1][:, 1], x_test[y_test == -1][:, 2], color='blue', label='False')

    ax[1].scatter(x_test[test_prediction == 1][:, 1], x_test[test_prediction == 1][:, 2], color='orange', label='True')
    ax[1].scatter(x_test[test_prediction == -1][:, 1], x_test[test_prediction == -1][:, 2], color='blue', label='False')

    print(f'Train Accuracy: {accuracy_train}')
    print(f'Test Accuracy: {accuracy_test}')
    """ End of your code
    """

    ax[0].legend()
    ax[1].legend()
    return fig

def logistic():
    """ Subtask 2: Logistic Loss as Convex Surrogate in Binary Classification

        Requirements for the plot:
        - the first subplot should contain the energy of the objective function plotted over the iterations
        - the other two subplots should contain groundtruth test data A and predictions of test data A, respectively, in a scatterplot
            - indicate the two classes with 2 different colors
            - use a legend
    """
    fig, ax = plt.subplots(1, 3, figsize=(15,5))
    plt.suptitle('Task 2 - Logistic Loss as Convex Surrogate in Binary Classification', fontsize=12)

    ax[0].set_title('Energy $E(\widetilde{\mathbf{w}})$')
    ax[1].set_title('Test Data A')
    ax[2].set_title('Test Predictions A')

    """ Start of your code 
    """

    #1.
    a_train, a_test, _, _ = load_data()
    x_train = a_train[:, :3]
    x_test = a_test[:, :3]
    y_train = np.squeeze(a_train[:, 3:])
    y_test = np.squeeze(a_test[:, 3:])

    w_tilde = np.random.randn(3)
    epsilon = 1.49e-08


    def error_function(w_tilde, x, y):
        exponential_term = np.exp(-y * (w_tilde @ x.T))
        term_1 = np.log(1+exponential_term)
        return np.sum(term_1)
    approximation = approx_fprime(w_tilde, error_function, epsilon, *[x_train, y_train])
    print(f'Approximation: {approximation}')

    def gradient(w_tilde,x,y):
        exp_term = np.exp(-y * (x @ w_tilde))
        term_1 = 1 / (1+exp_term)
        term_2 = term_1 * exp_term
        term_3 = np.sum(term_2.reshape(term_2.shape[0],1)*(-y.reshape(y.shape[0],1)*x), axis=0)
        return term_3

    grad = gradient(w_tilde, x_train, y_train)
    print(f'Gradient: {grad}')

    # 2.
    def calc_lipschitz(x):
        sum_term = x @ x.T
        sigma, _ = np.linalg.eigh(sum_term)
        return (1/4) * np.max(sigma)

    L = calc_lipschitz(x_train)


    loss_values = []
    old_loss = 100000
    latest_loss = 0
    k = 1
    w_tilde_previous, next_w = np.zeros_like(w_tilde), np.zeros_like(w_tilde)
    while abs(old_loss - latest_loss) > 0.001:
        old_loss = latest_loss
        beta = (k-1) / (k+1)
        w_slash = w_tilde + beta*(w_tilde - w_tilde_previous)
        w_tilde_previous = w_tilde
        next_w = w_slash - (1/L)*gradient(w_slash, x_train, y_train)
        w_tilde = next_w
        latest_loss = error_function(next_w, x_train, y_train)
        loss_values.append(latest_loss)

    sigma = lambda x: 1 / (1 + np.exp(-x))

    train_predictions = [1 if sigma(i) > 0.5 else -1 for i in (x_train @ next_w)]
    test_predictions = [1 if sigma(i) > 0.5 else -1 for i in (x_test @ next_w)]

    accuracy_train = np.mean([y_train == train_predictions])
    accuracy_test = np.mean([y_test == test_predictions])


    print(f'Train Accuracy: {accuracy_train}')
    print(f'Test Accuracy: {accuracy_test}')

    #3.
    ax[0].plot(loss_values)

    ax[1].scatter(x_test[y_test == 1][:, 1],
                  x_test[y_test == 1][:, 2], color='orange', label='True')
    ax[1].scatter(x_test[y_test == -1][:, 1], x_test[y_test == -1][:, 2], color='blue', label='False')

    ax[2].scatter(x_test[np.array(test_predictions) == 1][:, 1],
                  x_test[np.array(test_predictions) == 1][:, 2], color='orange', label='True')
    ax[2].scatter(x_test[np.array(test_predictions) == -1][:, 1],
                  x_test[np.array(test_predictions) == -1][:, 2], color='blue', label='False')

    """ End of your code
    """

    ax[1].legend()
    ax[2].legend()
    return fig

def svm_primal():
    """ Subtask 3: Hinge Loss as Convex Surrogate in Binary Classification

        Requirements for the plot:
        - the first subplot should contain the energy of the objective function plotted over the iterations
        - the next two subplots should contain predictions of train data A and test data A, respectively, in a scatterplot
            - indicate the two predicted classes with 2 different colors for both data sets 
            - use a legend
            - for both train and test data also plot the separating hyperplane and the margin at \pm 1
            - for the train data include the support vectors
    """
    fig, ax = plt.subplots(1,3, figsize=(15,5))
    plt.suptitle('Task 3 - Hinge Loss as Convex Surrogate in Binary Classification', fontsize=12)

    title_list = ['Energy', 'Train Predictions A', 'Test Predictions A']
    for idx, a in enumerate(ax.reshape(-1)):
        a.set_title(title_list[idx])

    """ Start of your code 
    """

    def nine(w, lamb, x_train, y_train):
        first = (lamb/2) * np.linalg.norm(w)
        sum = np.maximum(0, 1 - (y_train * (x_train @ w)))

        return first + np.mean(sum)


    def calc_g(w, x_train, y_train):
        assert len(x_train) == len(y_train), 'both must be of size n'
        g = -np.expand_dims(y_train, axis=-1) * x_train
        cond = y_train * (x_train @ w) >= 1

        g[cond] = 0

        return np.mean(g, axis=0)


    data_a_train, data_a_test, _ ,_ = load_data()

    alpha = 10e-2
    lamb = 0.005
    nr_weights = 3
    w = np.zeros(nr_weights)
    w_tilde = w
    x_train = data_a_train[:, :-1]
    y_train = data_a_train[:, -1]
    x_test = data_a_test[:, :-1]
    y_test = data_a_test[:, -1]


    approx_grad = approx_fprime(w, nine, np.sqrt(np.finfo(float).eps), 0, x_train, y_train)

    calc_grad = calc_g(w, x_train, y_train)

    print(approx_grad)
    print(calc_grad)

    hinge_loss = []
    for i in range(350):
        w_tilde = w_tilde - alpha * calc_g(w_tilde, x_train, y_train)
        w_tilde = w_tilde / (1 + lamb*alpha)

        hinge_loss.append(nine(w=w_tilde, lamb=lamb, x_train=x_train, y_train=y_train))

    print(w_tilde)

    ax[0].plot(hinge_loss)

    train_pred = np.sign(x_train @ w_tilde)
    test_pred = np.sign(x_test @ w_tilde)

    print(np.sign(x_train @ w_tilde))

    def calcuate_accuracy(y, y_pred):
        return np.mean(y == y_pred)

    train_acc = calcuate_accuracy(y_train, train_pred)
    test_acc = calcuate_accuracy(y_test, test_pred)

    print("Train accuarcy: " + str(train_acc) + "Test accuarcy: " + str(test_acc))


    plt.show()
    """ End of your code
    """

    ax[1].legend()
    ax[2].legend()
    return fig

def svm_dual():
    """ Subtask 4: Dual SVM

        Requirements for the plot:
        - the first subplot should contain the energy of the objective function plotted over the iterations
        - the next two subplots should contain predictions of train data B and test data B, respectively, in a scatterplot
            - indicate the two predicted classes with 2 different colors for both data sets 
            - use a legend
            - for both train and test data also plot the separating hyperplane and the margin at \pm 1
            - for the train data include the support vectors
    """

    fig, ax = plt.subplots(1,3, figsize=(15,5))
    plt.suptitle('Task 4 - Dual Support Vector Machine', fontsize=12)
    
    ax[0].set_title('Energy $D(\mathbf{a})$')
    ax[1].set_title('Train Predictions B')
    ax[2].set_title('Test Predictions B')

    """ Start of your code 
    """


    """ End of your code
    """

    ax[1].legend()
    ax[2].legend()
    return fig

if __name__ == '__main__':
    # load train/test datasets A and B globally
    data_a_train, data_a_test, data_b_train, data_b_test = load_data()

    tasks = [quadratic, logistic, svm_primal, svm_dual]
    pdf = PdfPages('figures.pdf')
    for task in tasks:
        f = task()
        pdf.savefig(f)

    pdf.close()
