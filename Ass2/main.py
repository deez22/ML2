#%%
""" Python code submission file.

IMPORTANT:
- Do not include any additional python packages.
- Do not change the interface and return values of the task functions.
- Only insert your code between the Start/Stop of your code tags.
- Prior to your submission, check that the pdf showing your plots is generated.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from typing import List

def task12():
    """ Subtask 1: Least Squares and Double Descent Phenomenon

        Requirements for the plot:
        - make one subplot for each lambda
        - each subplot should contain mean and std. of train/test errors
        - labels for mean train/test errors are 'mean train error', 'mean test error' and must be included in the plots

        Subtask 2: Dual Representation with Kernel

        Requirements for the plots:
        - make one subplot for each M
        - each subplot should contain the n=10th row of both the kernel matrix and the feature product \Phi\Phi^T
        - labels should be "features" and "kernel" and must be included in a legend
        - each subplot must contain a title with the number of random features and the mean absolute difference between kernel and feature product.
    """

    fig1, ax1 = plt.subplots(1, 3, figsize=(17, 5))
    plt.suptitle('Task 1 - Regularized Least Squares and Double Descent Phenomenon', fontsize=16)
    for a in ax1.reshape(-1):
        a.set_ylim([0, 400])
        a.set_ylabel('error')
        a.set_xlabel('number of random features')

    fig2, ax2 = plt.subplots(1, 3, figsize=(15, 5))
    plt.suptitle('Task 2 - Dual Representation with Kernel', fontsize=16)

    lams = [1e-8, 1e-5, 1e-3]  # use this for subtask 1
    m_array = [10, 200, 800]  # use this for subtask 2
    mae_array = 1e3 * np.ones((3))  # use this for subtask 2 (MAE = mean absolute error)

    """ Start of your code 
    """

    # %%
    N_train = 200
    N_test = 50
    # dim = 2
    d = 5
    sigma = 2
    M = 50
    r = 5

    # %%

    # generate random x samples
    def generate_x(N, d):
        norm = np.random.normal
        normal_deviates = norm(size=(d, N))

        radius = np.sqrt((normal_deviates ** 2).sum(axis=0))
        x = normal_deviates / radius

        return x

    # %%

    # generate random y samples
    def generate_y(x, sigma):
        ones_array = np.ones(d)
        y = []
        for x_n in x.T:
            y_first = np.power(0.25 + np.power((ones_array.T @ x_n), 2), -1)
            y_second = np.random.normal(0, np.power(sigma, 2), 1)
            y.append(y_first + y_second)

        return np.array(y)

    def generate_v(M, d):
        vec = np.random.randn(d, M)
        vec /= np.linalg.norm(vec, axis=0)
        return vec

    def calc_phi(x, v, M):
        term_1 = v.T @ x.T
        projected = (1 / np.sqrt(M)) * np.maximum(term_1, 0, term_1).T

        assert((projected.shape[0] == N_train or projected.shape[0] == N_test)  and projected.shape[1] == M)
        return projected

    def qr_inv(A, b):
        Q, R = np.linalg.qr(A)
        z = Q.T @ b
        return np.linalg.solve(R, z)

    def calc_w(y, l, phi, M):
        I = np.identity(M)
        w_star_first = np.dot(phi.T, phi) + l * I
        w_star_second = phi.T @ y  # np.dot(phi.T, y)
        w_star = qr_inv(w_star_first, w_star_second)

        return w_star

    def predict(x, w):
        y_hat = x @ w
        return y_hat

    def mse(N, y, y_hat):
        sum = 0
        for n in range(0, N):
            sum += np.power(y[n] - y_hat[n], 2)

        mse = sum / N
        return mse

    # %%

    def train_test_data(N_train, N_test, d, sigma):
        x_train = generate_x(N_train, d=d)
        y_train = generate_y(x_train, sigma=sigma)

        x_test = generate_x(N_test, d=d)
        y_test = generate_y(x_test, sigma=sigma)
        return x_train.T, y_train, x_test.T, y_test

    def calc_train_and_test_mse(x_train, y_train, x_test, y_test, M, lamb):
        # %%
        v = generate_v(M, d)
        phi = calc_phi(x_train, v, M)
        phi_test = calc_phi(x_test, v, M)
        w_star = calc_w(y=y_train, l=lamb, phi=phi, M=M)
        mse_train = mse(N_train, y_train, predict(phi, w_star))
        mse_test = mse(N_test, y_test, predict(phi_test, w_star))
        return mse_train, mse_test

    M_7 = [10 * k + 1 for k in range(0, 60)]

    x_train, y_train, x_test, y_test = train_test_data(N_train, N_test, d, sigma)

    axis_iterator = 0
    for lamb in lams:
        avg_train_loss = []
        std_train_loss = []
        avg_test_loss = []
        std_test_loss = []
        for M in M_7:
            train_loss_array = []
            test_loss_array = []
            for i in range(0, 5):
                train_loss, test_loss = calc_train_and_test_mse(x_train, y_train, x_test, y_test, M, lamb)
                train_loss_array.append(train_loss)
                test_loss_array.append(test_loss)
            avg_train_loss.append(np.average(train_loss_array))
            std_train_loss.append(np.std(train_loss_array))
            avg_test_loss.append(np.average(test_loss_array))
            std_test_loss.append(np.std(test_loss_array))

        print("Average train loss = " + str(avg_train_loss))
        print("Average test loss = " + str(avg_test_loss))
        train_err, = ax1[axis_iterator].plot(avg_train_loss)
        ax1[axis_iterator].fill_between(np.linspace(0,59,60), np.array(avg_train_loss) - np.array(std_train_loss)
                                        , np.array(avg_train_loss) + np.array(std_train_loss),alpha=0.2)

        test_err, = ax1[axis_iterator].plot(avg_test_loss)
        ax1[axis_iterator].fill_between(np.linspace(0,59,60), np.array(avg_test_loss) - np.array(std_test_loss)
                                        , np.array(avg_test_loss) + np.array(std_test_loss),alpha=0.2)

        ax1[axis_iterator].set_title("Lambda = " + str(lamb))
        ax1[axis_iterator].legend(handles = [train_err, test_err], labels = ['train_error', 'test_error'])
        axis_iterator += 1

    plt.show()
    print("hi")


    """ End of your code
    """

    for lam_idx, a in enumerate(ax1.reshape(-1)):
        a.legend()
        a.set_title(r'$\lambda=$' + str(lams[lam_idx]))

    for m_idx, a in enumerate(ax2.reshape(-1)):
        a.legend()
        a.set_title('#Features M=%i, MAE=%f' % (m_array[m_idx], (mae_array[m_idx])))

    return fig1, fig2


if __name__ == '__main__':
    pdf = PdfPages('figures.pdf')
    f1, f2 = task12()

    pdf.savefig(f1)
    pdf.savefig(f2)
    pdf.close()

