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
        a.set_ylim([0, 40])
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

    # generate random x samples
    def generate_x(N, d):
        norm = np.random.normal
        normal_deviates = norm(size=(d, N))

        radius = np.sqrt((normal_deviates ** 2).sum(axis=0))
        x = normal_deviates / radius

        #fig, ax = plt.subplots(subplot_kw=dict(projection='3d'))
        #ax.scatter(*x)
        #ax.set_aspect('auto')
        #plt.show()
        return x

    # %%

    # calculate y
    def generate_y(x, sigma):
        ones_array = np.ones(d)
        y = []
        for x_n in x.T:
            y_first = np.power(0.25 + np.power((ones_array.T * x_n), 2), -1)
            y_second = np.random.normal(0, np.power(sigma, 2), len(x_n))
            y.append(y_first + y_second)

        return np.array(y)

    def generate_v(M, d):
        vec = np.random.randn(d, M)
        vec /= np.linalg.norm(vec, axis=0)
        return vec

    def generate_phi(x, v, M):
        phi = []
        for x_n in x.T:
            lol = []
            for v_m in v.T:
                lol.append(v_m.T*x_n)

            phi.append(1 / np.sqrt(M) * (np.max(lol, 0)))

        return np.array(phi)

    def qr_inv(A, b):
        Q, R = np.linalg.qr(A)
        z = Q.T @ b
        return np.linalg.solve(R, z)

    def calc_w(y, l, phi):
        I = np.identity(len(phi[0]))
        w_star_first = np.dot(phi.T, phi) + l * I
        w_star_second = np.dot(phi.T, y)
        w_star = qr_inv(w_star_first, w_star_second)

        return w_star

    def predict(x, w):
        y_hat = np.dot(x.T, w)
        return y_hat

    def mse(N, y, y_hat):
        for n in range(0,N):
            sum = np.power(y[n] - y_hat[n], 2)

        mse = sum/N
        return mse
    # %%

    def calc_result(N_train, N_test, d, sigma, M):
        x_train = generate_x(N_train, d=d)
        y_train = generate_y(x_train, sigma=sigma)

        x_test = generate_x(N_test, d=d)
        y_test = generate_y(x_test, sigma=sigma)

        # %%

        v = generate_v(M, d)

        phi_train = generate_phi(x_train, v, M)
        phi_test = generate_phi(x_test, v, M)

        w_star_train = calc_w(y=y_train, l=1e-8, phi=phi_train)
        w_star_test = calc_w(y=y_test, l = 1e-8, phi=phi_test)

        mse_train = mse(N_train, y_train, predict(x_train, w_star_train))
        mse_test = mse(N_test, y_test, predict(x_test, w_star_test))

        return mse_train, mse_test


    N_train = 200
    N_test = 50
    # dim = 2
    d = 5
    sigma = 2
    M = 50
    r = 5

    M_7 = [10 * 0 + 1, 10 * 7 + 1, 10 * 25 + 1, 10 * 37 + 1, 10 * 60 + 1]

    avg_train_loss = []
    avg_test_loss = []

    for M in M_7:
        train_loss_array = []
        test_loss_array = []
        for i in range (0, 5):
            train_loss, test_loss = calc_result(N_train, N_test, d, sigma, M)
            train_loss_array.append(train_loss)
            test_loss_array.append(test_loss)
        avg_train_loss.append(np.average(train_loss_array))
        avg_test_loss.append(np.average(test_loss_array))

    print("Average train loss = " + str(avg_train_loss))
    print("Average test loss = " + str(avg_test_loss))

    print ("hi")


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

