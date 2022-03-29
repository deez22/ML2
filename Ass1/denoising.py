""" Python code submission file.

IMPORTANT:
- Do not include any additional python packages.
- Do not change the interface and return values of the task functions.
- Only insert your code between the Start/Stop of your code tags.
- Prior to your submission, check that the pdf showing your plots is generated.
"""

import numpy as np
import matplotlib.pyplot as plt
import medmnist
from matplotlib.backends.backend_pdf import PdfPages

def task2():
    """ Bayesian Denoising - 2D Toytask

        Requirements for the plots:
        - ax[0] should contain training data, the test data point, and the conditional mean/MAP using a Dirac prior.
        - ax[1-3] should contain training data, the test data point, the estimated pdf, and the conditional mean/MAP using the KDE prior for 3 different bandwidths h. 
    """
    fig, ax = plt.subplots(1, 4, figsize=(20,5))
    fig.suptitle('Task 2 - Bayesian Denoising of 2D Toytask', fontsize=16)
    ax[0].set_title(r'Dirac Prior')

    """ Start of your code
    """

    # 1.
    means = [[0, 0], [0,1.5], [1.5,1.5]]
    cov = [[0.075, 0], [0, 0.075]]
    sample_size = 300
    y = []
    x = []
    for mean in means:
        x_train, y_train = np.random.multivariate_normal(mean, cov, sample_size).T

        x.extend(x_train)
        y.extend(y_train)

    y = np.array([x,y]).T
    """
     if w is None:
        w = x.new_ones(x.shape[0])
    else:
        w = x.shape[0] * w

    d = x.data.size()[1] + 1
    n_s = x_i.data.size()[0]

    term_in_brackets = th.exp(-th.cdist(x, x_i) ** 2 / (2 * h ** 2))
    sum = th.sum(w * term_in_brackets, dim=1)
    fraction_term = 1 / (math.pow((2 * math.pi), (d / 2)) * n_s * math.pow(h, d))
    
    """


    # 2.
    #link for report: https://aakinshin.net/posts/kde-bw/
    h1 = 0.05  # don't forget to set the 3 chosen values h1,h2,h3 in your code
    h2 = 0.5
    h3 = 1

    N = sample_size * len(means)
    D = 2
    kde_list = []
    for h in [h1,h2,h3]:
        sum = 0
        f_kde = np.zeros_like(y)
        for n in range(0, N):
            """y_n = y[n]
            term_1 = (1 / np.power(np.sqrt(2 * np.pi) * h, D))
            dist = np.array([np.linalg.norm(i - y_n) for i in y])
            term_2 = np.exp(-dist ** 2 / (2 * h ** 2))
            sum += term_1 * term_2"""
            tmp = (y[n] - y) / h
            kernel = np.exp(-0.5 * tmp ** 2) / (np.sqrt(2 * np.pi) * h)
            f_kde[n] = kernel.sum() / (y.shape[0])
        kde = f_kde
        kde_list.append(kde)

    tmp_1, tmp_2 = zip(*y)
    ax[0].scatter(tmp_1,tmp_2)
    for i in  range(0, len(kde_list)):
        #ax[i+1].plot(y,kde_list[i])
        tmp_1, tmp_2 = zip(*kde_list[i])
        ax[i+1].hist2d(tmp_1, tmp_2)


    x_test = [0,1.5]

    for n in range (0, N):
        y_n = y[n]


    """ End of your code
    """

    ax[1].set_title(r'KDE Prior $h=$'+str(h1))
    ax[2].set_title(r'KDE Prior $h=$'+str(h2))
    ax[3].set_title(r'KDE Prior $h=$'+str(h3))
    for a in ax.reshape(-1):
        a.legend()

    return fig

def task3():
    """ Bayesian Image Denoising

        Requirements for the plots:
        - the first row should show your results for \sigma^2=0.1
        - the second row should show your results for \sigma^2=1.
        - arange your K images as a grid
    """

    fig, ax = plt.subplots(2, 4, figsize=(15,8))
    fig.suptitle('Task 3 - Bayesian Image Denoising', fontsize=16)

    ax[0,0].title.set_text(r'$\mathbf{y}^*$')
    ax[0,1].title.set_text(r'$\mathbf{x}$')
    ax[0,2].title.set_text(r'$\mathbf{\hat y}_{\operatorname{CM}}(\mathbf{x})$')
    ax[0,3].title.set_text(r'$\mathbf{\hat y}_{\operatorname{MAP}}(\mathbf{x})$')
    ax[0,0].set_ylabel(r'$\sigma^2=0.1$')
    ax[1,0].set_ylabel(r'$\sigma^2=1.$')

    for a in ax.reshape(-1):
        a.set_xticks([])
        a.set_yticks([])

    """ Start of your code
    """

    """ End of your code
    """
    return fig

if __name__ == '__main__':
    tasks = [task2, task3]

    pdf = PdfPages('figures.pdf')
    for task in tasks:
        f = task()
        pdf.savefig(f)

    pdf.close()