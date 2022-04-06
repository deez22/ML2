""" Python code submission file.

IMPORTANT:
- Do not include any additional python packages.
- Do not change the interface and return values of the task functions.
- Only insert your code between the Start/Stop of your code tags.
- Prior to your submission, check that the pdf showing your plots is generated.
"""
import numpy
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
    fig, ax = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle('Task 2 - Bayesian Denoising of 2D Toytask', fontsize=16)
    ax[0].set_title(r'Dirac Prior')

    """ Start of your code
    """

    # 1.
    means = [[0, 0], [0, 1.5], [1.5, 1.5]]
    cov = [[0.075, 0], [0, 0.075]]
    sample_size = 300
    y = []
    x = []
    z = []
    truth_count = 0
    for mean in means:
        x_train, y_train = np.random.multivariate_normal(mean, cov, sample_size).T
        x.extend(x_train)
        y.extend(y_train)
        z.extend([truth_count] * sample_size)
        truth_count += 1

    y = np.array([x, y]).T

    # 2.
    # link for report: https://aakinshin.net/posts/kde-bw/
    h1 = 0.01  # don't forget to set the 3 chosen values h1,h2,h3 in your code
    h2 = 0.1
    h3 = 0.4
    N = sample_size * len(means)
    D = 2
    kde_list = []

    def kde():
        for h in [h1, h2, h3]:
            f_kde = np.zeros_like(y)
            for n in range(0, N):
                tmp = (y - y[n]) / (2 * h)
                kernel = np.exp(-0.5 * tmp ** 2) / (np.sqrt(2 * np.pi) * h)
                f_kde[n] = kernel.sum() / (y.shape[0])
                kde = f_kde
            # kde = 1/N * sum
            kde_list.append(kde)

        ax[0].scatter(y.T[0], y.T[1], s=50, c=z)
        for i in range(0, len(kde_list)):
            kde = kde_list[i]
            ax[i + 1].scatter(y.T[0], y.T[1], s=50, c=kde.T[0])

        return kde_list

    # 3.
    x_test = [1.5, 0]

    # 4.
    def cmm(sigma, x_test, kde):
        argument_1 = 1 / (((2 * np.pi * sigma ** 2)) ** (D / 2))
        argument_2 = np.exp(-1 / (2 * sigma ** 2) * (y - x_test) ** 2)
        p_x_y_array = argument_1 * argument_2

        # 4.: (8.)
        p_y = kde
        p_x_y = p_x_y_array
        nominator = np.sum(y * p_y * p_x_y)
        denominator = np.sum(p_y * p_x_y)
        conditional_mean = nominator / denominator
        for i in range(0, len(kde_list)):
            ax[i + 1].scatter(x_test[0], x_test[1], c=conditional_mean)
        plt.show()
        return p_y, p_x_y, conditional_mean

    # 5.
    def map(p_y, p_x_y):
        y_max = np.unravel_index(y.argmax(), y.shape)
        y_map = p_y[y_max[0]][y_max[1]] * p_x_y[y_max[0]][y_max[1]]
        print(y_map)
        return y_map

    # 6
    kde_list = kde()
    p_y, p_x_y, conditional_mean = cmm(sigma=1, x_test=x_test, kde=kde_list[0])
    map = map(p_y, p_x_y)
    print("Conditional mean = " + str(conditional_mean))
    print("MAP = " + str(map))

    # 7

    """ End of your code
    """

    ax[1].set_title(r'KDE Prior $h=$' + str(h1))
    ax[2].set_title(r'KDE Prior $h=$' + str(h2))
    ax[3].set_title(r'KDE Prior $h=$' + str(h3))
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

    fig, ax = plt.subplots(2, 4, figsize=(15, 8))
    fig.suptitle('Task 3 - Bayesian Image Denoising', fontsize=16)

    ax[0, 0].title.set_text(r'$\mathbf{y}^*$')
    ax[0, 1].title.set_text(r'$\mathbf{x}$')
    ax[0, 2].title.set_text(r'$\mathbf{\hat y}_{\operatorname{CM}}(\mathbf{x})$')
    ax[0, 3].title.set_text(r'$\mathbf{\hat y}_{\operatorname{MAP}}(\mathbf{x})$')
    ax[0, 0].set_ylabel(r'$\sigma^2=0.1$')
    ax[1, 0].set_ylabel(r'$\sigma^2=1.$')

    for a in ax.reshape(-1):
        a.set_xticks([])
        a.set_yticks([])

    """ Start of your code
    """

    # 4
    M = 25
    N = 100

    dataset = np.load('bloodmnist.npz')
    dataset_images = dataset['train_images']
    dataset_labels = dataset['train_labels']

    random_numbers_train = np.random.randint(0, len(dataset_images), N)
    random_numbers_test = np.random.randint(0, len(dataset_images), M)
    y_train = []
    x_test = []

    y_train = np.empty((N,28,28,3), object)
    x_test = np.empty((25, 28, 28, 3), object)
    #normalize color
    y_train[:N] = dataset_images[:N] / 255
    x_test[:25] = dataset_images[1001:1026] / 255

    #normalize color



    # 5
    mean = 0
    x_test_noise = []
    for test_image in x_test:
        sigma = np.random.uniform(0.1, 1)
        test_image_flat = test_image.flatten()
        gaussian_noise = np.random.normal(mean, sigma, test_image_flat.shape)
        test_image_flat_noise = test_image_flat + gaussian_noise
        x_test_noise.append(
            test_image_flat_noise.reshape(test_image.shape[0], test_image.shape[1], test_image.shape[2]))

    # 6
    def logsumexp_stable(x):
        xmax = np.max(x)
        return xmax + np.log(np.sum(np.exp((x - xmax).astype(float))))

    def log_kde(y_train = y_train, h: float = 0.4):
        f_kde = np.zeros_like(y_train)
        N = y_train.shape[0]
        term_1 = numpy.log(N)
        D = 28*28*3
        term_2 = (D/2)*numpy.log(2*np.pi)
        term_3 = (D*numpy.log(h))


        for n in range(0, N):
            tmp = (y_train - y_train[n]) / (2 * h)
            kernel = -term_1 - term_2 - term_3 - (0.5 * tmp ** 2)  #/ (np.sqrt(2 * np.pi) * h)
            f_kde[n] = logsumexp_stable(kernel)/(y_train.shape[0])
        return f_kde

    result = log_kde()


    """h1 = 0.5
    kde_list = []
    def kde():
        for h in [h1]:
            f_kde = np.zeros_like(y_train)
            for n in range(0, N):
                tmp = (y_train - y_train[n]) / (2*h)
                kernel = np.exp(-0.5 * tmp**2) / (np.sqrt(2 * np.pi) * h)
                f_kde[n] = kernel.sum() / (y_train.shape[0])
                kde = f_kde
            #kde = 1/N * sum
            kde_list.append(kde)

        ax[0].scatter(y_train.T[0], y_train.T[1], s=50, c=dataset_labels)
        for i in  range(0, len(kde_list)):
            kde = kde_list[i]
            ax[i+1].scatter(y.T[0], y.T[1], s=50, c=kde.T[0])

        return kde_list"""

    print("hello")
    """ End of your code
    """
    return fig


if __name__ == '__main__':
    #tasks = [task2]
    tasks = [task3]

    pdf = PdfPages('figures.pdf')
    for task in tasks:
        f = task()
        pdf.savefig(f)

    pdf.close()
