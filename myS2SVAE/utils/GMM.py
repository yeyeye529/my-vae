'''Author: Ron Weiss <ronweiss@gmail.com>, Gael Varoquaux
Modified by Thierry Guillemot <thierry.guillemot.work@gmail.com>
License: BSD 3 clause'''

import matplotlib as mpl
import matplotlib.pyplot as plt

import numpy as np

from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict
import math
import logging

# print(__doc__)


class gmm(object):
    def __init__(self, K, cov_type = 'diag', max_iter = 100, random_state=0,init_params='kmeans',means_init=None): # 'kmeans'
        super(gmm, self).__init__()
        self.K = K
        self.cov_type = cov_type  # acturally we only support 'diag' type :(
        self.gmm_model = GaussianMixture(n_components=K,
                                         covariance_type=cov_type,
                                         max_iter=max_iter,
                                         random_state=random_state,
                                         init_params=init_params,
                                         means_init=means_init,
                                         #reg_covar=0.01
                                         )

    def train(self, X_train, output_class_path = None, sent_list = None, vocab = None):
        self.gmm_model.fit(X_train, )
        # y_train_pred = self.gmm_model.predict(X_train)
        # print(y_train_pred)
        # probs = self.gmm_model.predict_proba(X_train)
        # print(probs)
        # from collections import defaultdict
        # cids = defaultdict(list)
        # for i, x in enumerate(y_train_pred):
        #     cids[x].append(i)
        #
        # for c, ids in cids.items():
        #     print(c, len(ids))

        # print(self.gmm_model.converged_)
        # print(self.gmm_model.n_iter_)
        # print(self.gmm_model.means_.shape)
        # print(self.gmm_model.covariances_.shape)
        # print(self.gmm_model.means_)
        # print(np.sqrt(self.gmm_model.covariances_))

        return self.gmm_model.means_, np.sqrt(self.gmm_model.covariances_)

        clusters = defaultdict(list)
        for id, cid in enumerate(y_train_pred):
            clusters[cid].append(id)


        if output_class_path == None:
            return
        print("saving cluster.txt")
        fout = open(output_class_path, "w")
        for cid, sids in clusters.items():
            fout.write("cluster " + str(cid) + "\n")
            for s in sids:
                if vocab == None:
                    fout.write(' '.join(sent_list[s]))
                else:
                    fout.write(' '.join([vocab[w] for w in sent_list[s] if vocab[w] != '<pad>']))
                fout.write("\n")
        fout.close()
        exit()
        return self.gmm_model.means_, np.sqrt(self.gmm_model.covariances_)









# colors = ['navy', 'turquoise', 'darkorange']
#
#
# def make_ellipses(gmm, ax):
#     for n, color in enumerate(colors):
#         if gmm.covariance_type == 'full':
#             covariances = gmm.covariances_[n][:2, :2]
#         elif gmm.covariance_type == 'tied':
#             covariances = gmm.covariances_[:2, :2]
#         elif gmm.covariance_type == 'diag':
#             covariances = np.diag(gmm.covariances_[n][:2])
#         elif gmm.covariance_type == 'spherical':
#             covariances = np.eye(gmm.means_.shape[1]) * gmm.covariances_[n]
#         v, w = np.linalg.eigh(covariances)
#         u = w[0] / np.linalg.norm(w[0])
#         angle = np.arctan2(u[1], u[0])
#         angle = 180 * angle / np.pi  # convert to degrees
#         v = 2. * np.sqrt(2.) * np.sqrt(v)
#         ell = mpl.patches.Ellipse(gmm.means_[n, :2], v[0], v[1],
#                                   180 + angle, color=color)
#         ell.set_clip_box(ax.bbox)
#         ell.set_alpha(0.5)
#         ax.add_artist(ell)
#
# iris = datasets.load_iris()
#
# # Break up the dataset into non-overlapping training (75%) and testing
# # (25%) sets.
# skf = StratifiedKFold(n_splits=4)
# # Only take the first fold.
# train_index, test_index = next(iter(skf.split(iris.data, iris.target)))
#
#
# X_train = iris.data[train_index]
# y_train = iris.target[train_index]
# X_test = iris.data[test_index]
# y_test = iris.target[test_index]
#
#
# n_classes = len(np.unique(y_train))
# print('n_classes', n_classes)
# print(X_test)
#
# # Try GMMs using different types of covariances.
# estimators = dict((cov_type, GaussianMixture(n_components=n_classes,
#                    covariance_type=cov_type, max_iter=20, random_state=0))
#                   for cov_type in ['spherical', 'diag', 'tied', 'full'])
#
# n_estimators = len(estimators)
#
# print('n_estimators', n_estimators)
#
# plt.figure(figsize=(3 * n_estimators // 2, 6))
# plt.subplots_adjust(bottom=.01, top=0.95, hspace=.15, wspace=.05,
#                     left=.01, right=.99)
#
#
# for index, (name, estimator) in enumerate(estimators.items()):
#     print(index, name)
#     # Since we have class labels for the training data, we can
#     # initialize the GMM parameters in a supervised manner.
#     estimator.means_init = np.array([X_train[y_train == i].mean(axis=0)
#                                     for i in range(n_classes)])
#
#     # Train the other parameters using the EM algorithm.
#     estimator.fit(X_train)
#
#     h = plt.subplot(2, n_estimators // 2, index + 1)
#     make_ellipses(estimator, h)
#
#     for n, color in enumerate(colors):
#         data = iris.data[iris.target == n]
#         plt.scatter(data[:, 0], data[:, 1], s=0.8, color=color,
#                     label=iris.target_names[n])
#     # Plot the test data with crosses
#     for n, color in enumerate(colors):
#         data = X_test[y_test == n]
#         plt.scatter(data[:, 0], data[:, 1], marker='x', color=color)
#
#     y_train_pred = estimator.predict(X_train)
#     train_accuracy = np.mean(y_train_pred.ravel() == y_train.ravel()) * 100
#     plt.text(0.05, 0.9, 'Train accuracy: %.1f' % train_accuracy,
#              transform=h.transAxes)
#
#     y_test_pred = estimator.predict(X_test)
#     test_accuracy = np.mean(y_test_pred.ravel() == y_test.ravel()) * 100
#     plt.text(0.05, 0.8, 'Test accuracy: %.1f' % test_accuracy,
#              transform=h.transAxes)
#
#     plt.xticks(())
#     plt.yticks(())
#     plt.title(name)
#     print('train_accuracy', train_accuracy)
#     print('test_accuracy', test_accuracy)
#
# plt.legend(scatterpoints=1, loc='lower right', prop=dict(size=12))
#
#
# # plt.show()
# plt.savefig("gmm_tmp.png")