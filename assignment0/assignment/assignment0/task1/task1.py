# Task 1: Data dimensionality, distance-based classifiers
# The purpose of this task is to develop some intuitions about clouds of points in high-dimensional spaces. In
# particular, you are supposed to use dimensionality reduction techniques to visualize your data, develop a
# very simple algorithm for classifying hand-written digits, and compare it to another distance-based classifier.
# 1. For each digit d,(d = 0, 1, ..., 9), let us consider a cloud of points in 256-dimensional space, Cd, which
# consists of all training images (vectors) that represent d. For each cloud Cd we can calculate its center,
# cd, which is just a 256-dimensional vector of means over all coordinates of vectors that belong to Cd.
# Once we have these centers, we can easily classify new images: by calculating the distance from the
# vector that represents this image to each of the 10 centers, the closest center defines the label of the
# image. Next, calculate the distances between the centers of the 10 clouds, distij = dist(ci
# , cj ), for
# i, j = 0, 1, ...9. Given all these distances, try to say something about the expected accuracy of your
# classifier. What pairs of digits seem to be most difficult to separate?
# 2. Experiment with three dimensionality reduction algorithms: PCA, U-MAP, T-SNE and apply them
# to the MNIST data to generate a visualization of the different classes, preferably in 2D. You are free
# to use any library to do this (preferably scikit-learn and umap-learn packages from PyPI.).
# Does the visualization agree with your intuitions and the between-class distance matrix distij ?
# 3. Use the mean pixel values of each digit category obtained in part 1 to implement a Nearest mean
# classifier. Apply your classifier to all points from the training set and calculate the percentage of
# correctly classified digits. Do the same with the test set, using the centers that were calculated from
# the training set.
# 4. A less naive distance-based approach is the KNN (K-Nearest-Neighbor) classifier (you can either implement it yourself or use the one from sklearn package). Repeat the same procedure as in part 3
# by using this method. Then, for both classifiers, generate a confusion matrix which should provide a
# deeper insight into classes that are difficult to separate. A confusion matrix is here a 10-by-10 matrix
# (cij ), where cij contains the percentage (or count) of digits i that are classified as j. Which digits are
# most difficult to classify correctly? Again, for calculating and visualising confusion matrices you may
# use the sklearn package. Describe your findings, and compare the performance of your classifiers on
# the train and test sets

# 任务1：数据的高维特征与基于距离的分类器
#
# 这个任务的主要目的是帮助你理解高维空间中点云（数据分布）的特性，并通过可视化和简单的分类算法，探索手写数字识别问题。具体来说，任务分为以下几个部分：
#
# 1.首先，对于每个数字d（d=0,1,...,9），我们把所有训练集中属于该数字的图片（每张图片是256维向量）看作一个点云Cd。对于每个点云Cd，可以计算其中心cd（即所有属于该类的向量在每个维度上的均值，得到一个256维的中心向量）。有了这10个中心后，我们可以用“最近中心”法来分类新图片：计算新图片与每个中心的距离，距离最近的中心对应的数字就是分类结果。接着，计算这10个中心两两之间的距离dist_ij（i,j=0,...,9）。如果某两个中心之间的距离很小，说明这两类数字的分布非常接近，分类器容易把它们混淆，导致分类错误；如果所有中心之间的距离都很大，则分类器的预期准确率会很高。因此，可以通过观察中心距离矩阵dist_ij，初步判断分类器的难点和整体准确率：距离越大，分类越容易，准确率越高；距离越小，分类越困难，容易出错。实际准确率还需结合数据分布和样本数量，但中心距离为我们提供了重要的可分性直观依据。
#
# 2. 使用三种降维算法（PCA、U-MAP、T-SNE）对MNIST数据进行降维（最好降到2维），并可视化不同数字类别的分布。你可以自由选择库（推荐scikit-learn和umap-learn）。观察可视化结果是否与你通过中心距离矩阵dist_ij得到的直觉一致。
#
# 3. 利用第1步得到的每类数字的均值向量，实现一个“最近均值分类器”（Nearest Mean Classifier）。用它对训练集和测试集进行分类，并计算分类准确率。
#
# 4. 实现一个更复杂的基于距离的分类器——K近邻（KNN）分类器（可以自己实现，也可以用sklearn库）。重复第3步的实验，并为两种分类器都生成混淆矩阵（confusion matrix，10x10矩阵，c_ij表示真实为i但被分为j的样本比例或数量）。分析哪些数字最难被正确分类，并比较两种分类器在训练集和测试集上的表现。
#
# 总结：本任务通过高维数据的中心、距离、降维可视化和两种基于距离的分类器，帮助你理解高维空间中数据分布的特性，以及不同数字类别之间的可分性。

from json import load
import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "dataset"))
OUT_DIR = os.path.join(BASE_DIR, "out")

TRAIN_IN_PATH = os.path.join(DATA_DIR, "train_in.csv")
TRAIN_OUT_PATH = os.path.join(DATA_DIR, "train_out.csv")
TEST_IN_PATH = os.path.join(DATA_DIR, "test_in.csv")
TEST_OUT_PATH = os.path.join(DATA_DIR, "test_out.csv")


def load_data():
    X_train = pd.read_csv(TRAIN_IN_PATH, header=None).values
    y_train = pd.read_csv(TRAIN_OUT_PATH, header=None).values.ravel()
    X_test = pd.read_csv(TEST_IN_PATH, header=None).values
    y_test = pd.read_csv(TEST_OUT_PATH, header=None).values.ravel().astype(int)
    assert (
        X_train.shape[1] == 256
    ), f"Expected feature dimension 256, got {X_train.shape[1]}"
    assert (
        y_train.shape[0] == X_train.shape[0]
    ), f"y_train sample size should match X_train, got y_train: {y_train.shape[0]}, X_train: {X_train.shape[0]}"
    return X_train, y_train, X_test, y_test


def compute_digit_centers(X, y, digit_classes):
    centers = {}
    for d in range(digit_classes):
        d_indexs = []
        for i in range(len(y)):
            if y[i] == d:
                d_indexs.append(i)
        # Iterate over the digit indexs to find corresponding X numpy indices
        X_digit = X[d_indexs]
        # Compute the mean vector of all samples in this class
        centers[d] = X_digit.mean(axis=0)
    return centers


def pair_center_distances(centers):
    # centers 是 shape (n, d) 的二维数组，n为类别数，d为特征维数
    n = centers.shape[0]
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            # 计算第i和第j个center之间的欧氏距离
            # 用 sqrt 计算欧氏距离
            D[i, j] = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
    return D


def visualize_2d(method_name, X, y, random_state=42):
    if method_name == "PCA":
        Z = PCA(n_components=2, random_state=random_state).fit_transform(X)
    elif method_name == "TSNE":
        Z = TSNE(
            n_components=2, random_state=random_state, init="pca", learning_rate="auto"
        ).fit_transform(X)
    elif method_name == "UMAP":
        reducer = umap.UMAP(n_components=2, random_state=random_state)
        Z = reducer.fit_transform(X)
    else:
        raise ValueError("Unknown method")

    fig, axes = plt.subplots(1, 1, figsize=(7, 6))
    fig.tight_layout(pad=4.0)

    # 定义颜色调色板
    colors = plt.cm.tab10(np.linspace(0, 1, 10))

    for d in range(10):
        axes.scatter(
            Z[y == d, 0], Z[y == d, 1], s=8, c=[colors[d]], label=str(d), alpha=0.7
        )
    axes.legend(markerscale=2, fontsize=8, ncol=5, frameon=False)
    axes.set_title(f"{method_name} 2D Visualization (Training Set)")
    fig.tight_layout()

    plt.show()
    plt.close(fig)


def nearest_mean_predict(X, centers):
    print(f"X shape: {X.shape}, centers shape: {centers.shape}")

    assert (
        X.shape[1] == centers.shape[1]
    ), f"Number of samples in X ({X.shape[1]}) does not match number of centers ({centers.shape[1]}), please check your input."

    preds = []
    # 遍历X中的每一个样本x（每个x是256维向量），再遍历centers中的每个center（也是256维向量），计算两者的欧氏距离，返回距离最近的center索引
    for i in range(X.shape[0]):
        x = X[i]  # Extract the i-th sample, a 256-dimensional vector
        min_dist = float("inf")
        min_idx = -1
        for j in range(centers.shape[0]):
            center = centers[j]  # Extract the j-th center, a 256-dimensional vector
            dist = np.linalg.norm(x - center)  # Compute Euclidean distance
            if dist < min_dist:
                min_dist = dist
                min_idx = j
        preds.append(min_idx)

    return np.array(preds)


def sort_pair(pair_distances):
    pairs_sorted = []
    for pair in pair_distances:
        pairs_sorted.append(pair)
    # 按照距离（即元组的第二个元素）排序
    for i in range(len(pairs_sorted)):
        for j in range(i + 1, len(pairs_sorted)):
            if pairs_sorted[i][1] > pairs_sorted[j][1]:
                pairs_sorted[i], pairs_sorted[j] = pairs_sorted[j], pairs_sorted[i]
    return pairs_sorted


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("loading data...", flush=True)
    X_train, y_train, X_test, y_test = load_data()
    print(
        f"x train data shape: {X_train.shape}, test data shape: {X_test.shape}",
        flush=True,
    )
    print(
        f"y train data shape: {y_train.shape}, test data shape: {y_test.shape}",
        flush=True,
    )

    # 1) digit centers and pairwise center distance matrix
    print("Calculating digit centers and pairwise center distance matrix...")

    digit_num = len(np.unique(y_train))
    print(f"Detected {digit_num} digit classes.")

    centers = compute_digit_centers(X_train, y_train, digit_classes=digit_num)
    print("Each digit center shape:", centers[0].shape)

    centers = np.stack([centers[d] for d in range(digit_num)], axis=0)  # (10,256)
    assert centers.shape == (
        10,
        256,
    ), f"Expected centers shape (10,256), got {centers.shape}"

    D = pair_center_distances(centers)
    assert D.shape == (10, 10), f"Expected distance matrix shape (10,10), got {D.shape}"

    pair_distances = [
        ((i, j), D[i, j]) for i in range(digit_num) for j in range(i + 1, digit_num)
    ]
    pairs_sorted = sort_pair(pair_distances)

    print("Top 3 closest pairs of class centers and their distances:", flush=True)
    for (i, j), dist in pairs_sorted[:3]:
        print(f"  ({i}, {j}): {dist:.4f}")
    likely_hard_pairs = [(i, j) for (i, j), _ in pairs_sorted[:3]]

    # 2) Dimensionality reduction visualization (training set)
    print("Dimensionality reduction visualization (training set)...")
    visualize_2d("PCA", X_train, y_train)
    visualize_2d("TSNE", X_train, y_train)
    visualize_2d("UMAP", X_train, y_train)

    # 3) Implement Nearest Mean Classifier
    print("Implementing Nearest Mean Classifier...")
    y_pred_train_nmc = nearest_mean_predict(X_train, centers)
    y_pred_test_nmc = nearest_mean_predict(X_test, centers)

    acc_train_nmc = np.mean(y_train == y_pred_train_nmc)
    acc_test_nmc = np.mean(y_test == y_pred_test_nmc)
    print(f"Nearest Mean Classifier training accuracy: {acc_train_nmc:.4f}", flush=True)
    print(f"Nearest Mean Classifier test accuracy: {acc_test_nmc:.4f}", flush=True)

    # 4) KNN classifier
    print("Training and evaluating KNN classifier (k=5, Euclidean)...", flush=True)
    knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean", weights="uniform")
    knn.fit(X_train, y_train)
    y_pred_train_knn = knn.predict(X_train)
    y_pred_test_knn = knn.predict(X_test)
    acc_train_knn = np.mean(y_train == y_pred_train_knn)
    acc_test_knn = np.mean(y_test == y_pred_test_knn)
    print(f"KNN training accuracy: {acc_train_knn:.4f}", flush=True)
    print(f"KNN test accuracy: {acc_test_knn:.4f}", flush=True)

    # 5) Generate confusion matrix
    cm_train_nmc = confusion_matrix(y_train, y_pred_train_nmc, labels=list(range(10)))
    cm_test_nmc = confusion_matrix(y_test, y_pred_test_nmc, labels=list(range(10)))

    cm_train_knn = confusion_matrix(y_train, y_pred_train_knn, labels=list(range(10)))
    cm_test_knn = confusion_matrix(y_test, y_pred_test_knn, labels=list(range(10)))

    print("\n—— Conclusions and Observations ——")
    print(
        f"Nearest Mean Classifier: Train {acc_train_nmc:.4f}, Test {acc_test_nmc:.4f}"
    )
    print(f"KNN (k=5): Train {acc_train_knn:.4f}, Test {acc_test_knn:.4f}")
    print(
        "Digit pairs that are likely hard to distinguish (based on smallest center distances):",
        likely_hard_pairs,
    )
    print("\nNMC Training Set Confusion Matrix:")
    print(cm_train_nmc)
    print("\nNMC Test Set Confusion Matrix:")
    print(cm_test_nmc)
    print("\nKNN Training Set Confusion Matrix:")
    print(cm_train_knn)
    print("\nKNN Test Set Confusion Matrix:")
    print(cm_test_knn)


if __name__ == "__main__":
    main()
