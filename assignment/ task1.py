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


import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier

# 可选：UMAP（若未安装则自动跳过）
try:
    import umap

    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

DATA_DIR = "/Users/zhengxuzhang/Code/zhengxz/d2l/asignment/dataset"
OUT_DIR = "/Users/zhengxuzhang/Code/zhengxz/d2l/asignment"

TRAIN_IN = os.path.join(DATA_DIR, "train_in.csv")
TRAIN_OUT = os.path.join(DATA_DIR, "train_out.csv")
TEST_IN = os.path.join(DATA_DIR, "test_in.csv")
TEST_OUT = os.path.join(DATA_DIR, "test_out.csv")


def load_data():
    X_train = pd.read_csv(TRAIN_IN, header=None).values
    y_train = pd.read_csv(TRAIN_OUT, header=None).values.ravel().astype(int)
    X_test = pd.read_csv(TEST_IN, header=None).values
    y_test = pd.read_csv(TEST_OUT, header=None).values.ravel().astype(int)
    assert X_train.shape[1] == 256, f"期望特征维度256，实际{X_train.shape[1]}"
    assert X_test.shape[1] == 256, f"期望特征维度256，实际{X_test.shape[1]}"
    return X_train, y_train, X_test, y_test


def compute_class_centers(X, y, num_classes=10):
    centers = []
    for d in range(num_classes):
        mask = y == d
        centers.append(X[mask].mean(axis=0))
    centers = np.vstack(centers)
    return centers  # shape (10, 256)


def pairwise_center_distances(centers):
    # 无需scipy，直接用向量化计算欧氏距离矩阵
    # D_ij = ||c_i - c_j|| = sqrt(||c_i||^2 + ||c_j||^2 - 2 c_i·c_j)
    C2 = np.sum(centers**2, axis=1, keepdims=True)  # (10,1)
    G = centers @ centers.T  # (10,10)
    D2 = C2 + C2.T - 2 * G
    D2 = np.maximum(D2, 0.0)
    return np.sqrt(D2)


def visualize_2d(method_name, X, y, out_path, random_state=42):
    if method_name == "PCA":
        Z = PCA(n_components=2, random_state=random_state).fit_transform(X)
    elif method_name == "TSNE":
        Z = TSNE(
            n_components=2, random_state=random_state, init="pca", learning_rate="auto"
        ).fit_transform(X)
    elif method_name == "UMAP":
        if not HAS_UMAP:
            print("UMAP 未安装，跳过 UMAP 可视化。可执行: pip install umap-learn")
            return
        reducer = umap.UMAP(n_components=2, random_state=random_state)
        Z = reducer.fit_transform(X)
    else:
        raise ValueError("Unknown method")

    plt.figure(figsize=(7, 6))
    palette = sns.color_palette("tab10", 10)
    for d in range(10):
        plt.scatter(
            Z[y == d, 0], Z[y == d, 1], s=8, c=[palette[d]], label=str(d), alpha=0.7
        )
    plt.legend(markerscale=2, fontsize=8, ncol=5, frameon=False)
    plt.title(f"{method_name} 2D 可视化（训练集）")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def nearest_mean_predict(X, centers):
    # 距离平方：||x - c||^2 = ||x||^2 + ||c||^2 - 2 x·c
    X2 = np.sum(X**2, axis=1, keepdims=True)  # (n,1)
    C2 = np.sum(centers**2, axis=1, keepdims=True).T  # (1,10)
    XC = X @ centers.T  # (n,10)
    d2 = X2 + C2 - 2 * XC
    return np.argmin(d2, axis=1)


def plot_confusion(cm, title, out_path, normalize=True):
    if normalize:
        cm_display = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)
    else:
        cm_display = cm
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_display,
        annot=False,
        cmap="Blues",
        square=True,
        cbar=True,
        xticklabels=list(range(10)),
        yticklabels=list(range(10)),
    )
    plt.xlabel("预测标签")
    plt.ylabel("真实标签")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_matrix_csv(matrix, out_path, row_labels=None, col_labels=None):
    df = pd.DataFrame(matrix, index=row_labels, columns=col_labels)
    df.to_csv(out_path, index=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("加载数据...")
    X_train, y_train, X_test, y_test = load_data()
    print(f"训练集: {X_train.shape}, 测试集: {X_test.shape}")

    # 1) 类中心与中心间距离矩阵
    print("计算类别中心与中心间距离矩阵...")
    centers = compute_class_centers(X_train, y_train, num_classes=10)
    D = pairwise_center_distances(centers)
    save_matrix_csv(
        D,
        os.path.join(OUT_DIR, "centers_distance_matrix.csv"),
        row_labels=[f"c{i}" for i in range(10)],
        col_labels=[f"c{j}" for j in range(10)],
    )
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        D,
        annot=False,
        cmap="mako",
        square=True,
        xticklabels=list(range(10)),
        yticklabels=list(range(10)),
    )
    plt.title("类别中心欧氏距离矩阵")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "centers_distance_matrix_heatmap.png"), dpi=200)
    plt.close()

    pairs = [((i, j), D[i, j]) for i in range(10) for j in range(i + 1, 10)]
    pairs_sorted = sorted(pairs, key=lambda x: x[1])
    print("中心最相近的前5对类别及距离：")
    for (i, j), dist in pairs_sorted[:5]:
        print(f"  ({i}, {j}): {dist:.4f}")
    likely_hard_pairs = [(i, j) for (i, j), _ in pairs_sorted[:5]]

    # 2) 降维可视化（训练集）
    print("生成 PCA/TSNE/UMAP 可视化（训练集）...")
    visualize_2d("PCA", X_train, y_train, os.path.join(OUT_DIR, "pca_2d.png"))
    visualize_2d("TSNE", X_train, y_train, os.path.join(OUT_DIR, "tsne_2d.png"))
    if HAS_UMAP:
        visualize_2d("UMAP", X_train, y_train, os.path.join(OUT_DIR, "umap_2d.png"))

    # 3) 最近均值分类器
    print("评估 最近均值分类器 (Nearest Mean Classifier)...")
    y_pred_train_nmc = nearest_mean_predict(X_train, centers)
    y_pred_test_nmc = nearest_mean_predict(X_test, centers)
    acc_train_nmc = accuracy_score(y_train, y_pred_train_nmc)
    acc_test_nmc = accuracy_score(y_test, y_pred_test_nmc)
    print(f"NMC 训练集准确率: {acc_train_nmc:.4f}")
    print(f"NMC 测试集准确率: {acc_test_nmc:.4f}")

    cm_train_nmc = confusion_matrix(y_train, y_pred_train_nmc, labels=list(range(10)))
    cm_test_nmc = confusion_matrix(y_test, y_pred_test_nmc, labels=list(range(10)))
    plot_confusion(
        cm_train_nmc,
        "NMC 训练集混淆矩阵(归一化)",
        os.path.join(OUT_DIR, "cm_nmc_train.png"),
    )
    plot_confusion(
        cm_test_nmc,
        "NMC 测试集混淆矩阵(归一化)",
        os.path.join(OUT_DIR, "cm_nmc_test.png"),
    )
    save_matrix_csv(
        cm_train_nmc,
        os.path.join(OUT_DIR, "cm_nmc_train_counts.csv"),
        row_labels=[f"true_{i}" for i in range(10)],
        col_labels=[f"pred_{j}" for j in range(10)],
    )
    save_matrix_csv(
        cm_test_nmc,
        os.path.join(OUT_DIR, "cm_nmc_test_counts.csv"),
        row_labels=[f"true_{i}" for i in range(10)],
        col_labels=[f"pred_{j}" for j in range(10)],
    )

    # 4) KNN 分类器
    print("训练与评估 KNN 分类器 (k=5, Euclidean)...")
    knn = KNeighborsClassifier(n_neighbors=5, metric="euclidean", weights="uniform")
    knn.fit(X_train, y_train)
    y_pred_train_knn = knn.predict(X_train)
    y_pred_test_knn = knn.predict(X_test)
    acc_train_knn = accuracy_score(y_train, y_pred_train_knn)
    acc_test_knn = accuracy_score(y_test, y_pred_test_knn)
    print(f"KNN 训练集准确率: {acc_train_knn:.4f}")
    print(f"KNN 测试集准确率: {acc_test_knn:.4f}")

    cm_train_knn = confusion_matrix(y_train, y_pred_train_knn, labels=list(range(10)))
    cm_test_knn = confusion_matrix(y_test, y_pred_test_knn, labels=list(range(10)))
    plot_confusion(
        cm_train_knn,
        "KNN 训练集混淆矩阵(归一化)",
        os.path.join(OUT_DIR, "cm_knn_train.png"),
    )
    plot_confusion(
        cm_test_knn,
        "KNN 测试集混淆矩阵(归一化)",
        os.path.join(OUT_DIR, "cm_knn_test.png"),
    )
    save_matrix_csv(
        cm_train_knn,
        os.path.join(OUT_DIR, "cm_knn_train_counts.csv"),
        row_labels=[f"true_{i}" for i in range(10)],
        col_labels=[f"pred_{j}" for j in range(10)],
    )
    save_matrix_csv(
        cm_test_knn,
        os.path.join(OUT_DIR, "cm_knn_test_counts.csv"),
        row_labels=[f"true_{i}" for i in range(10)],
        col_labels=[f"pred_{j}" for j in range(10)],
    )

    print("\n—— 结论与观察 ——")
    print(f"最近均值分类器：训练 {acc_train_nmc:.4f}，测试 {acc_test_nmc:.4f}")
    print(f"KNN(k=5)：训练 {acc_train_knn:.4f}，测试 {acc_test_knn:.4f}")
    print("可能较难区分的数字对（基于中心距离最小）：", likely_hard_pairs)
    print("详见目录下生成的可视化与CSV/PNG文件。")


if __name__ == "__main__":
    main()
