# -*- coding: utf-8 -*-
"""
任务2:实现一个多类感知机(multi-class perceptron)算法 —— 中文翻译与要点说明 + 参考实现模板

一、任务要求（原文要点的中文翻译与凝练）
- 从零实现一个多类感知机训练算法。多类感知机可被看作是 10 个相互独立的二分类线性分类器(One-vs-Rest),每个分类器学习判断某个数字是否属于它对应的那一类。
- 参考《Understanding Deep Learning》教材第 6.1 章，特别是式 (6.3) 与 (6.7)。
  - (6.3) 是权重更新的梯度下降形式：φ ← φ - a· ∂L/∂φ
  - (6.7) 给出了平方损失在一维线性模型下对参数的梯度：
      ∂ℓ_i/∂φ = [ 2(φ₀ + φ₁ x_i - y_i),
                   2 x_i (φ₀ + φ₁ x_i - y_i) ]ᵀ
    其中 φ₀ 是偏置(bias),φ₁ 是普通权重。这个二维写法是为了直观说明“偏置项的梯度=2残差;权重项的梯度=2 x_i 残差”。

- 训练建议：
  1) 在训练集上训练，每个 epoch 之后计算并记录平均损失/准确率，并画图。
  2) 在测试集上评估（损失/准确率），并将曲线与训练集对比绘制。
  3) 由于随机初始化导致非确定性，建议重复实验数次，感受精度的稳定性。
  4) 尝试不同的权重初始化与学习率，观察其对训练效果的影响。

- 效率建议：
  - 尽量使用矩阵运算，减少循环。比如在特征矩阵前拼接一列全 1 来承载偏置项。
  - 将网络权重存为矩阵 W（例如若特征维为 256，拼接偏置后为 257 维，10 类，则 W 形状为 257×10）。
  - 模型对所有样本的输出可写为 T·W，其中 T 是带偏置的输入矩阵（N×257，N 为样本数）。
  - 预测类别可用 numpy.argmax() 在类别维上取最大。
  - 线性可分的训练集上，多类感知机应能在数秒内收敛。

- 比较问题：
  - 将本任务的单层多类感知机与任务1中的距离类方法(如最近中心/最近邻等）在准确率上的表现进行比较。

二、进一步解释（帮助你把式子迁移到多类场景）
- 从“二分类平方损失的梯度”可抽象出：对任一线性模型 s = φ₀ + φᵀx,其单样本残差 r = s − y,
  则对“带偏置的参数向量” φ̃ = [φ₀; φ] 的梯度为 ∂ℓ/∂φ̃ = 2·r·[1; x]。
- 多类情形常用 One-vs-Rest(OvR)或等价的“误分类样本将权重从预测类拉向真实类”的感知机更新：
  - 记带偏置输入为 x̃ = [1; x]（维度 D+1），权重矩阵 W ∈ R^{(D+1)×C}。
  - 分类打分 s = x̃ᵀ W（得到长度为 C 的分数向量），预测 ŷ = argmax_c s_c。
  - 若 ŷ ≠ y（真实类），则执行：
      W[:, y] ← W[:, y] + α · x̃
      W[:, ŷ] ← W[:, ŷ] − α · x̃
    这就是经典多类感知机的“拉真类/推错类”更新。
- 若你采用平方损失（对 one-hot 目标），也可以整批（batch）做矩阵化的梯度下降：
  - 令 Y 为 one-hot（N×C），T 为带偏置的输入（N×(D+1)），S = T·W（N×C），残差 R = S − Y。
  - 平方损失 L = (1/N)·∥R∥²（或 1/(2N) 视实现而定），则
      ∂L/∂W = (2/N) · Tᵀ · R
    再用式 (6.3) 更新 W。

三、实现步骤建议
1) 预处理：将输入 X（N×D）前拼接一列 1，得到 T（N×(D+1)）。
2) 初始化：W ~ N(0, σ²) 或均匀分布；尝试多种种子看稳定性。
3) 训练循环（epoch）：
   - 计算打分 S = T·W，预测 ŷ = argmax(S, axis=1)。
   - 计算训练准确率；若用平方损失，计算 L = mean((S − Y)²)。
   - 按选择的更新规则更新 W（感知机式逐样本/小批或平方损失的整批梯度）。
4) 每个 epoch 后评估测试集并记录历史；最后绘图对比训练/测试曲线。
5) 和任务1方法的准确率做简要对比讨论。

下面给出一个可直接使用的多类感知机（OvR）参考实现（numpy），既支持“经典感知机误分类更新”，也支持“平方损失的批量梯度下降”两种训练方式，便于对照式 (6.3)/(6.7) 理解。

使用方式提示：
- 你只需准备好 X_train, y_train, X_test, y_test（y 为 0..C-1 的整数标签），调用下面的 fit() 与 evaluate()。
- 若你想画学习曲线，记录 history 后自行用 matplotlib 绘图即可。
"""


import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd


class MultiClassPerceptron:
    """
    多类感知机，支持两种训练方式：
    1. 经典感知机误分类更新（逐样本/小批量）
    2. 平方损失的批量梯度下降
    """

    def __init__(
        self,
        n_classes,
        learning_rate=0.01,
        epochs=20,
        batch_size=None,
        init_scale=0.01,
        seed=None,
        update_rule="perceptron",  # "perceptron" 或 "squared_loss"
    ):
        self.n_classes = n_classes
        self.lr = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.init_scale = init_scale
        self.seed = seed
        self.update_rule = update_rule
        self.W = None  # 权重矩阵

    def _init_weights(self, d):
        if self.seed is not None:
            np.random.seed(self.seed)
        # W: (D+1, C)
        self.W = np.random.randn(d + 1, self.n_classes) * self.init_scale

    def _add_bias(self, X):
        # X: (N, D) -> (N, D+1)
        N = X.shape[0]
        return np.hstack([np.ones((N, 1)), X])

    def fit(self, X, y, X_val=None, y_val=None):
        """
        训练模型，返回每个epoch的损失和准确率历史
        """
        X = self._add_bias(X)
        N, D_plus1 = X.shape
        if self.W is None:
            self._init_weights(D_plus1 - 1)
        history = {"train_loss": [], "train_acc": []}
        if X_val is not None and y_val is not None:
            X_val = self._add_bias(X_val)
            history["val_loss"] = []
            history["val_acc"] = []

        for ep in range(self.epochs):
            if self.update_rule == "perceptron":
                # 经典感知机逐样本更新
                idxs = np.arange(N)
                np.random.shuffle(idxs)
                for idx in idxs:
                    x_i = X[idx]  # (D+1,)
                    y_i = y[idx]
                    s = np.dot(x_i, self.W)  # (C,)
                    y_pred = np.argmax(s)
                    if y_pred != y_i:
                        # 推错类，拉真类
                        self.W[:, y_i] += self.lr * x_i
                        self.W[:, y_pred] -= self.lr * x_i
                # 计算损失和准确率
                S = np.dot(X, self.W)
                y_pred = np.argmax(S, axis=1)
                acc = np.mean(y_pred == y)
                # 感知机本身没有损失，这里可用误分类数/率
                loss = np.mean(y_pred != y)
            elif self.update_rule == "squared_loss":
                # 平方损失的批量梯度下降
                # 构造one-hot标签
                Y = np.zeros((N, self.n_classes))
                Y[np.arange(N), y] = 1.0
                S = np.dot(X, self.W)
                R = S - Y  # (N, C)
                loss = np.mean((R) ** 2)
                grad = (2.0 / N) * np.dot(X.T, R)  # (D+1, C)
                self.W -= self.lr * grad
                y_pred = np.argmax(S, axis=1)
                acc = np.mean(y_pred == y)
            else:
                raise ValueError("未知的更新规则: %s" % self.update_rule)
            history["train_loss"].append(loss)
            history["train_acc"].append(acc)

            # 验证集
            if X_val is not None and y_val is not None:
                S_val = np.dot(X_val, self.W)
                y_pred_val = np.argmax(S_val, axis=1)
                acc_val = np.mean(y_pred_val == y_val)
                if self.update_rule == "squared_loss":
                    Y_val = np.zeros((X_val.shape[0], self.n_classes))
                    Y_val[np.arange(X_val.shape[0]), y_val] = 1.0
                    loss_val = np.mean((S_val - Y_val) ** 2)
                else:
                    loss_val = np.mean(y_pred_val != y_val)
                history["val_loss"].append(loss_val)
                history["val_acc"].append(acc_val)
            # 可选：打印每个epoch的结果
            # print(f"Epoch {ep+1}: train_acc={acc:.4f}, train_loss={loss:.4f}")

        return history

    def predict(self, X):
        X = self._add_bias(X)
        S = np.dot(X, self.W)
        return np.argmax(S, axis=1)

    def evaluate(self, X, y):
        y_pred = self.predict(X)
        acc = np.mean(y_pred == y)
        return acc


# ========== 用法示例 ==========

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "..", "dataset"))
OUT_DIR = os.path.join(BASE_DIR, "out")

TRAIN_IN = os.path.join(DATA_DIR, "train_in.csv")
TRAIN_OUT = os.path.join(DATA_DIR, "train_out.csv")
TEST_IN = os.path.join(DATA_DIR, "test_in.csv")
TEST_OUT = os.path.join(DATA_DIR, "test_out.csv")


def load_data():
    X_train = pd.read_csv(TRAIN_IN, header=None).values
    y_train = pd.read_csv(TRAIN_OUT, header=None).values.ravel().astype(int)
    X_test = pd.read_csv(TEST_IN, header=None).values
    y_test = pd.read_csv(TEST_OUT, header=None).values.ravel().astype(int)
    assert (
        X_train.shape[1] == 256
    ), f"Expected feature dimension 256, but got {X_train.shape[1]}"
    assert (
        X_test.shape[1] == 256
    ), f"Expected feature dimension 256, but got {X_test.shape[1]}"
    return X_train, y_train, X_test, y_test


def run_demo():
    # 假设你已经有 X_train, y_train, X_test, y_test
    # 这里只做演示，实际请用你的数据
    # X_train, y_train, X_test, y_test = ...

    # 这里用随机数据做演示
    # np.random.seed(42)
    N, D, C = 200, 256, 10
    # X_train = np.random.randn(N, D)
    # y_train = np.random.randint(0, C, size=N)
    # X_test = np.random.randn(50, D)
    # y_test = np.random.randint(0, C, size=50)
    X_train, y_train, X_test, y_test = load_data()

    # 1. 训练
    clf = MultiClassPerceptron(
        n_classes=C, learning_rate=0.001, epochs=10, update_rule="perceptron"
    )

    history = clf.fit(X_train, y_train, X_test, y_test)

    # 2. 绘制学习曲线
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["train_acc"], label="train_acc")
    plt.plot(history["val_acc"], label="test_acc")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title("训练/测试准确率随epoch变化")
    plt.legend()
    plt.grid(True)
    plt.show()

    # 3. 最终测试集准确率
    acc = clf.evaluate(X_test, y_test)
    print(f"测试集准确率: {acc:.4f}")

    # 4. 尝试不同初始化/学习率
    # for lr in [0.001, 0.01, 0.1]:
    #     clf = MultiClassPerceptron(n_classes=C, learning_rate=lr, epochs=20)
    #     clf.fit(X_train, y_train)
    #     acc = clf.evaluate(X_test, y_test)
    #     print(f"learning_rate={lr}, test_acc={acc:.4f}")


# 如果需要直接运行本文件可取消注释
if __name__ == "__main__":
    run_demo()
