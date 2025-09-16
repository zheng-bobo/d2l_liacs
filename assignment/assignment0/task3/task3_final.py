# -*- coding: utf-8 -*-
"""
任务3:实现 XOR 网络 与 梯度下降算法（可选）

一、任务说明（翻译与要点）
- 从零实现一个两层前馈神经网络:2 个输入、隐藏层 2 个神经元、输出层 1 个神经元。
- 网络共有 9 个可训练权重：
  - 每个隐藏单元各有 3 条入边:来自两个输入与一个常数偏置1 → 共 2x(2权重+1偏置)=6
  - 输出单元也有 3 条入边：来自两个隐藏单元与一个偏置 → 共 3
  - 总计 6+3=9
- 激活函数：所有非输入节点默认使用 sigmoid(可尝试 tanh 或 ReLU 比较差异）。
- 数据集:XOR 的 4 个样本与标签
  (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0
- 误差函数:4 个样本的均方误差(MSE)。
- 梯度：先计算输出层 [w1, w2, b] 的梯度，再反向传播到隐藏层权重 [w11, w12, w21, w22, c1, c2]。
  参考《Understanding Deep Learning》第 7 章的推导。
- 训练：随机初始化权重，使用梯度下降迭代
    weights ← weights - η * gradient
  监控 MSE 与误分类数（输出>0.5 判 1 否则 0)。
  试验不同初始化与学习率；也可尝试“懒办法”：不断随机权重，直到恰好实现 XOR。
- 扩展：尝试 tanh 或 ReLU,观察对训练过程的影响并解释差异。

二、实现简介
- 结构：输入层(2) → 隐藏层(2) → 输出层(1)
- 参数（共 9 个）：
  - 输入→隐藏:W1∈R^{2x2},b1∈R^{2}
  - 隐藏→输出:W2∈R^{2},b2∈R
- 前向：
  h = act(X @ W1 + b1)
  y_hat = act(h @ W2 + b2)   # act=Sigmoid/Tanh/ReLU
- 损失:MSE = mean((y_hat - y)^2)
- 反向（以逐样本或小批为单位；此处对 4 个样本整批计算，显式写出链式法则）
"""

import torch
from typing import Tuple, Dict


def dsigmoid(h):
    # h 已经是 sigmoid(z) 的输出
    return h * (1 - h)


def dtanh(h):
    # h 已经是 tanh(z) 的输出
    return 1 - h**2


def drelu(z):
    return (z > 0).float()


class XORNet:
    """
    2-2-1 的最小 MLP，用于学习 XOR。
    参数总数=9：W1(2x2)=4, b1(2)=2, W2(2)=2, b2(1)=1
    """

    def __init__(
        self,
        activation="sigmoid",
        weight_scale: float = 0.5,
        seed: int = 42,
        dtype=torch.float32,
    ):
        self.dtype = dtype
        self.seed = seed

        if self.seed is not None:
            torch.manual_seed(self.seed)

        self.W1 = torch.normal(
            mean=0.0, std=weight_scale, size=(2, 2), dtype=self.dtype
        )
        self.b1 = torch.normal(mean=0.0, std=weight_scale, size=(2,), dtype=self.dtype)
        self.W2 = torch.normal(mean=0.0, std=weight_scale, size=(2,), dtype=self.dtype)
        self.b2 = torch.normal(mean=0.0, std=weight_scale, size=(), dtype=self.dtype)

        if activation == "sigmoid":
            self.act = torch.sigmoid
        elif activation == "tanh":
            self.act = torch.tanh
        elif activation == "relu":
            self.act = torch.relu
        else:
            raise ValueError("未知激活函数")

        self.activation_name = activation

    def forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        X: (N,2)
        返回 y_hat 以及缓存（用于反向传播）
        """
        z1 = X @ self.W1 + self.b1  # (N,2)
        h = self.act(z1)  # (N,2)
        z2 = h @ self.W2 + self.b2  # (N,)
        y_hat = torch.sigmoid(z2)  # 输出层用 sigmoid
        cache = {"X": X, "z1": z1, "h": h, "z2": z2, "y_hat": y_hat}
        return y_hat, cache

    @staticmethod
    def mse(y_hat: torch.Tensor, y: torch.Tensor) -> float:
        return float(torch.mean((y_hat - y) ** 2).item())

    @staticmethod
    def miscls(y_hat: torch.Tensor, y: torch.Tensor, thr: float = 0.5) -> int:
        y_bin = torch.zeros_like(y_hat, dtype=torch.int)
        y_bin[y_hat >= thr] = 1
        return int(torch.sum(y_bin != y).item())

    def gradients(
        self, cache: Dict[str, torch.Tensor], y: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算整批（4 个样本）的梯度。损失 L = mean((y_hat-y)^2)
        记 N=4。链式法则：
          dL/dy_hat = 2/N * (y_hat - y)
          y_hat = sigmoid(z2) → dy_hat/dz2 = y_hat*(1-y_hat)
          z2 = h·W2 + b2
          h = act(z1), z1 = X·W1 + b1
        """
        X = cache["X"]  # (N,2)
        z1 = cache["z1"]  # (N,2)
        h = cache["h"]  # (N,2)
        z2 = cache["z2"]  # (N,)
        y_hat = cache["y_hat"]  # (N,)

        N = X.shape[0]
        dL_dyhat = (2.0 / N) * (y_hat - y)  # (N,)

        # 输出层：sigmoid
        dyhat_dz2 = y_hat * (1.0 - y_hat)  # (N,)
        dL_dz2 = dL_dyhat * dyhat_dz2  # (N,)

        # 对 W2, b2
        # z2_i = sum_j h_ij * W2_j + b2
        dL_dW2 = h.t() @ dL_dz2  # (2,)
        dL_db2 = torch.sum(dL_dz2)  # ()

        # 传播到隐藏层输出 h
        dL_dh = dL_dz2.unsqueeze(1) * self.W2.unsqueeze(0)  # (N,2)

        # 隐藏层激活导数
        if self.activation_name == "sigmoid":
            dh_dz1 = dsigmoid(h)  # (N,2)
        elif self.activation_name == "tanh":
            dh_dz1 = dtanh(h)
        elif self.activation_name == "relu":
            dh_dz1 = drelu(z1)
        else:
            raise ValueError("未知激活函数")

        dL_dz1 = dL_dh * dh_dz1  # (N,2)

        # z1 = X·W1 + b1
        dL_dW1 = X.t() @ dL_dz1  # (2,2)
        dL_db1 = torch.sum(dL_dz1, dim=0)  # (2,)

        return {
            "W1": dL_dW1,
            "b1": dL_db1,
            "W2": dL_dW2,
            "b2": dL_db2,
        }

    def step(self, grads: Dict[str, torch.Tensor], lr: float) -> None:
        self.W1 -= lr * grads["W1"]
        self.b1 -= lr * grads["b1"]
        self.W2 -= lr * grads["W2"]
        self.b2 -= lr * grads["b2"]

    def train(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        lr: float = 0.5,
        epochs: int = 5000,
        verbose_every: int = 500,
    ) -> Dict[str, list]:
        history = {"mse": [], "mis": []}
        for ep in range(1, epochs + 1):
            y_hat, cache = self.forward(X)
            loss = self.mse(y_hat, y)
            mis = self.miscls(y_hat, y)
            history["mse"].append(loss)
            history["mis"].append(mis)

            grads = self.gradients(cache, y)
            self.step(grads, lr)

            if verbose_every and (ep % verbose_every == 0):
                print(f"[{ep:5d}] mse={loss:.6f} mis={mis}")
        return history

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        y_hat, _ = self.forward(X)
        return (y_hat >= 0.5).int()


def load_xor() -> Tuple[torch.Tensor, torch.Tensor]:
    X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
    y = torch.tensor([0, 1, 1, 0], dtype=torch.float32)
    return X, y


def lazy_search(seed_start: int = 0, trials: int = 10000) -> Tuple[XORNet, int]:
    """
    “懒办法”：不断随机权重，若恰好能实现 XOR（训练集全对且 MSE 很小）则返回。
    注意：对 sigmoid 等，存在一大块参数空间可实现 XOR，但盲抽命中概率仍不高。
    """
    X, y = load_xor()
    for k in range(trials):
        net = XORNet(activation="sigmoid", weight_scale=2.0, seed=seed_start + k)
        y_hat, _ = net.forward(X)
        if (net.miscls(y_hat, y) == 0) and (net.mse(y_hat, y) < 1e-4):
            return net, k + 1
    return net, trials


def run():
    X, y = load_xor()

    # 训练一个使用 sigmoid 的网络（最稳）
    net = XORNet(activation="sigmoid", weight_scale=0.5, seed=0)
    hist = net.train(X, y, lr=0.5, epochs=5000, verbose_every=500)

    y_hat, _ = net.forward(X)
    print("Final mse=", XORNet.mse(y_hat, y), "mis=", XORNet.miscls(y_hat, y))
    print("Outputs:", torch.round(y_hat, decimals=4), "Pred:", net.predict(X))

    # 可选：尝试不同激活/学习率（tanh 常需较小 lr；ReLU 可能卡住，需合适 init 与 lr）
    # for act, lr in [("tanh", 0.1), ("relu", 0.05)]:
    #     net2 = XORNet(activation=act, weight_scale=0.5, seed=1)
    #     print(f"\n[{act}] training...")
    #     net2.fit(X, y, lr=lr, epochs=5000, verbose_every=1000)
    #     y2, _ = net2.forward(X)
    #     print("Final mse=", XORNet.mse(y2, y), "mis=", XORNet.miscls(y2, y))

    # 可选：懒办法
    # found_net, tries = lazy_search(seed_start=100, trials=50000)
    # y3, _ = found_net.forward(X)
    # print(f"\nLazy approach: tries={tries}, mse={XORNet.mse(y3, y)}, mis={XORNet.miscls(y3, y)}")


if __name__ == "__main__":
    run()
