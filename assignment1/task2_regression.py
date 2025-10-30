import argparse
import os
import random

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Input, Model, layers, losses, optimizers


TOTAL_MINUTES = 12 * 60


def parse_args():
    parser = argparse.ArgumentParser(
        description="Regress analog clock time as decimal hours using a CNN."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join("A1 Data", "A1_data_75"),
        help="Directory containing images.npy and labels.npy.",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs.")
    parser.add_argument("--batch-size", type=int, default=64, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay coefficient for AdamW.",
    )
    parser.add_argument("--seed", type=int, default=22, help="Random seed.")
    args, unknown = parser.parse_known_args()
    if unknown:
        print("Ignoring unrecognized arguments: {}".format(unknown))
    return args


def wraparound_minutes(pred_hours, true_hours):
    pred_minutes = (pred_hours % 12.0) * 60.0
    true_minutes = (true_hours % 12.0) * 60.0
    diff = np.abs(pred_minutes - true_minutes)
    return np.minimum(diff, TOTAL_MINUTES - diff)


class ClockRegressionDataset:
    """Load clock images and convert labels to decimal hours in [0,12).
    Note: mapping 11:55 -> 11.916 and 0:05 -> 0.083 introduces wrap-around ambiguity;
    the downstream loss is plain MSE on hours/12, so values near the 12→0 boundary
    can still yield large penalties despite representing similar times."""

    def __init__(self, data_dir, batch_size, seed):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seed = seed

        self.images = None
        self.labels_hm = None
        self.hours = None  # ground truth hours in [0,12)
        self.targets = None  # normalized to [0,1]

        self.train_images = None
        self.val_images = None
        self.test_images = None

        self.train_hours = None
        self.val_hours = None
        self.test_hours = None

        self.train_targets = None
        self.val_targets = None
        self.test_targets = None

        self.input_shape = None

        self.load_data()
        self.preprocess()
        self.split_data()

    def load_data(self):
        images_path = os.path.join(self.data_dir, "images.npy")
        labels_path = os.path.join(self.data_dir, "labels.npy")

        if not os.path.exists(images_path) or not os.path.exists(labels_path):
            raise FileNotFoundError(
                "Could not find images/labels in {}".format(self.data_dir)
            )

        self.images = np.load(images_path)
        self.labels_hm = np.load(labels_path)

        if self.images.ndim != 3:
            raise ValueError(
                "Expected images with shape (N,H,W), got {}".format(self.images.shape)
            )
        if self.labels_hm.shape[-1] != 2:
            raise ValueError("Labels should have shape (N, 2) for hour and minute.")

    def preprocess(self):
        # The original image pixel values are in the range [0,255]. Dividing by 255.0 normalizes them to the [0,1] range.
        images = self.images.astype(np.float32) / 255.0
        # For grayscale images, the original shape is (N, H, W), but the model expects shape (N, H, W, 1).
        # expand_dims adds a channel dimension at the end, making images compatible with the network input requirements.
        images = np.expand_dims(images, -1)

        hours = (self.labels_hm[:, 0] % 12) + (self.labels_hm[:, 1] / 60.0)

        targets = hours / 12.0  # normalize to [0,1] for sigmoid output

        self.images = images
        self.hours = hours
        self.targets = targets.astype(np.float32)
        self.input_shape = images.shape[1:]

    def split_data(self):
        rng = np.random.default_rng(self.seed)
        indices = rng.permutation(len(self.images))

        n_total = len(indices)
        n_train = int(0.8 * n_total)
        n_val = int(0.1 * n_total)

        train_idx = indices[:n_train]
        val_idx = indices[n_train : n_train + n_val]
        test_idx = indices[n_train + n_val :]

        self.train_images = self.images[train_idx]
        self.val_images = self.images[val_idx]
        self.test_images = self.images[test_idx]

        self.train_hours = self.hours[train_idx]
        self.val_hours = self.hours[val_idx]
        self.test_hours = self.hours[test_idx]

        self.train_targets = self.targets[train_idx]
        self.val_targets = self.targets[val_idx]
        self.test_targets = self.targets[test_idx]


class CommonSenseRegressionCallback(tf.keras.callbacks.Callback):
    """Compute common-sense time error after each epoch."""

    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer

    def on_epoch_end(self, epoch, logs=None):
        dataset = self.trainer.dataset
        preds = self.model.predict(
            dataset.val_images, batch_size=dataset.batch_size, verbose=0
        )
        preds_hours = (preds.squeeze() * 12.0).clip(min=0.0)
        diffs = wraparound_minutes(preds_hours, dataset.val_hours)

        metrics_dict = {
            "mean_diff": float(np.nanmean(diffs)),
            "median_diff": float(np.nanmedian(diffs)),
            "pct_within_5": float(np.nanmean(diffs <= 5)),
            "pct_within_15": float(np.nanmean(diffs <= 15)),
        }
        self.trainer.last_val_metrics = metrics_dict
        self.trainer.cs_history.append(metrics_dict)

        print(
            "  -> val_mean_diff={:.2f} min, median={:.2f}, within5={:.3f}, within15={:.3f}".format(
                metrics_dict["mean_diff"],
                metrics_dict["median_diff"],
                metrics_dict["pct_within_5"],
                metrics_dict["pct_within_15"],
            )
        )

        if metrics_dict["mean_diff"] < self.trainer.best_val_diff:
            self.trainer.best_val_diff = metrics_dict["mean_diff"]
            self.trainer.best_epoch = epoch + 1
            self.trainer.best_weights = self.model.get_weights()

        if logs is not None:
            logs["val_mean_diff"] = metrics_dict["mean_diff"]
            logs["val_median_diff"] = metrics_dict["median_diff"]
            logs["val_within_5"] = metrics_dict["pct_within_5"]
            logs["val_within_15"] = metrics_dict["pct_within_15"]


class ClockRegressionTrainer:
    """CNN regressor predicting decimal hours (0-12)."""

    def __init__(self, dataset, learning_rate, weight_decay):
        if dataset.input_shape is None:
            raise ValueError("Dataset input_shape is missing.")

        self.dataset = dataset
        self.model = self.build_model(dataset.input_shape)
        self.optimizer = optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        )

        # 这里使用 model.compile 方法配置模型的训练方式。
        # optimizer 指定了优化器为 AdamW，对训练权重的更新提供方式。
        # loss 设置为均方误差（MSE），用来衡量预测值与真实值的差异。
        # metrics=["mae"] 表示在训练和验证时还会额外监控平均绝对误差（MAE），便于评估模型性能。
        self.model.compile(
            optimizer=self.optimizer,
            loss=losses.MeanSquaredError(),
            metrics=["mae"],
        )

        self.best_epoch = 0
        self.best_val_diff = float("inf")
        self.best_weights = None
        self.last_val_metrics = None
        self.cs_history = []
        self.history_obj = None

    @staticmethod
    def build_model(input_shape):
        inputs = Input(shape=input_shape)
        x = layers.Conv2D(32, 3, padding="same", use_bias=False)(inputs)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPool2D()(x)

        x = layers.Conv2D(64, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPool2D()(x)

        x = layers.Conv2D(128, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.MaxPool2D()(x)

        x = layers.Conv2D(256, 3, padding="same", use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.GlobalAveragePooling2D()(x)

        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(1, activation="sigmoid")(x)  # scaled to [0,1]

        return Model(inputs=inputs, outputs=outputs)

    def fit(self, epochs):
        callback = CommonSenseRegressionCallback(self)
        history = self.model.fit(
            self.dataset.train_images,
            self.dataset.train_targets,
            epochs=epochs,
            batch_size=self.dataset.batch_size,
            shuffle=True,
            validation_data=(self.dataset.val_images, self.dataset.val_targets),
            callbacks=[callback],
            verbose=2,
        )
        self.history_obj = history.history

        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            print(
                "Loaded best model from epoch {} (val mean diff {:.2f} minutes).".format(
                    self.best_epoch, self.best_val_diff
                )
            )

    def evaluate_test(self):
        loss, mae = self.model.evaluate(
            self.dataset.test_images,
            self.dataset.test_targets,
            batch_size=self.dataset.batch_size,
            verbose=0,
        )
        preds = self.model.predict(
            self.dataset.test_images, batch_size=self.dataset.batch_size, verbose=0
        )
        preds_hours = (preds.squeeze() * 12.0).clip(min=0.0)
        diffs = wraparound_minutes(preds_hours, self.dataset.test_hours)

        metrics_dict = {
            "loss": float(loss),
            "mae_hours": float(mae * 12.0),
            "mean_diff": float(np.nanmean(diffs)),
            "median_diff": float(np.nanmedian(diffs)),
            "pct_within_5": float(np.nanmean(diffs <= 5)),
            "pct_within_15": float(np.nanmean(diffs <= 15)),
        }

        print(
            "Test: loss={:.4f} mae(hours)={:.3f} mean_diff={:.2f} median_diff={:.2f} within_5={:.3f} within_15={:.3f}".format(
                metrics_dict["loss"],
                metrics_dict["mae_hours"],
                metrics_dict["mean_diff"],
                metrics_dict["median_diff"],
                metrics_dict["pct_within_5"],
                metrics_dict["pct_within_15"],
            )
        )
        return metrics_dict

    def plot_metrics(self):
        if not self.history_obj:
            print("No training history to plot.")
            return

        epochs = range(1, len(self.history_obj["loss"]) + 1)

        train_loss = self.history_obj["loss"]
        val_loss = self.history_obj.get("val_loss", [])

        train_mae = self.history_obj["mae"]
        val_mae = self.history_obj.get("val_mae", [])
        train_accuracy = [max(0.0, 1.0 - m) for m in train_mae]
        val_accuracy = [max(0.0, 1.0 - m) for m in val_mae]

        cs_epochs = range(1, len(self.cs_history) + 1)
        cs_accuracy = [
            max(0.0, 1.0 - entry["mean_diff"] / TOTAL_MINUTES)
            for entry in self.cs_history
        ]

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_accuracy, label="Train Accuracy")
        if val_accuracy:
            plt.plot(epochs, val_accuracy, label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (approx)")
        plt.title("Training Accuracy (1 - MAE)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("regression_accuracy.png", dpi=150)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, train_loss, label="Train Loss")
        if val_loss:
            plt.plot(epochs, val_loss, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.title("Training Loss")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("regression_loss.png", dpi=150)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(cs_epochs, cs_accuracy, label="Val Common-Sense Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (1 - mean_diff/720)")
        plt.title("Validation Common-Sense Accuracy")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("regression_common_sense_accuracy.png", dpi=150)
        plt.close()


def configure_device():
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("Using GPU: {}".format(gpus[0].name))
        except RuntimeError as exc:
            print(
                "GPU configuration failed ({}); using default device settings.".format(
                    exc
                )
            )
    else:
        print("No GPU detected; falling back to CPU.")


def main():
    args = parse_args()

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    configure_device()

    dataset = ClockRegressionDataset(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    trainer = ClockRegressionTrainer(
        dataset=dataset,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )

    trainer.fit(args.epochs)
    trainer.evaluate_test()
    trainer.plot_metrics()


if __name__ == "__main__":
    main()
