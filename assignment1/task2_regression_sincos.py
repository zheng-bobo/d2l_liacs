import argparse
import os
import random

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import Input, Model, layers, losses, optimizers


TOTAL_MINUTES = 12 * 60
HOUR_CLASSES = 12


def parse_args():
    parser = argparse.ArgumentParser(
        description="Multi-head CNN with hour classification + sine/cosine minute regression."
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
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "gpu"],
        help="Device preference. 'auto' uses GPU when available.",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        print("Ignoring unrecognized arguments: {}".format(unknown))
    return args


def wraparound_minutes(pred_minutes, true_minutes):
    diff = np.abs(pred_minutes - true_minutes)
    return np.minimum(diff, TOTAL_MINUTES - diff)


class ClockSineCosineDataset:
    """Prepare images + hour classes + sine/cosine of minute angle."""

    def __init__(self, data_dir, batch_size, seed):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seed = seed

        self.images = None
        self.labels_hm = None

        self.train_images = None
        self.val_images = None
        self.test_images = None

        self.train_hours = None
        self.val_hours = None
        self.test_hours = None

        self.train_hour_onehot = None
        self.val_hour_onehot = None
        self.test_hour_onehot = None

        self.train_min_sin_cos = None
        self.val_min_sin_cos = None
        self.test_min_sin_cos = None

        self.train_minutes_full = None
        self.val_minutes_full = None
        self.test_minutes_full = None

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
        images = self.images.astype(np.float32) / 255.0
        images = np.expand_dims(images, -1)

        hours = (self.labels_hm[:, 0] % 12).astype(np.int32)
        minutes = self.labels_hm[:, 1].astype(np.float32)
        angle = minutes / 60.0 * 2 * np.pi
        sin_cos = np.stack([np.sin(angle), np.cos(angle)], axis=1).astype(np.float32)

        hour_onehot = tf.keras.utils.to_categorical(hours, HOUR_CLASSES).astype(
            np.float32
        )

        self.images = images
        self.hours = hours
        self.hour_onehot = hour_onehot
        self.sin_cos_targets = sin_cos
        self.minutes_full = hours * 60.0 + minutes
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

        self.train_hour_onehot = self.hour_onehot[train_idx]
        self.val_hour_onehot = self.hour_onehot[val_idx]
        self.test_hour_onehot = self.hour_onehot[test_idx]

        self.train_min_sin_cos = self.sin_cos_targets[train_idx]
        self.val_min_sin_cos = self.sin_cos_targets[val_idx]
        self.test_min_sin_cos = self.sin_cos_targets[test_idx]

        self.train_minutes_full = self.minutes_full[train_idx]
        self.val_minutes_full = self.minutes_full[val_idx]
        self.test_minutes_full = self.minutes_full[test_idx]


class CommonSenseSineCallback(tf.keras.callbacks.Callback):
    """Compute wrap-around minutes for sine/cosine minute head."""

    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer

    def on_epoch_end(self, epoch, logs=None):
        dataset = self.trainer.dataset
        hour_probs, sin_cos = self.model.predict(
            dataset.val_images, batch_size=dataset.batch_size, verbose=0
        )
        hour_pred = np.argmax(hour_probs, axis=1)
        sin_vals = sin_cos[:, 0]
        cos_vals = sin_cos[:, 1]
        angles = np.arctan2(sin_vals, cos_vals)
        angles = angles % (2 * np.pi)
        minute_pred = angles / (2 * np.pi) * 60.0
        pred_minutes = hour_pred * 60.0 + minute_pred
        diffs = wraparound_minutes(pred_minutes, dataset.val_minutes_full)

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


class ClockSineCosineTrainer:
    """Hour classification + sine/cosine minute regression trainer."""

    def __init__(self, dataset, learning_rate, weight_decay):
        if dataset.input_shape is None:
            raise ValueError("Dataset input_shape is missing.")

        self.dataset = dataset
        self.model = self.build_model(dataset.input_shape)
        self.optimizer = optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        )

        self.model.compile(
            optimizer=self.optimizer,
            loss={
                "hour_head": losses.CategoricalCrossentropy(from_logits=False),
                "minute_head": losses.MeanSquaredError(),
            },
            metrics={
                "hour_head": ["accuracy"],
                "minute_head": [tf.keras.metrics.MeanAbsoluteError()],
            },
            loss_weights={"hour_head": 1.0, "minute_head": 1.0},
        )

        self.best_epoch = 0
        self.best_val_diff = float("inf")
        self.best_weights = None
        self.last_val_metrics = None
        self.history_obj = None
        self.cs_history = []

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

        hour_output = layers.Dense(
            HOUR_CLASSES, activation="softmax", name="hour_head"
        )(x)
        minute_output = layers.Dense(2, activation="tanh", name="minute_head")(x)

        return Model(inputs=inputs, outputs=[hour_output, minute_output])

    def fit(self, epochs):
        callback = CommonSenseSineCallback(self)
        history = self.model.fit(
            self.dataset.train_images,
            {
                "hour_head": self.dataset.train_hour_onehot,
                "minute_head": self.dataset.train_min_sin_cos,
            },
            epochs=epochs,
            batch_size=self.dataset.batch_size,
            shuffle=True,
            validation_data=(
                self.dataset.val_images,
                {
                    "hour_head": self.dataset.val_hour_onehot,
                    "minute_head": self.dataset.val_min_sin_cos,
                },
            ),
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
        results = self.model.evaluate(
            self.dataset.test_images,
            {
                "hour_head": self.dataset.test_hour_onehot,
                "minute_head": self.dataset.test_min_sin_cos,
            },
            batch_size=self.dataset.batch_size,
            verbose=0,
            return_dict=True,
        )

        hour_probs, sin_cos = self.model.predict(
            self.dataset.test_images, batch_size=self.dataset.batch_size, verbose=0
        )
        hour_pred = np.argmax(hour_probs, axis=1)
        sin_vals = sin_cos[:, 0]
        cos_vals = sin_cos[:, 1]
        angles = np.arctan2(sin_vals, cos_vals)
        angles = (angles + 2 * np.pi) % (2 * np.pi)
        minute_pred = angles / (2 * np.pi) * 60.0
        pred_minutes = hour_pred * 60.0 + minute_pred
        diffs = wraparound_minutes(pred_minutes, self.dataset.test_minutes_full)

        true_angle = (self.dataset.test_minutes_full % 60.0) / 60.0 * 2 * np.pi
        angle_diff = np.arctan2(
            np.sin(angles - true_angle), np.cos(angles - true_angle)
        )
        angle_mae = float(np.mean(np.abs(angle_diff)))

        metrics_dict = {
            "loss": float(results["loss"]),
            "hour_acc": float(results.get("hour_head_accuracy", np.nan)),
            "minute_angle_mae": angle_mae,
            "mean_diff": float(np.nanmean(diffs)),
            "median_diff": float(np.nanmedian(diffs)),
            "pct_within_5": float(np.nanmean(diffs <= 5)),
            "pct_within_15": float(np.nanmean(diffs <= 15)),
        }

        print(
            "Test: loss={:.4f} hour_acc={:.3f} minute_angle_mae(rad)={:.3f} mean_diff={:.2f} median_diff={:.2f} within_5={:.3f} within_15={:.3f}".format(
                metrics_dict["loss"],
                metrics_dict["hour_acc"],
                metrics_dict["minute_angle_mae"],
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

        hour_acc = self.history_obj.get("hour_head_accuracy", [])
        val_hour_acc = self.history_obj.get("val_hour_head_accuracy", [])

        minute_mae = self.history_obj.get("minute_head_mean_absolute_error", [])
        val_minute_mae = self.history_obj.get("val_minute_head_mean_absolute_error", [])

        total_loss = self.history_obj["loss"]
        val_total_loss = self.history_obj.get("val_loss", [])

        hour_loss = self.history_obj.get("hour_head_loss", [])
        val_hour_loss = self.history_obj.get("val_hour_head_loss", [])

        minute_loss = self.history_obj.get("minute_head_loss", [])
        val_minute_loss = self.history_obj.get("val_minute_head_loss", [])

        cs_epochs = range(1, len(self.cs_history) + 1)
        cs_accuracy = [
            max(0.0, 1.0 - entry["mean_diff"] / TOTAL_MINUTES)
            for entry in self.cs_history
        ]

        plt.figure(figsize=(8, 5))
        if hour_acc:
            plt.plot(epochs, hour_acc, label="Train Hour Acc")
        if val_hour_acc:
            plt.plot(epochs, val_hour_acc, label="Val Hour Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Hour Accuracy (Sine/Cosine)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("sincos_accuracy.png", dpi=150)
        plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(epochs, total_loss, label="Train Total Loss")
        if val_total_loss:
            plt.plot(epochs, val_total_loss, label="Val Total Loss")
        if hour_loss:
            plt.plot(epochs, hour_loss, "--", label="Train Hour Loss")
        if val_hour_loss:
            plt.plot(epochs, val_hour_loss, "--", label="Val Hour Loss")
        if minute_loss:
            plt.plot(epochs, minute_loss, "--", label="Train Minute Loss")
        if val_minute_loss:
            plt.plot(epochs, val_minute_loss, "--", label="Val Minute Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Loss Curves (Sine/Cosine)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("sincos_losses.png", dpi=150)
        plt.close()

        if minute_mae or val_minute_mae:
            plt.figure(figsize=(8, 5))
            if minute_mae:
                plt.plot(epochs, minute_mae, label="Train Minute Head MAE")
            if val_minute_mae:
                plt.plot(epochs, val_minute_mae, label="Val Minute Head MAE")
            plt.xlabel("Epoch")
            plt.ylabel("MAE (sin/cos space)")
            plt.title("Minute Head MAE (Sine/Cosine)")
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig("sincos_minute_mae.png", dpi=150)
            plt.close()

        plt.figure(figsize=(8, 5))
        plt.plot(cs_epochs, cs_accuracy, label="Val Common-Sense Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy (1 - mean_diff/720)")
        plt.title("Validation Common-Sense Accuracy")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig("sincos_common_sense_accuracy.png", dpi=150)
        plt.close()


def configure_device(preference):
    if preference == "cpu":
        try:
            tf.config.set_visible_devices([], "GPU")
            print("Forcing CPU execution as requested.")
        except RuntimeError as exc:
            print("Could not disable GPU ({}). Continuing on CPU.".format(exc))
        return

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
        if preference == "gpu":
            print("No GPU detected; falling back to CPU.")


def main():
    args = parse_args()

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    configure_device(args.device)

    dataset = ClockSineCosineDataset(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seed=args.seed,
    )

    trainer = ClockSineCosineTrainer(
        dataset=dataset,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
    )

    trainer.fit(args.epochs)
    trainer.evaluate_test()
    trainer.plot_metrics()


if __name__ == "__main__":
    main()
