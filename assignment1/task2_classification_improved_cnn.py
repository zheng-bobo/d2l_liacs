import argparse
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras import Input, Model, layers, losses, metrics, optimizers


TOTAL_MINUTES = 12 * 60


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a CNN to read analog clocks with configurable label granularity."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=os.path.join("A1 Data", "A1_data_75"),
        help="Directory containing images.npy and labels.npy.",
    )
    parser.add_argument(
        "--epochs", type=int, default=40, help="Number of training epochs."
    )
    parser.add_argument("--batch-size", type=int, default=48, help="Mini-batch size.")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate.")
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay coefficient for AdamW optimizer.",
    )
    parser.add_argument(
        "--seed", type=int, default=22, help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        nargs="+",
        default=[24, 144, 720],
        help="One or more class counts to train (must divide 720 evenly).",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        print("Ignoring unrecognized arguments: {}".format(unknown))
    return args


def wraparound_diff(pred_minutes, true_minutes):
    diff = np.abs(pred_minutes - true_minutes)
    return np.minimum(diff, TOTAL_MINUTES - diff)


class ClockDataset:
    """Load, preprocess, split, and serve the clock dataset."""

    def __init__(self, data_dir, batch_size, seed, num_classes):
        if TOTAL_MINUTES % num_classes != 0:
            raise ValueError("num_classes must divide {} evenly.".format(TOTAL_MINUTES))

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seed = seed
        self.num_classes = num_classes
        self.minutes_per_class = TOTAL_MINUTES // num_classes

        self.images = None
        self.labels_hm = None
        self.minutes = None

        self.train_images = None
        self.val_images = None
        self.test_images = None

        self.train_labels = None
        self.val_labels = None
        self.test_labels = None

        self.train_minutes = None
        self.val_minutes = None
        self.test_minutes = None

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
        minutes = (self.labels_hm[:, 0] * 60 + self.labels_hm[:, 1]).astype(np.int32)

        self.images = images
        self.minutes = minutes
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

        self.train_minutes = self.minutes[train_idx]
        self.val_minutes = self.minutes[val_idx]
        self.test_minutes = self.minutes[test_idx]

        self.train_labels = self.train_minutes // self.minutes_per_class
        self.val_labels = self.val_minutes // self.minutes_per_class
        self.test_labels = self.test_minutes // self.minutes_per_class


class CommonSenseCallback(tf.keras.callbacks.Callback):
    """Evaluate common-sense error after each epoch and track best weights."""

    def __init__(self, trainer):
        super().__init__()
        self.trainer = trainer

    def on_epoch_end(self, epoch, logs=None):
        dataset = self.trainer.dataset
        logits = self.model.predict(
            dataset.val_images, batch_size=dataset.batch_size, verbose=0
        )
        preds = np.argmax(logits, axis=1)
        preds_minutes = preds * self.trainer.minutes_per_class
        diffs = wraparound_diff(preds_minutes, dataset.val_minutes)

        metrics_dict = {
            "mean_diff": float(np.nanmean(diffs)),
            "median_diff": float(np.nanmedian(diffs)),
            "pct_within_5": float(np.nanmean(diffs <= 5)),
            "pct_within_15": float(np.nanmean(diffs <= 15)),
        }
        self.trainer.last_val_metrics = metrics_dict

        print(
            "val_mean_diff={:.2f} min, median={:.2f}, within5={:.3f}, within15={:.3f}".format(
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


class ClockDataClassificationTrainer:
    """Train and evaluate a CNN clock reader."""

    def __init__(self, dataset, learning_rate, weight_decay):
        if dataset.input_shape is None:
            raise ValueError("Dataset input_shape is missing.")

        self.dataset = dataset
        self.num_classes = dataset.num_classes
        self.minutes_per_class = dataset.minutes_per_class

        self.model = self.build_model(dataset.input_shape, self.num_classes)
        self.loss_fn = losses.SparseCategoricalCrossentropy(from_logits=False)
        self.optimizer = optimizers.AdamW(
            learning_rate=learning_rate, weight_decay=weight_decay
        )
        self.model.compile(
            optimizer=self.optimizer, loss=self.loss_fn, metrics=["accuracy"]
        )

        self.best_epoch = 0
        self.best_val_diff = float("inf")
        self.best_weights = None
        self.last_val_metrics = None

    @staticmethod
    def build_model(input_shape, num_classes):
        inputs = Input(shape=input_shape)
        aug_layers = tf.keras.Sequential(
            [
                layers.RandomRotation(0.1),
                layers.RandomZoom(0.1),
                layers.RandomContrast(0.15),
            ],
            name="augmentation",
        )
        x = aug_layers(inputs)

        def conv_block(x, filters, pool=True):
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.ReLU()(x)
            if pool:
                x = layers.MaxPool2D()(x)
            return x

        x = conv_block(x, 32)
        x = conv_block(x, 64)
        x = conv_block(x, 128)

        height = input_shape[0]
        if height >= 128:
            x = conv_block(x, 256)
            x = conv_block(x, 512, pool=False)
        else:
            x = conv_block(x, 256, pool=False)

        x = layers.GlobalAveragePooling2D()(x)
        x = layers.Dropout(0.4)(x)
        x = layers.Dense(512, activation="relu")(x)
        x = layers.Dropout(0.3)(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)

        return Model(inputs=inputs, outputs=outputs)

    def fit(self, epochs):
        cs_callback = CommonSenseCallback(self)
        # 'plateau' is used to automatically reduce the learning rate when the validation loss ('val_loss') plateaus, helping the model escape local minima. For instance, factor=0.5 halves the learning rate, and patience=5 means it will wait for 5 epochs before making an adjustment.
        plateau = tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5, min_lr=1e-5, verbose=1
        )
        # 'early' is used to stop training early if the validation loss does not improve for several epochs, helping to prevent overfitting. Patience=10 allows up to 10 epochs without improvement, and restore_best_weights will automatically reload the best weights achieved during training.
        early = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True, verbose=1
        )
        self.model.fit(
            self.dataset.train_images,
            self.dataset.train_labels,
            epochs=epochs,
            batch_size=self.dataset.batch_size,
            shuffle=True,
            validation_data=(self.dataset.val_images, self.dataset.val_labels),
            callbacks=[cs_callback, plateau, early],
            verbose=2,
        )

        if self.best_weights is not None:
            self.model.set_weights(self.best_weights)
            print(
                "Loaded best model from epoch {} (val mean diff {:.2f} minutes).".format(
                    self.best_epoch, self.best_val_diff
                )
            )

    def evaluate_test(self):
        loss, acc = self.model.evaluate(
            self.dataset.test_images,
            self.dataset.test_labels,
            batch_size=self.dataset.batch_size,
            verbose=0,
        )

        logits = self.model.predict(
            self.dataset.test_images, batch_size=self.dataset.batch_size, verbose=0
        )
        preds = np.argmax(logits, axis=1)
        preds_minutes = preds * self.minutes_per_class
        diffs = wraparound_diff(preds_minutes, self.dataset.test_minutes)

        test_metrics = {
            "loss": float(loss),
            "acc": float(acc),
            "mean_diff": float(np.nanmean(diffs)),
            "median_diff": float(np.nanmedian(diffs)),
            "pct_within_5": float(np.nanmean(diffs <= 5)),
            "pct_within_15": float(np.nanmean(diffs <= 15)),
        }

        print(
            "Test: loss={:.4f} acc={:.3f} mean_diff={:.2f} median_diff={:.2f} within_5={:.3f} within_15={:.3f}".format(
                test_metrics["loss"],
                test_metrics["acc"],
                test_metrics["mean_diff"],
                test_metrics["median_diff"],
                test_metrics["pct_within_5"],
                test_metrics["pct_within_15"],
            )
        )
        return test_metrics


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

    for num_classes in args.num_classes:
        print("\n" + "=" * 80)
        print(
            "Training with {} classes ({} minutes per class)".format(
                num_classes, TOTAL_MINUTES // num_classes
            )
        )
        print("=" * 80)

        dataset = ClockDataset(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            seed=args.seed,
            num_classes=num_classes,
        )

        trainer = ClockDataClassificationTrainer(
            dataset=dataset,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
        )

        trainer.fit(args.epochs)
        trainer.evaluate_test()


if __name__ == "__main__":
    main()
