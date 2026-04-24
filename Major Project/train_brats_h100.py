import argparse
import json
import os
from pathlib import Path

# Set TF/CUDA env vars before importing TensorFlow.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Concatenate,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Input,
    LeakyReLU,
    MaxPooling2D,
    UpSampling2D,
    multiply,
)
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K


def configure_gpu_and_precision() -> None:
    physical_devices = tf.config.list_physical_devices("GPU")
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
            print(f"Memory growth enabled for {device}")

    policy = mixed_precision.Policy("mixed_float16")
    mixed_precision.set_global_policy(policy)
    print(f"Num GPUs Available: {len(physical_devices)}")
    print(f"Mixed Precision Policy: {policy.name}")


def load_h5_arrays(h5_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(h5_path, "r") as f:
        x = f["X"][:].astype(np.float32)
        y = f["Y"][:].astype(np.float32)
    return x, y


def augment_pair(x: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    # Keep image and mask transforms synchronized with stateless ops.
    seed = tf.random.uniform((2,), maxval=2**31 - 1, dtype=tf.int32)
    x = tf.image.stateless_random_flip_left_right(x, seed)
    y = tf.image.stateless_random_flip_left_right(y, seed)

    seed = tf.random.uniform((2,), maxval=2**31 - 1, dtype=tf.int32)
    x = tf.image.stateless_random_flip_up_down(x, seed)
    y = tf.image.stateless_random_flip_up_down(y, seed)

    k = tf.random.uniform((), minval=0, maxval=4, dtype=tf.int32)
    x = tf.image.rot90(x, k=k)
    y = tf.image.rot90(y, k=k)
    return x, y


def to_multi_output(x: tf.Tensor, y: tf.Tensor) -> tuple[tf.Tensor, dict[str, tf.Tensor]]:
    return x, {"final_output": y, "out_3": y, "out_2": y, "out_1": y}


def make_dataset(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    augment: bool,
) -> tf.data.Dataset:
    ds = tf.data.Dataset.from_tensor_slices((x, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(x), reshuffle_each_iteration=True)
    if augment:
        ds = ds.map(augment_pair, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.map(to_multi_output, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def attention_block2d(x, g, inter_channel, i, data_format="channels_last"):
    theta_x = Conv2D(
        inter_channel,
        [2, 2],
        strides=[2, 2],
        data_format=data_format,
        kernel_regularizer=regularizers.l2(1e-5),
    )(x)
    phi_g = Conv2D(
        inter_channel,
        [1, 1],
        strides=[1, 1],
        dilation_rate=i,
        data_format=data_format,
        kernel_regularizer=regularizers.l2(1e-5),
    )(g)
    f = LeakyReLU(negative_slope=0.2)(Add()([theta_x, phi_g]))
    psi_f = Conv2D(
        1,
        [1, 1],
        strides=[1, 1],
        dilation_rate=i,
        data_format=data_format,
        kernel_regularizer=regularizers.l2(1e-5),
    )(f)
    sigm_psi_f = Activation(activation="sigmoid")(psi_f)
    rate = UpSampling2D(size=[2, 2])(sigm_psi_f)
    att_x = multiply([x, rate])
    return att_x


def res_block(x, nb_filters, strides, i):
    res_path = BatchNormalization()(x)
    res_path = LeakyReLU(negative_slope=0.2)(res_path)
    pool = MaxPooling2D(pool_size=(2, 2))(res_path)
    res_path = Conv2D(
        filters=nb_filters[0],
        kernel_size=(3, 3),
        padding="same",
        dilation_rate=i,
        strides=strides[1],
        kernel_regularizer=regularizers.l2(1e-5),
    )(pool)
    res_path = BatchNormalization()(res_path)
    res_path = LeakyReLU(negative_slope=0.2)(res_path)
    res_path = Conv2D(
        filters=nb_filters[1],
        kernel_size=(3, 3),
        padding="same",
        dilation_rate=i,
        strides=strides[1],
        kernel_regularizer=regularizers.l2(1e-5),
    )(res_path)
    shortcut = Conv2D(
        nb_filters[1],
        kernel_size=(1, 1),
        dilation_rate=i,
        strides=strides[1],
        kernel_regularizer=regularizers.l2(1e-5),
    )(pool)
    shortcut = BatchNormalization()(shortcut)
    res_path = Add()([shortcut, res_path])
    return res_path


def encoder(x):
    to_decoder = []
    main_path = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        dilation_rate=(1, 1),
        strides=(1, 1),
        kernel_regularizer=regularizers.l2(1e-5),
    )(x)
    main_path = BatchNormalization()(main_path)
    main_path = LeakyReLU(negative_slope=0.2)(main_path)
    main_path = Conv2D(
        filters=64,
        kernel_size=(3, 3),
        padding="same",
        dilation_rate=(1, 1),
        strides=(1, 1),
        kernel_regularizer=regularizers.l2(1e-5),
    )(main_path)
    shortcut = Conv2D(
        filters=64,
        kernel_size=(1, 1),
        strides=(1, 1),
        dilation_rate=(1, 1),
        kernel_regularizer=regularizers.l2(1e-5),
    )(x)
    shortcut = BatchNormalization()(shortcut)
    main_path = Add()([shortcut, main_path])
    to_decoder.append(main_path)
    main_path = res_block(main_path, [128, 128], [(2, 2), (1, 1)], (1, 1))
    to_decoder.append(main_path)
    main_path = res_block(main_path, [256, 256], [(2, 2), (1, 1)], (2, 2))
    to_decoder.append(main_path)
    main_path = res_block(main_path, [512, 512], [(2, 2), (1, 1)], (4, 4))
    to_decoder.append(main_path)
    return to_decoder


def res_block_decoder(x, nb_filters, strides, i):
    res_path = BatchNormalization()(x)
    res_path = LeakyReLU(negative_slope=0.2)(res_path)
    res_path = Conv2D(
        filters=nb_filters[0],
        kernel_size=(3, 3),
        padding="same",
        dilation_rate=i,
        strides=strides[1],
        kernel_regularizer=regularizers.l2(1e-5),
    )(res_path)
    res_path = BatchNormalization()(res_path)
    res_path = LeakyReLU(negative_slope=0.2)(res_path)
    res_path = Conv2D(
        filters=nb_filters[1],
        kernel_size=(3, 3),
        padding="same",
        dilation_rate=i,
        strides=strides[1],
        kernel_regularizer=regularizers.l2(1e-5),
    )(res_path)
    shortcut = Conv2D(
        nb_filters[1],
        kernel_size=(1, 1),
        strides=strides[1],
        dilation_rate=i,
        kernel_regularizer=regularizers.l2(1e-5),
    )(x)
    shortcut = BatchNormalization()(shortcut)
    res_path = Add()([shortcut, res_path])
    return res_path


def three_times_sample(main_path):
    hyper1 = Conv2DTranspose(
        256,
        (2, 2),
        strides=(2, 2),
        padding="same",
        dilation_rate=(1, 1),
        kernel_regularizer=regularizers.l2(1e-5),
    )(main_path)
    hyper2 = Conv2DTranspose(
        128,
        (2, 2),
        strides=(2, 2),
        padding="same",
        dilation_rate=(1, 1),
        kernel_regularizer=regularizers.l2(1e-5),
    )(hyper1)
    hyper3 = Conv2DTranspose(
        64,
        (2, 2),
        strides=(2, 2),
        padding="same",
        dilation_rate=(1, 1),
        kernel_regularizer=regularizers.l2(1e-5),
    )(hyper2)
    return hyper3


def two_times_sample(main_path):
    hyper1 = Conv2DTranspose(
        128,
        (2, 2),
        strides=(2, 2),
        padding="same",
        dilation_rate=(1, 1),
        kernel_regularizer=regularizers.l2(1e-5),
    )(main_path)
    hyper2 = Conv2DTranspose(
        64,
        (2, 2),
        strides=(2, 2),
        padding="same",
        dilation_rate=(1, 1),
        kernel_regularizer=regularizers.l2(1e-5),
    )(hyper1)
    return hyper2


def one_time_sample(main_path):
    hyper1 = Conv2DTranspose(
        64,
        (2, 2),
        strides=(2, 2),
        padding="same",
        dilation_rate=(1, 1),
        kernel_regularizer=regularizers.l2(1e-5),
    )(main_path)
    return hyper1


def decoder(x, from_encoder, dropout_rate=0.2):
    attention_path1 = attention_block2d(from_encoder[3], x, 256, (1, 1), data_format="channels_last")
    main_path1 = Conv2DTranspose(
        512,
        (2, 2),
        strides=(2, 2),
        padding="same",
        dilation_rate=(1, 1),
        kernel_regularizer=regularizers.l2(1e-5),
    )(x)
    main_path1 = Concatenate()([main_path1, attention_path1])
    main_path1 = res_block_decoder(main_path1, [512, 512], [(1, 1), (1, 1)], (4, 4))
    main_path1 = Dropout(dropout_rate)(main_path1)
    hc1 = three_times_sample(main_path1)
    out_1 = Conv2D(filters=4, kernel_size=(1, 1), activation="softmax", name="out_1")(hc1)

    attention_path2 = attention_block2d(from_encoder[2], main_path1, 128, (1, 1), data_format="channels_last")
    main_path2 = Conv2DTranspose(
        256,
        (2, 2),
        strides=(2, 2),
        dilation_rate=(1, 1),
        padding="same",
        kernel_regularizer=regularizers.l2(1e-5),
    )(main_path1)
    main_path2 = Concatenate()([main_path2, attention_path2])
    main_path2 = res_block_decoder(main_path2, [256, 256], [(1, 1), (1, 1)], (2, 2))
    main_path2 = Dropout(dropout_rate)(main_path2)
    hc2 = two_times_sample(main_path2)
    out_2 = Conv2D(filters=4, kernel_size=(1, 1), activation="softmax", name="out_2")(hc2)

    attention_path3 = attention_block2d(from_encoder[1], main_path2, 64, (1, 1), data_format="channels_last")
    main_path3 = Conv2DTranspose(
        128,
        (2, 2),
        strides=(2, 2),
        padding="same",
        dilation_rate=(1, 1),
        kernel_regularizer=regularizers.l2(1e-5),
    )(main_path2)
    main_path3 = Concatenate()([main_path3, attention_path3])
    main_path3 = res_block_decoder(main_path3, [128, 128], [(1, 1), (1, 1)], (1, 1))
    main_path3 = Dropout(dropout_rate)(main_path3)
    hc3 = one_time_sample(main_path3)
    out_3 = Conv2D(filters=4, kernel_size=(1, 1), activation="softmax", name="out_3")(hc3)

    attention_path4 = attention_block2d(from_encoder[0], main_path3, 32, (1, 1), data_format="channels_last")
    main_path4 = Conv2DTranspose(
        64,
        (2, 2),
        strides=(2, 2),
        padding="same",
        dilation_rate=(1, 1),
        kernel_regularizer=regularizers.l2(1e-5),
    )(main_path3)
    main_path4 = Concatenate()([main_path4, attention_path4])
    main_path4 = res_block_decoder(main_path4, [64, 64], [(1, 1), (1, 1)], (1, 1))
    main_path4 = Dropout(dropout_rate)(main_path4)
    return main_path4, out_3, out_2, out_1


def aru_gd(input_shape, dropout_rate=0.2):
    inputs = Input(shape=input_shape)
    to_decoder = encoder(inputs)
    path = res_block(to_decoder[3], [1024, 1024], [(2, 2), (1, 1)], (8, 8))
    final_out, out_3, out_2, out_1 = decoder(path, from_encoder=to_decoder, dropout_rate=dropout_rate)
    final_out = Conv2D(filters=4, kernel_size=(1, 1), activation="softmax", name="final_output")(final_out)
    return Model(inputs=inputs, outputs=[final_out, out_3, out_2, out_1])


def dice_coef_class(y_true, y_pred, class_idx: int, smooth=1e-6):
    y_true_f = K.flatten(y_true[:, :, :, class_idx])
    y_pred_f = K.flatten(y_pred[:, :, :, class_idx])
    intersection = K.sum(y_true_f * y_pred_f)
    return (2.0 * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_score(y_true, y_pred):
    d0 = dice_coef_class(y_true, y_pred, 0)
    d1 = dice_coef_class(y_true, y_pred, 1)
    d2 = dice_coef_class(y_true, y_pred, 2)
    d3 = dice_coef_class(y_true, y_pred, 3)
    return (d0 + d1 + d2 + d3) / 4.0


def weighted_dice_score(y_true, y_pred):
    ds0 = dice_coef_class(y_true, y_pred, 0)
    ds1 = dice_coef_class(y_true, y_pred, 1)
    ds2 = dice_coef_class(y_true, y_pred, 2)
    ds3 = dice_coef_class(y_true, y_pred, 3)
    return (ds0 + (5.0 * ds1) + (2.0 * ds2) + (4.0 * ds3)) / 12.0


def weighted_dice_loss(y_true, y_pred):
    return 1.0 - weighted_dice_score(y_true, y_pred)


def weighted_log_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
    weights = tf.constant([1.0, 5.0, 2.0, 4.0], dtype=tf.float32)
    loss = y_true * tf.math.log(y_pred) * weights
    return tf.reduce_mean(-tf.reduce_sum(loss, axis=-1))


def gen_dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return weighted_dice_loss(y_true, y_pred) + weighted_log_loss(y_true, y_pred)


def save_training_plots(history_dict: dict, plots_dir: Path) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)

    if "loss" in history_dict:
        plt.figure(figsize=(10, 4))
        plt.plot(history_dict["loss"], label="Train Loss")
        if "val_loss" in history_dict:
            plt.plot(history_dict["val_loss"], label="Val Loss")
        plt.title("Training vs Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "loss_curve.png", dpi=150)
        plt.close()

    if "final_output_dice_score" in history_dict:
        plt.figure(figsize=(10, 4))
        plt.plot(history_dict["final_output_dice_score"], label="Train Dice (Final)")
        if "val_final_output_dice_score" in history_dict:
            plt.plot(history_dict["val_final_output_dice_score"], label="Val Dice (Final)")
        plt.title("Final Output Dice Score")
        plt.xlabel("Epoch")
        plt.ylabel("Dice")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "dice_curve_final_output.png", dpi=150)
        plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ARU-GD on BraTS HDF5 with H100-ready settings.")
    parser.add_argument("--train-h5", default="h5_dataset/train_data.h5")
    parser.add_argument("--valid-h5", default="h5_dataset/valid_data.h5")
    parser.add_argument("--test-h5", default="h5_dataset/test_data.h5")
    parser.add_argument("--output-dir", default="saved_models")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--dropout-rate", type=float, default=0.2)
    parser.add_argument("--plots-dir", default="saved_models/plots", help="Directory to save plots/images")
    parser.add_argument("--no-augment", action="store_true", help="Disable train-time augmentation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_gpu_and_precision()

    train_h5 = Path(args.train_h5)
    valid_h5 = Path(args.valid_h5)
    test_h5 = Path(args.test_h5)
    output_dir = Path(args.output_dir)
    plots_dir = Path(args.plots_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("Loading train/valid/test arrays in memory (single-phase training)...")
    x_train, y_train = load_h5_arrays(train_h5)
    x_valid, y_valid = load_h5_arrays(valid_h5)
    x_test, y_test = load_h5_arrays(test_h5)

    print(f"Train: X={x_train.shape}, Y={y_train.shape}")
    print(f"Valid: X={x_valid.shape}, Y={y_valid.shape}")
    print(f"Test : X={x_test.shape}, Y={y_test.shape}")

    train_ds = make_dataset(
        x_train,
        y_train,
        batch_size=args.batch_size,
        shuffle=True,
        augment=not args.no_augment,
    )
    valid_ds = make_dataset(
        x_valid,
        y_valid,
        batch_size=args.batch_size,
        shuffle=False,
        augment=False,
    )
    test_ds = make_dataset(
        x_test,
        y_test,
        batch_size=args.batch_size,
        shuffle=False,
        augment=False,
    )

    tf.keras.backend.clear_session()
    model = aru_gd((240, 240, 4), dropout_rate=args.dropout_rate)

    losses = {
        "final_output": gen_dice_loss,
        "out_3": gen_dice_loss,
        "out_2": gen_dice_loss,
        "out_1": gen_dice_loss,
    }
    loss_weights = {
        "final_output": 0.5,
        "out_3": 0.125,
        "out_2": 0.125,
        "out_1": 0.125,
    }

    base_optimizer = Adam(learning_rate=args.lr)
    optimizer = mixed_precision.LossScaleOptimizer(base_optimizer)

    model.compile(
        optimizer=optimizer,
        loss=losses,
        loss_weights=loss_weights,
        metrics={
            "final_output": dice_score,
            "out_3": dice_score,
            "out_2": dice_score,
            "out_1": dice_score,
        },
    )

    checkpoint_path = output_dir / "model_h100_best.keras"
    csv_log_path = output_dir / "train_log.csv"

    callbacks = [
        ModelCheckpoint(str(checkpoint_path), monitor="val_loss", save_best_only=True, mode="min", verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8, min_lr=1e-7, verbose=1),
        CSVLogger(str(csv_log_path)),
    ]

    history = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
    )

    final_model_path = output_dir / "model_h100_final.keras"
    model.save(str(final_model_path), include_optimizer=True)
    print(f"Final model saved to {final_model_path}")

    print("\nEvaluating on test dataset...")
    test_results = model.evaluate(test_ds, verbose=1, return_dict=True)
    for key, value in test_results.items():
        print(f"{key}: {value:.6f}")

    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)

    save_training_plots(history.history, plots_dir)
    print(f"Plots saved in: {plots_dir}")


if __name__ == "__main__":
    main()
