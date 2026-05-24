"""Train EfficientNetB0 transfer learning model."""

import argparse
import sys

import pandas as pd
from tensorflow import keras

from config import (
    BATCH_SIZE,
    EFFICIENTNET_BEST_MODEL_PATH,
    EFFICIENTNET_CURVES_FIG,
    EFFICIENTNET_EPOCHS,
    EFFICIENTNET_FINAL_MODEL_PATH,
    EFFICIENTNET_HISTORY_CSV,
    FINE_TUNE_EPOCHS,
    FINE_TUNE_LEARNING_RATE,
    TRANSFER_LEARNING_RATE,
    TRAIN_SPLIT_CSV,
    VAL_SPLIT_CSV,
    ensure_directories,
)
from utils.data_utils import (
    create_balanced_tf_dataset,
    create_tf_dataset,
    dataframe_memory_note,
    load_split_dataframe,
)
from utils.model_utils import (
    build_callbacks,
    build_efficientnetb0,
    compile_model,
    compute_balanced_class_weights,
    history_to_dataframe,
    plot_training_curves,
    print_class_weights,
    save_history,
    set_global_determinism,
)


def unfreeze_last_efficientnet_layers(model: keras.Model, num_layers: int) -> None:
    """Unfreeze the last EfficientNet layers while keeping BatchNorm frozen."""
    efficientnet_layers = [
        layer
        for layer in model.layers
        if layer.name.startswith(("stem_", "block", "top_"))
    ]

    if not efficientnet_layers:
        print("Could not identify EfficientNet backbone layers; skipping fine-tuning.")
        return

    for layer in efficientnet_layers:
        layer.trainable = False

    for layer in efficientnet_layers[-num_layers:]:
        if not isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = True

    trainable_count = sum(1 for layer in model.layers if layer.trainable)
    print(f"Fine-tuning enabled. Trainable layers in full model: {trainable_count}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train EfficientNetB0 transfer model.")
    parser.add_argument("--epochs", type=int, default=EFFICIENTNET_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=TRANSFER_LEARNING_RATE)
    parser.add_argument(
        "--weights",
        choices=["imagenet", "none"],
        default="imagenet",
        help="Use ImageNet weights by default. Use 'none' only if download is unavailable.",
    )
    parser.add_argument(
        "--fine-tune",
        action="store_true",
        help="Optionally unfreeze the last EfficientNet layers after initial training.",
    )
    parser.add_argument("--fine-tune-epochs", type=int, default=FINE_TUNE_EPOCHS)
    parser.add_argument("--fine-tune-layers", type=int, default=20)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=FINE_TUNE_LEARNING_RATE)
    parser.add_argument("--no-augmentation", action="store_true")
    parser.add_argument(
        "--balanced-sampling",
        action="store_true",
        help="Use equal class sampling instead of relying only on class weights.",
    )
    parser.add_argument(
        "--focal-loss",
        action="store_true",
        help="Use sparse focal loss to focus training on harder examples.",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=4,
        help="EarlyStopping patience.",
    )
    args = parser.parse_args()

    ensure_directories()
    set_global_determinism()

    print("Training EfficientNetB0 transfer learning model...\n")
    try:
        train_df = load_split_dataframe(TRAIN_SPLIT_CSV)
        val_df = load_split_dataframe(VAL_SPLIT_CSV)
    except FileNotFoundError as exc:
        print(f"ERROR:\n{exc}")
        sys.exit(1)

    print(dataframe_memory_note(train_df))
    if args.balanced_sampling:
        train_ds = create_balanced_tf_dataset(train_df, batch_size=args.batch_size)
        steps_per_epoch = max(1, len(train_df) // args.batch_size)
        class_weight_arg = None
        print("Using balanced class sampling for training batches.")
    else:
        train_ds = create_tf_dataset(train_df, batch_size=args.batch_size, shuffle=True)
        steps_per_epoch = None
        class_weight_arg = compute_balanced_class_weights(train_df["diagnosis"].to_numpy())

    val_ds = create_tf_dataset(val_df, batch_size=args.batch_size, shuffle=False)

    class_weights = compute_balanced_class_weights(train_df["diagnosis"].to_numpy())
    print_class_weights(class_weights)

    weights = None if args.weights == "none" else "imagenet"
    try:
        model = build_efficientnetb0(
            weights=weights,
            train_base=False,
            use_augmentation=not args.no_augmentation,
        )
    except Exception as exc:
        print(
            "ERROR: EfficientNetB0 could not be created. If this happened while "
            "downloading ImageNet weights, check your internet connection or rerun "
            "with --weights none for a no-pretraining fallback."
        )
        raise exc

    loss_name = "sparse_focal_loss" if args.focal_loss else "sparse_categorical_crossentropy"
    model = compile_model(model, learning_rate=args.learning_rate, loss=loss_name)
    model.summary()

    callbacks = build_callbacks(EFFICIENTNET_BEST_MODEL_PATH, patience=args.patience)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        class_weight=class_weight_arg,
        callbacks=callbacks,
        verbose=1,
    )

    history_frames = [history_to_dataframe(history, phase="frozen_backbone")]

    if args.fine_tune:
        print("\nStarting optional fine-tuning phase...")
        unfreeze_last_efficientnet_layers(model, args.fine_tune_layers)
        model = compile_model(
            model,
            learning_rate=args.fine_tune_learning_rate,
            loss=loss_name,
        )
        fine_tune_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            steps_per_epoch=steps_per_epoch,
            class_weight=class_weight_arg,
            callbacks=callbacks,
            verbose=1,
        )
        history_frames.append(history_to_dataframe(fine_tune_history, phase="fine_tuning"))
    else:
        print("\nFine-tuning is disabled by default to keep computational cost conservative.")

    history_df = pd.concat(history_frames, ignore_index=True)
    save_history(history_df, EFFICIENTNET_HISTORY_CSV)
    plot_training_curves(history_df, EFFICIENTNET_CURVES_FIG, title="EfficientNetB0 Training")

    model.save(EFFICIENTNET_FINAL_MODEL_PATH)
    print(f"\nBest EfficientNetB0 model saved to: {EFFICIENTNET_BEST_MODEL_PATH}")
    print(f"Final EfficientNetB0 model saved to: {EFFICIENTNET_FINAL_MODEL_PATH}")
    print(f"Training history saved to: {EFFICIENTNET_HISTORY_CSV}")
    print(f"Training curves saved to: {EFFICIENTNET_CURVES_FIG}")
    print("EfficientNetB0 training completed.")


if __name__ == "__main__":
    main()
