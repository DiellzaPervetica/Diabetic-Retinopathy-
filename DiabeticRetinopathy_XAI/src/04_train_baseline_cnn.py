"""Train the lightweight baseline CNN model."""

import argparse
import sys

from config import (
    BASELINE_CURVES_FIG,
    BASELINE_EPOCHS,
    BASELINE_HISTORY_CSV,
    BASELINE_MODEL_PATH,
    BATCH_SIZE,
    LEARNING_RATE,
    TRAIN_SPLIT_CSV,
    VAL_SPLIT_CSV,
    ensure_directories,
)
from utils.data_utils import create_tf_dataset, dataframe_memory_note, load_split_dataframe
from utils.model_utils import (
    build_baseline_cnn,
    build_callbacks,
    compile_model,
    compute_balanced_class_weights,
    history_to_dataframe,
    plot_training_curves,
    print_class_weights,
    save_history,
    set_global_determinism,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train baseline CNN.")
    parser.add_argument("--epochs", type=int, default=BASELINE_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE)
    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Disable lightweight training augmentation.",
    )
    args = parser.parse_args()

    ensure_directories()
    set_global_determinism()

    print("Training baseline CNN...\n")
    try:
        train_df = load_split_dataframe(TRAIN_SPLIT_CSV)
        val_df = load_split_dataframe(VAL_SPLIT_CSV)
    except FileNotFoundError as exc:
        print(f"ERROR:\n{exc}")
        sys.exit(1)

    print(dataframe_memory_note(train_df))
    train_ds = create_tf_dataset(train_df, batch_size=args.batch_size, shuffle=True)
    val_ds = create_tf_dataset(val_df, batch_size=args.batch_size, shuffle=False)

    class_weights = compute_balanced_class_weights(train_df["diagnosis"].to_numpy())
    print_class_weights(class_weights)

    model = build_baseline_cnn(use_augmentation=not args.no_augmentation)
    model = compile_model(model, learning_rate=args.learning_rate)
    model.summary()

    callbacks = build_callbacks(BASELINE_MODEL_PATH)
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1,
    )

    history_df = history_to_dataframe(history, phase="baseline")
    save_history(history_df, BASELINE_HISTORY_CSV)
    plot_training_curves(history_df, BASELINE_CURVES_FIG, title="Baseline CNN Training")

    print(f"\nBest baseline model saved to: {BASELINE_MODEL_PATH}")
    print(f"Training history saved to: {BASELINE_HISTORY_CSV}")
    print(f"Training curves saved to: {BASELINE_CURVES_FIG}")
    print("Baseline CNN training completed.")


if __name__ == "__main__":
    main()

