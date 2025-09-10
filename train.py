# -*- coding: UTF-8 -*-

"""
Train a neural network and save the parameters (weights) to an HDF5 file
"""

import math
import os
import pickle
import random
import traceback

# Set matplotlib backend to Agg for non-graphical environments
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from matplotlib.ticker import MaxNLocator
from tensorflow.keras import Model
from tensorflow.keras.layers import LSTM, Dense, Dropout

matplotlib.use("Agg")

from network import *
from utils import *

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)
random.seed(42)

"""
==== Concepts of some terms ====
# Batch size: The number of samples in one batch. The number of samples used in one iteration 
#             (Forward operation for calculating the loss function and BackPropagation operation 
#             for updating neural network parameters). Larger batch sizes require more memory.
# Iteration: One update of the network's weights. Each weight update requires a forward pass 
#            and a backward pass using a batch of data.
# Epoch: One complete pass through the entire training dataset.

# Example: If the training set has 1000 samples and the batch size is 10:
#          Training the entire dataset once requires 100 iterations and 1 epoch.
#          Usually, we train for multiple epochs.
"""


# Callback class for plotting and saving training losses
class PlotLosses(tf.keras.callbacks.Callback):
    def __init__(self, save_path="training_loss.png"):
        super(PlotLosses, self).__init__()
        self.save_path = save_path
        self.epochs = []
        self.losses = []

    def on_train_begin(self, logs={}):
        self.epochs = []
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.epochs.append(epoch + 1)
        self.losses.append(logs.get("loss"))

        print(f"PlotLosses: Saving figure to {self.save_path}")

        try:
            self._plot_and_save()
        except Exception as e:
            print(f"Error plotting losses: {str(e)}")
            traceback.print_exc()

    def _plot_and_save(self):
        plt.figure(figsize=(12, 8))

        # Plot raw training loss
        plt.plot(self.epochs, self.losses, "b-", linewidth=2, label="Training Loss")

        plt.title("Training Loss During Training", fontsize=18)
        plt.xlabel("Epoch", fontsize=16)
        plt.ylabel("Loss (Categorical Crossentropy)", fontsize=16)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(fontsize=14)

        # Use plain number format instead of scientific notation
        plt.ticklabel_format(style="plain", axis="y")

        ax = plt.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        plt.tight_layout()

        # Ensure save directory exists
        save_dir = os.path.dirname(self.save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        plt.savefig(self.save_path)
        plt.close()
        print(f"Successfully saved figure to {self.save_path}")


# Learning rate scheduler with slower decay rate
def lr_scheduler(epoch, lr):
    if epoch < 60:  # Maintain initial learning rate for more epochs
        return lr
    else:
        return lr * math.exp(-0.01)  # Smaller decay rate


class PerplexityLogger(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        # Training perplexity
        if "loss" in logs:
            logs["perplexity"] = float(np.exp(logs["loss"]))
        # Validation perplexity
        if "val_loss" in logs:
            logs["val_perplexity"] = float(np.exp(logs["val_loss"]))

        # Print nicely
        msg = f"Epoch {epoch+1} | "
        msg += (
            f"loss: {logs.get('loss'):.4f}, val_loss: {logs.get('val_loss'):.4f}, "
            if "val_loss" in logs
            else ""
        )
        msg += (
            f"acc: {logs.get('accuracy'):.4f}, val_acc: {logs.get('val_accuracy'):.4f}, "
            if "val_accuracy" in logs
            else ""
        )
        msg += (
            f"ppl: {logs.get('perplexity'):.4f}, val_ppl: {logs.get('val_perplexity'):.4f}"
            if "val_perplexity" in logs
            else ""
        )
        print(msg)


# Train the neural network
def train():
    # Create output directory if it doesn't exist
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {os.path.abspath(output_dir)}")

    # Load and prepare data
    notes = get_notes()
    print(f"Total notes/chords in dataset: {len(notes)}")

    # Get number of unique pitches
    pitch_names = sorted(set(item for item in notes))
    num_pitch = len(pitch_names)
    print(f"Number of unique notes/chords: {num_pitch}")

    # Create mappings between pitches and integers
    pitch_to_int = {pitch: num for num, pitch in enumerate(pitch_names)}
    int_to_pitch = {num: pitch for num, pitch in enumerate(pitch_names)}

    # Prepare sequences for training
    sequence_length = 100
    network_input, network_output = prepare_sequences(
        notes, pitch_to_int, sequence_length, num_pitch
    )

    print(f"Training data shape: {network_input.shape}")
    print(f"Output data shape: {network_output.shape}")

    # Build the model with regularization
    model = build_model(network_input.shape[1:], num_pitch)
    print(model.summary())

    # Define callbacks
    filepath = os.path.join(output_dir, "weights-{epoch:02d}-{loss:.4f}.keras")

    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath,
        monitor="loss",  # Monitor training loss
        verbose=1,
        save_best_only=True,
        mode="min",
    )

    plot_losses = PlotLosses(save_path=os.path.join(output_dir, "training_loss.png"))

    # Early stopping: stop when loss ≤ 0.05
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor="loss",
        baseline=0.04,  # Stop if loss falls below this value
        patience=10,  # Max epochs to wait after reaching baseline
        verbose=1,
        mode="min",
        restore_best_weights=True,
    )

    learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(
        lr_scheduler, verbose=1
    )

    callbacks_list = [
        checkpoint,
        plot_losses,
        early_stopping,
        learning_rate_scheduler,
        PerplexityLogger(),
    ]

    # Compile the model
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        metrics=["accuracy"],
    )

    # Split data into training and validation sets
    validation_split = 0.2
    print(
        f"Using {1-validation_split:.0%} for training, {validation_split:.0%} for validation"
    )

    # Train the model with increased epochs
    history = model.fit(
        network_input,
        network_output,
        epochs=200,  # Maximum training epochs
        batch_size=64,
        validation_split=validation_split,
        callbacks=callbacks_list,
        verbose=1,
    )

    # Determine best epoch based on training loss
    best_epoch = np.argmin(history.history["loss"]) + 1
    print(
        f"Best epoch: {best_epoch} with loss: {history.history['loss'][best_epoch-1]:.4f}"
    )
    print(
        f"Training accuracy at best epoch: {history.history['accuracy'][best_epoch-1]:.4f}"
    )

    # Plot and save the final loss curve
    epochs = range(1, len(history.history["loss"]) + 1)

    try:
        # --- Train vs Validation Loss ---
        loss_plot_path = os.path.join(output_dir, "train_val_loss.png")
        plt.figure(figsize=(12, 8))
        plt.plot(
            epochs, history.history["loss"], "b-", linewidth=2, label="Training Loss"
        )
        if "val_loss" in history.history:
            plt.plot(
                epochs,
                history.history["val_loss"],
                "r--",
                linewidth=2,
                label="Validation Loss",
            )
        plt.title("Training & Validation Loss", fontsize=18)
        plt.xlabel("Epoch", fontsize=16)
        plt.ylabel("Loss (Categorical Crossentropy)", fontsize=16)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig(loss_plot_path)
        plt.close()
        print(f"✅ Saved train/val loss plot to: {loss_plot_path}")

        # --- Categorical Cross-Entropy Loss ---
        cce_plot_path = os.path.join(output_dir, "categorical_crossentropy_loss.png")
        plt.figure(figsize=(12, 8))
        plt.plot(
            epochs,
            history.history["loss"],
            "b-",
            linewidth=2,
            label="Training CCE Loss",
        )
        if "val_loss" in history.history:
            plt.plot(
                epochs,
                history.history["val_loss"],
                "r--",
                linewidth=2,
                label="Validation CCE Loss",
            )
        plt.title("Categorical Crossentropy Loss Over Epochs", fontsize=18)
        plt.xlabel("Epoch", fontsize=16)
        plt.ylabel("CCE Loss", fontsize=16)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig(cce_plot_path)
        plt.close()
        print(f"✅ Saved categorical crossentropy loss plot to: {cce_plot_path}")

        # --- Train vs Validation Accuracy ---
        if "accuracy" in history.history:
            acc_plot_path = os.path.join(output_dir, "train_val_accuracy.png")
            plt.figure(figsize=(12, 8))
            plt.plot(
                epochs,
                history.history["accuracy"],
                "b-",
                linewidth=2,
                label="Training Accuracy",
            )
            if "val_accuracy" in history.history:
                plt.plot(
                    epochs,
                    history.history["val_accuracy"],
                    "r--",
                    linewidth=2,
                    label="Validation Accuracy",
                )
            plt.title("Training & Validation Accuracy", fontsize=18)
            plt.xlabel("Epoch", fontsize=16)
            plt.ylabel("Accuracy", fontsize=16)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend(fontsize=14)
            plt.tight_layout()
            plt.savefig(acc_plot_path)
            plt.close()
            print(f"✅ Saved train/val accuracy plot to: {acc_plot_path}")

        # --- Perplexity over epochs (exp(loss)) ---
        perp_plot_path = os.path.join(output_dir, "perplexity.png")
        plt.figure(figsize=(12, 8))
        plt.plot(
            epochs,
            np.exp(history.history["loss"]),
            "b-",
            linewidth=2,
            label="Training Perplexity",
        )
        if "val_loss" in history.history:
            plt.plot(
                epochs,
                np.exp(history.history["val_loss"]),
                "r--",
                linewidth=2,
                label="Validation Perplexity",
            )
        plt.title("Perplexity Over Epochs", fontsize=18)
        plt.xlabel("Epoch", fontsize=16)
        plt.ylabel("Perplexity (exp(loss))", fontsize=16)
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.legend(fontsize=14)
        plt.tight_layout()
        plt.savefig(perp_plot_path)
        plt.close()
        print(f"✅ Saved perplexity plot to: {perp_plot_path}")

        # --- Learning Rate Schedule (if logged) ---
        if "lr" in history.history:
            lr_plot_path = os.path.join(output_dir, "learning_rate.png")
            plt.figure(figsize=(12, 8))
            plt.plot(
                epochs, history.history["lr"], "g-", linewidth=2, label="Learning Rate"
            )
            plt.title("Learning Rate Schedule", fontsize=18)
            plt.xlabel("Epoch", fontsize=16)
            plt.ylabel("Learning Rate", fontsize=16)
            plt.grid(True, linestyle="--", alpha=0.7)
            plt.legend(fontsize=14)
            plt.yscale("log")
            plt.tight_layout()
            plt.savefig(lr_plot_path)
            plt.close()
            print(f"✅ Saved learning rate plot to: {lr_plot_path}")

    except Exception as e:
        print(f"⚠️ Error saving plots: {str(e)}")
        traceback.print_exc()

    # Save the final model and mappings
    model.save(os.path.join(output_dir, "final_model.h5"))
    print(f"Final model saved to: {os.path.join(output_dir, 'final_model.h5')}")

    # Save mappings for later use
    with open(os.path.join(output_dir, "pitch_to_int.pkl"), "wb") as f:
        pickle.dump(pitch_to_int, f)

    with open(os.path.join(output_dir, "int_to_pitch.pkl"), "wb") as f:
        pickle.dump(int_to_pitch, f)

    print("Training completed successfully!")


def prepare_sequences(notes, pitch_to_int, sequence_length, num_pitch):
    """
    Prepare sequences for neural network training
    """
    network_input = []
    network_output = []

    # Generate training sequences
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i : i + sequence_length]
        sequence_out = notes[i + sequence_length]

        network_input.append([pitch_to_int[char] for char in sequence_in])
        network_output.append(pitch_to_int[sequence_out])

    n_patterns = len(network_input)

    # Reshape and normalize input
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(num_pitch)

    # One-hot encode output
    network_output = tf.keras.utils.to_categorical(network_output)

    return network_input, network_output


def build_model(input_shape, num_pitch):
    """
    Build an LSTM-based model with residual connections and batch normalization
    """
    inputs = tf.keras.Input(shape=input_shape)

    x = LSTM(128, return_sequences=True)(inputs)
    x = Dropout(0.2)(x)
    x = Dense(128, activation="relu")(x)
    outputs = Dense(num_pitch, activation="softmax")(x)

    # Create model
    model = Model(inputs=inputs, outputs=outputs)

    return model


if __name__ == "__main__":
    train()


# # -*- coding: UTF-8 -*-

# """
# Train a neural network and save the parameters (weights) to an keras file
# """

# import os
# import random
# import traceback

# # Set matplotlib backend to Agg for non-graphical environments
# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
# import tensorflow as tf
# from matplotlib.ticker import MaxNLocator

# matplotlib.use("Agg")

# from network import *
# from utils import *

# # Set random seeds for reproducibility
# tf.random.set_seed(42)
# np.random.seed(42)
# random.seed(42)


# """
# ==== Concepts of some terms ====
# # Batch size: The number of samples in one batch. The number of samples used in one iteration
# #             (Forward operation for calculating the loss function and BackPropagation operation
# #             for updating neural network parameters). Larger batch sizes require more memory.
# # Iteration: One update of the network's weights. Each weight update requires a forward pass
# #            and a backward pass using a batch of data.
# # Epoch: One complete pass through the entire training dataset.
# """


# # Callback class for plotting and saving training losses
# class PlotLosses(tf.keras.callbacks.Callback):
#     def __init__(self, save_path="training_loss.png"):
#         super(PlotLosses, self).__init__()
#         self.save_path = save_path
#         self.epochs = []
#         self.losses = []
#         self.val_losses = []

#     def on_train_begin(self, logs={}):
#         self.epochs = []
#         self.losses = []
#         self.val_losses = []

#     def on_epoch_end(self, epoch, logs={}):
#         self.epochs.append(epoch + 1)
#         self.losses.append(logs.get("loss"))
#         self.val_losses.append(logs.get("val_loss"))  # <--- NEW

#         print(f"PlotLosses: Saving figure to {self.save_path}")

#         try:
#             self._plot_and_save()
#         except Exception as e:
#             print(f"Error plotting losses: {str(e)}")
#             traceback.print_exc()

#     def _plot_and_save(self):
#         plt.figure(figsize=(12, 8))

#         # Plot training + validation loss
#         plt.plot(self.epochs, self.losses, "b-", linewidth=2, label="Training Loss")
#         if any(v is not None for v in self.val_losses):
#             plt.plot(
#                 self.epochs,
#                 self.val_losses,
#                 "r--",
#                 linewidth=2,
#                 label="Validation Loss",
#             )

#         plt.title("Loss During Training", fontsize=18)
#         plt.xlabel("Epoch", fontsize=16)
#         plt.ylabel("Loss (Categorical Crossentropy)", fontsize=16)
#         plt.grid(True, linestyle="--", alpha=0.7)
#         plt.legend(fontsize=14)

#         plt.ticklabel_format(style="plain", axis="y")
#         ax = plt.gca()
#         ax.xaxis.set_major_locator(MaxNLocator(integer=True))

#         plt.tight_layout()

#         save_dir = os.path.dirname(self.save_path)
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)

#         plt.savefig(self.save_path)
#         plt.close()
#         print(f"Successfully saved figure to {self.save_path}")


# # Callback to log learning rate each epoch
# class LrLogger(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         logs = logs or {}
#         # Compatible with TF 2.11+ (Adam has .learning_rate instead of .lr)
#         lr = self.model.optimizer.learning_rate
#         if isinstance(lr, tf.keras.optimizers.schedules.LearningRateSchedule):
#             lr = lr(self.model.optimizer.iterations)  # evaluate schedule
#         logs["lr"] = float(tf.keras.backend.get_value(lr))


# # Train the neural network
# def train():
#     # Create output directory if it doesn't exist
#     output_dir = "output"
#     os.makedirs(output_dir, exist_ok=True)
#     print(f"Output directory: {os.path.abspath(output_dir)}")

#     # Load and prepare data
#     notes = get_notes()
#     print(f"Total notes/chords in dataset: {len(notes)}")

#     # Get number of unique pitches
#     pitch_names = sorted(set(item for item in notes))
#     num_pitch = len(pitch_names)
#     print(f"Number of unique notes/chords: {num_pitch}")

#     # Create mappings between pitches and integers
#     pitch_to_int = {pitch: num for num, pitch in enumerate(pitch_names)}
#     int_to_pitch = {num: pitch for num, pitch in enumerate(pitch_names)}

#     # Prepare sequences for training
#     sequence_length = 100
#     network_input, network_output = prepare_sequences(
#         notes, pitch_to_int, sequence_length, num_pitch
#     )

#     print(f"Training data shape: {network_input.shape}")
#     print(f"Output data shape: {network_output.shape}")

#     # Build the model with regularization
#     model = build_model(network_input.shape[1:], num_pitch)
#     print(model.summary())

#     # Define callbacks
#     filepath = os.path.join(output_dir, "weights-best.keras")

#     checkpoint = tf.keras.callbacks.ModelCheckpoint(
#         filepath,
#         monitor="val_loss",  # Monitor validation loss
#         verbose=1,
#         save_best_only=True,
#         mode="min",
#     )

#     plot_losses = PlotLosses(save_path=os.path.join(output_dir, "training_loss.png"))

#     # Early stopping: stop when val_loss stops improving
#     early_stopping = tf.keras.callbacks.EarlyStopping(
#         monitor="val_loss",
#         patience=50,
#         verbose=1,
#         mode="min",
#         restore_best_weights=True,
#     )

#     # Reduce LR on plateau
#     reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
#         monitor="val_loss", factor=0.5, patience=10, min_lr=1e-6, verbose=1
#     )

#     callbacks_list = [checkpoint, plot_losses, early_stopping, reduce_lr, LrLogger()]

#     # Compile the model
#     model.compile(
#         loss="categorical_crossentropy",
#         optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
#         metrics=["accuracy"],
#     )

#     # Split data into training and validation sets
#     validation_split = 0.2
#     print(
#         f"Using {1-validation_split:.0%} for training, {validation_split:.0%} for validation"
#     )

#     # Train the model
#     history = model.fit(
#         network_input,
#         network_output,
#         epochs=2000,
#         batch_size=64,
#         validation_split=validation_split,
#         callbacks=callbacks_list,
#         verbose=1,
#     )

#     # Determine best epoch based on training loss
#     best_epoch = np.argmin(history.history["loss"]) + 1
#     print(
#         f"Best epoch: {best_epoch} with loss: {history.history['loss'][best_epoch-1]:.4f}"
#     )
#     print(
#         f"Training accuracy at best epoch: {history.history['accuracy'][best_epoch-1]:.4f}"
#     )

#     # Plot and save the final loss curve
#     epochs = range(1, len(history.history["loss"]) + 1)
#     final_plot_path = os.path.join(output_dir, "training_loss_final.png")

#     try:
#         plt.figure(figsize=(14, 10))

#         # Loss plot
#         plt.subplot(2, 1, 1)
#         plt.plot(
#             epochs, history.history["loss"], "b-", linewidth=2, label="Training Loss"
#         )

#         plt.title("Training Loss During Training", fontsize=18)
#         plt.xlabel("Epoch", fontsize=16)
#         plt.ylabel("Loss", fontsize=16)
#         plt.grid(True, linestyle="--", alpha=0.7)
#         plt.legend(fontsize=14)

#         plt.ticklabel_format(style="plain", axis="y")
#         ax = plt.gca()
#         ax.xaxis.set_major_locator(MaxNLocator(integer=True))

#         # Annotation for best epoch
#         plt.axvline(x=best_epoch, color="g", linestyle="--", alpha=0.7)
#         plt.annotate(
#             f"Best Epoch: {best_epoch}",
#             xy=(best_epoch, history.history["loss"][best_epoch - 1]),
#             xytext=(best_epoch + 5, history.history["loss"][best_epoch - 1] + 0.05),
#             arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8),
#             fontsize=12,
#         )

#         # Learning rate plot (safe)
#         if "lr" in history.history:
#             plt.subplot(2, 1, 2)
#             plt.plot(
#                 epochs, history.history["lr"], "g-", linewidth=2, label="Learning Rate"
#             )
#             plt.title("Learning Rate Schedule", fontsize=18)
#             plt.xlabel("Epoch", fontsize=16)
#             plt.ylabel("Learning Rate", fontsize=16)
#             plt.grid(True, linestyle="--", alpha=0.7)
#             plt.legend(fontsize=14)
#             plt.yscale("log")
#         else:
#             print("⚠️ Skipping LR plot (no 'lr' in history).")

#         plt.tight_layout()

#         save_dir = os.path.dirname(final_plot_path)
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)

#         plt.savefig(final_plot_path)
#         plt.close()
#         print(f"Final training plot saved to: {final_plot_path}")
#     except Exception as e:
#         print(f"Error saving final training plot: {str(e)}")
#         traceback.print_exc()

#     # Save the final model and mappings
#     model.save(os.path.join(output_dir, "final_model.keras"))
#     print(f"Final model saved to: {os.path.join(output_dir, 'final_model.keras')}")

#     # Save mappings
#     import pickle

#     with open(os.path.join(output_dir, "pitch_to_int.pkl"), "wb") as f:
#         pickle.dump(pitch_to_int, f)

#     with open(os.path.join(output_dir, "int_to_pitch.pkl"), "wb") as f:
#         pickle.dump(int_to_pitch, f)

#     print("Training completed successfully!")


# def prepare_sequences(notes, pitch_to_int, sequence_length, num_pitch):
#     """
#     Prepare sequences for neural network training
#     """
#     network_input = []
#     network_output = []

#     for i in range(0, len(notes) - sequence_length, 1):
#         sequence_in = notes[i : i + sequence_length]
#         sequence_out = notes[i + sequence_length]

#         network_input.append([pitch_to_int[char] for char in sequence_in])
#         network_output.append(pitch_to_int[sequence_out])

#     n_patterns = len(network_input)

#     network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
#     network_input = network_input / float(num_pitch)

#     network_output = tf.keras.utils.to_categorical(network_output)

#     return network_input, network_output


# def build_model(input_shape, num_pitch):
#     """
#     Build an LSTM-based model with residual connections and batch normalization
#     """
#     inputs = tf.keras.Input(shape=input_shape)

#     # First LSTM block
#     x = tf.keras.layers.LSTM(512, return_sequences=True)(inputs)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.Dropout(0.3)(x)

#     # Second LSTM block with residual connection
#     residual = x
#     x = tf.keras.layers.LSTM(512, return_sequences=True)(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.Dropout(0.3)(x)
#     x = tf.keras.layers.Add()([x, residual])  # Residual connection

#     # Third LSTM block with residual connection
#     residual = x
#     x = tf.keras.layers.LSTM(512, return_sequences=False)(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.Dropout(0.3)(x)

#     # Dense layers with L2 regularization
#     x = tf.keras.layers.Dense(
#         1024, kernel_regularizer=tf.keras.regularizers.l2(0.0001)
#     )(x)
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.Activation("relu")(x)
#     x = tf.keras.layers.Dropout(0.3)(x)

#     x = tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.0001))(
#         x
#     )
#     x = tf.keras.layers.BatchNormalization()(x)
#     x = tf.keras.layers.Activation("relu")(x)
#     x = tf.keras.layers.Dropout(0.3)(x)

#     # Output layer
#     outputs = tf.keras.layers.Dense(num_pitch, activation="softmax")(x)

#     # Create model
#     model = tf.keras.Model(inputs=inputs, outputs=outputs)

#     return model


# if __name__ == "__main__":
#     train()
