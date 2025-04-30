"""
Advanced Machine Learning, 2025, HW4

Author: Enzo B. Durel (enzo.durel@gmail.com)

Plotting script to analyse models results and performances.
"""

from chesapeake_loader4 import create_single_dataset
import tensorflow as tf

# Gpus initialization
gpus = tf.config.experimental.list_physical_devices('GPU')
n_visible_devices = len(gpus)
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"Enabled memory growth for {len(gpus)} GPU(s).")
    except RuntimeError as e:
        print(f"Error setting memory growth: {e}")

# Set threading parallelism
import os
cpus_per_task = int(os.environ.get("SLURM_CPUS_PER_TASK", 0))
if cpus_per_task > 1:
    tf.config.threading.set_intra_op_parallelism_threads(cpus_per_task // 2)
    tf.config.threading.set_inter_op_parallelism_threads(cpus_per_task // 2)

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from sklearn.metrics import confusion_matrix
from parser import check_args, create_parser
    
#########################################
#             Load Results              #
#########################################

def load_trained_model(model_dir, substring_name):
    """
    Load a trained models
    """
    model_files = [f for f in os.listdir(model_dir) if substring_name in f and f.endswith(".keras")]

    if not model_files:
        raise ValueError(f"No model found in {model_dir} matching {substring_name}")

    model_path = os.path.join(model_dir, model_files[0])
    model = tf.keras.models.load_model(model_path)

    return model

def load_results_iter(results_dir):
    """
    Generator to load model results from a directory.
    Reduce memory usage compared to load results method.
    """
    files = [os.path.join(results_dir, f) for f in os.listdir(results_dir) if f.endswith(".pkl")]
    
    for filename in files:
        with open(filename, "rb") as fp:
            data = pickle.load(fp)
            yield data
            

def load_results(results_dir):
    """
    Load model results from a directory
    """
    results = []
    files = []
    for r_dir in results_dir:
        files.extend([os.path.join(r_dir, f) for f in os.listdir(r_dir) if f.endswith(".pkl")])

    for filename in files:
        with open(filename, "rb") as fp:
            data = pickle.load(fp)
            results.append(data)

    return results

#########################################
#             Plot Methods              #
#########################################

def prediction_example_from_a_model(args, model, fold, num_examples=10, filename="predict_example.png"):
    """
    Plots a few examples of predictions from a model.

    :params args: Command-line arguments
    :params model: Model to use for prediction
    :params fold: Fold number
    :params num_examples: Number of examples to plot
    :params filename: Filename to save the plot
    """

    # Use of single dataset to avoid loading other datasets
    test_ds = create_single_dataset(base_dir=args.dataset,
                                    full_sat=True,
                                    patch_size=None,
                                    partition='valid',
                                    fold=fold,
                                    filt='*',
                                    cache_path='',
                                    repeat=False,
                                    shuffle=None,
                                    batch_size=args.batch,
                                    prefetch=args.prefetch,
                                    num_parallel_calls=args.num_parallel_calls)

    for batch in test_ds.take(1):
        inputs, true_labels = batch

        preds = model.predict(inputs)
        pred_labels = np.argmax(preds, axis=-1) # Get the class with the high prob

        _, axes = plt.subplots(num_examples, 3, figsize=(10, num_examples * 3))
        for i in range(num_examples):
            rgb = inputs[i, :, :, :3].numpy() # Raw image
            gt = true_labels[i].numpy() # Good semantic labelled image
            pred = pred_labels[i] # Predict image

            axes[i, 0].imshow(rgb)
            axes[i, 0].set_title("Input RGB")
            axes[i, 0].axis('off')

            axes[i, 1].imshow(gt, vmin=0, vmax=6)
            axes[i, 1].set_title("Ground Truth")
            axes[i, 1].axis('off')

            axes[i, 2].imshow(pred, vmin=0, vmax=6)
            axes[i, 2].set_title("Prediction")
            axes[i, 2].axis('off')

    plt.tight_layout()
    plt.savefig(filename)
    

def plot_combined_confusion_matrix(args, models, num_classes, class_names, title="Confusion Matrix", filename="figure_4.png"):
    """
    Computes and plots a combined confusion matrix from multiple model rotations.

    :params args: Command-line arguments
    :params models: List of trained models
    :params num_classes: Number of classes
    :params class_names: List of class names
    :params title: Title of the plot
    :params filename: Filename to save the plot
    """
    all_y_true = []
    all_y_pred = []

    for i, model in enumerate(models):

        test_ds = create_single_dataset(base_dir=args.dataset,
                                        full_sat=True,
                                        patch_size=None,
                                        partition='valid',
                                        fold=i,
                                        filt='*',
                                        cache_path='',
                                        repeat=False,
                                        shuffle=None,
                                        batch_size=args.batch,
                                        prefetch=args.prefetch,
                                        num_parallel_calls=args.num_parallel_calls)
        
        for x_batch, y_batch in test_ds:
            preds = model.predict(x_batch)
            y_pred = np.argmax(preds, axis=-1)  
            y_true = y_batch.numpy()            

            all_y_pred.append(y_pred.flatten())
            all_y_true.append(y_true.flatten())
            
    y_true_flat = np.concatenate(all_y_true)
    y_pred_flat = np.concatenate(all_y_pred)

    labels = list(range(num_classes))
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=labels)

    _, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, cmap="Blues")

    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.colorbar(im, ax=ax)
    plt.xticks(np.arange(num_classes), class_names, rotation=45)
    plt.yticks(np.arange(num_classes), class_names)

    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="black")

    plt.tight_layout()
    plt.savefig(filename)


def plot_test_accuracy_scatter(shallow_results_dir, deep_results_dir, filename="test_acc.png"):
    """
    Plots a scatter plot comparing test accuracies of shallow and deep models.

    :params shallow_results_dir: Directory containing shallow model results
    :params deep_results_dir: Directory containing deep model results
    :params filename: Filename to save the plot
    """
    # shallow_accuracies, deep_accuracies = [], []
    # rotations = list(range(min(len(shallow_results), len(deep_results))))
    rotations = list(range(5))
    
    # Define color map: distinct color for each rotation
    colors = plt.cm.get_cmap("tab10", len(rotations))
    
    shallow_results_iter = load_results_iter(shallow_results_dir)
    deep_results_iter = load_results_iter(deep_results_dir)
    
    # Scatter plot
    plt.figure(figsize=(7,7))
    for i in range(len(rotations)):

        shallow_accuracy = next(shallow_results_iter)["predict_testing_eval"][1]
        deep_accuracy = next(deep_results_iter)["predict_testing_eval"][1]
        
        plt.scatter(shallow_accuracy, deep_accuracy, color=colors(i), label=f"Rot {rotations[i]}")
        plt.text(shallow_accuracy, deep_accuracy, f"{rotations[i]}", fontsize=10, ha='right', va='bottom')

    # Diagonal line    
    plt.plot([0.85, 0.96], [0.85, 0.96], 'k--', lw=2)  # y = x line

    plt.xlabel("Shallow Model Accuracy")
    plt.ylabel("Deep Model Accuracy")
    plt.title("Test Accuracy: Deep vs. Shallow")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)


#########################################
#            Main Function              #
#########################################
    
if __name__ == "__main__":
    # Parse command-line arguments
    parser = create_parser()
    args = parser.parse_args()
    check_args(args)

    # Dataset metadata
    nb_rotation = 5
    num_classes = 7
    class_names = ["No Class", "Water", "Forest", "Low Veg", "Barren", "Impervious", "Road"]
    deep_dir = "./models/deep_1/"
    shallow_dir = "./models/shallow_1/"

    #######################
    #     Load Models     #
    #######################
    
    shallow_models = []
    deep_models = []

    for i in range(nb_rotation):
        try:
            shallow_model = load_trained_model(shallow_dir, f"rot_0{i}")
            shallow_models.append(shallow_model)
        except Exception as e:
            print(f"Error loading shallow model: {e}")
        
        try:
            deep_model = load_trained_model(deep_dir, f"rot_0{i}")
            deep_models.append(deep_model)
        except Exception as e:
            print(f"Error loading deep model: {e}")

    #######################
    #       Plotting      #
    #######################
            
    # Example of prediction from a model
    prediction_example_from_a_model(args, deep_models[2], 2, num_examples=10, filename="figure_5b.png")
    prediction_example_from_a_model(args, shallow_models[0], 0, num_examples=10, filename="figure_5a.png")

    # Confusion matrix
    plot_combined_confusion_matrix(args=args, models=shallow_models, class_names=class_names, title="Shallow Model Confusion Matrix", filename="figure_3a.png", num_classes=num_classes)
    plot_combined_confusion_matrix(args=args, models=deep_models, class_names=class_names, title="Deep Model Confusion Matrix", filename="figure_3b.png", num_classes=num_classes)

    # Test accuracy scatter plot
    plot_test_accuracy_scatter(shallow_dir, deep_dir, filename="figure_4.png")

