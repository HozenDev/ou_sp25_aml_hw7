"""
Advanced Machine Learning, 2025, HW4

Author: Enzo B. Durel (enzo.durel@gmail.com)

Plotting script to analyse models results and performances.
"""

from chesapeake_loader4 import create_diffusion_dataset, create_single_dataset, create_diffusion_example
from diffusion_tools import compute_beta_alpha, convert_image
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

def prediction_example_from_a_model(args, model, fold, timestamps, num_examples=3, filename="predict_example.png"):
    """
    Plots a few examples of predictions from a model.

    :params args: Command-line arguments
    :params model: Model to use for prediction
    :params fold: Fold number
    :params num_examples: Number of examples to plot
    :params filename: Filename to save the plot
    """

    # Compute alpha schedule for HW7 diffusion
    _, alpha_np, _ = compute_beta_alpha(
        nsteps=args.nsteps,
        beta_start=0.0001,
        beta_end=0.02
    )

    _, dataset = create_diffusion_dataset(
        alpha=alpha,
        base_dir=args.dataset,
        patch_size=args.image_size[0],
        fold=fold,
        filt='*[012345678]',
        cache_dir=args.cache,
        repeat=True,
        batch_size=args.batch,
        prefetch=args.prefetch,
        num_parallel_calls=args.num_parallel_calls,
        time_sampling_exponent=args.time_sampling_exponent
    )

    nsteps = args.nsteps

    alpha_tf = tf.constant(alpha_np, dtype=tf.float32)
    patch_size = 256

    fig, axes = plt.subplots(nsteps + 1, num_examples, figsize=(4 * num_examples, 3 * (nsteps + 1)))

    for col in range(num_examples):
        try:
            I, L = next(iter(dataset))
        except StopIteration:
            break

        for t in range(nsteps):
            t_tensor = tf.constant([t], dtype=tf.int32)
            label, _, noised_image, true_noise = create_diffusion_example(I['image_input'], I['label_input'], patch_size, alpha_tf, t_tensor)

            # Predict noise
            model_inputs = {
                'label_input': tf.expand_dims(label, axis=0),
                'image_input': tf.expand_dims(noised_image, axis=0),
                'time_input': tf.expand_dims(t_tensor, axis=0)
            }
            predicted_noise = model.predict(model_inputs, verbose=0)[0]

            a_t = tf.gather(alpha_tf, t_tensor)[0]
            sqrt_a = tf.sqrt(a_t)
            sqrt_1_a = tf.sqrt(1 - a_t)

            # Denoise prediction
            pred_denoised = (noised_image - sqrt_1_a * predicted_noise) / sqrt_a
            pred_denoised = convert_image(pred_denoised.numpy())

            ax = axes[t, col] if num_examples > 1 else axes[t]
            ax.imshow(pred_denoised)
            if col == 0:
                ax.set_ylabel(f"Step {t}", fontsize=10)
            ax.axis('off')

        # At the end: show true denoised
        t_final = tf.constant([nsteps - 1], dtype=tf.int32)
        _, _, noised_image_final, true_noise_final = create_diffusion_example(I[..., :3], L, patch_size, alpha_tf, t_final)
        a_final = tf.gather(alpha_tf, t_final)[0]
        sqrt_a_final = tf.sqrt(a_final)
        sqrt_1_a_final = tf.sqrt(1 - a_final)
        true_denoised = (noised_image_final - sqrt_1_a_final * true_noise_final) / sqrt_a_final
        true_denoised = convert_image(true_denoised.numpy())

        ax = axes[-1, col] if num_examples > 1 else axes[-1]
        ax.imshow(true_denoised)
        if col == 0:
            ax.set_ylabel("True", fontsize=12)
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(filename)

def generate_figure2(model, alpha, sigma, patch_size=256, nsteps=50, seed=None, save_path='figure2.png'):
    """
    Generate and plot one sample diffusion sequence (Figure 2) using the trained model.
    
    Args:
        model: Trained Keras diffusion model.
        alpha: np.array of accumulated alpha values (length = nsteps).
        sigma: np.array of sigma values (length = nsteps).
        label_path: Path to a .npz file to extract image/label for conditioning.
        patch_size: Size of image patches.
        nsteps: Number of diffusion steps.
        seed: Optional seed for reproducibility.
        save_path: Where to save the resulting figure.
    """

    if seed is not None:
        tf.random.set_seed(seed)
        np.random.seed(seed)

    # Load one image-label pair using the provided loader
    ds_train, ds = create_diffusion_dataset(
        alpha=alpha,
        base_dir=args.dataset,
        patch_size=args.image_size[0],
        fold=0,
        filt='*[012345678]',
        cache_dir=args.cache,
        repeat=True,
        batch_size=args.batch,
        prefetch=args.prefetch,
        num_parallel_calls=args.num_parallel_calls,
        time_sampling_exponent=args.time_sampling_exponent
    )
    
    I = None
    L = None
    for I, L in ds.take(1):
        break

    # Use one-hot semantic label and normalize image to +/-1
    L_oh = tf.one_hot(L, 7)
    I = I[..., :3]  # RGB only
    I = tf.cast(I, tf.float32)
    L_oh = tf.cast(L_oh, tf.float32)

    # Start from pure noise
    current = tf.random.normal(shape=(1, patch_size, patch_size, 3))

    frames = []

    for t in reversed(range(nsteps)):
        t_tensor = tf.constant([[t]], dtype=tf.int32)

        # Predict noise to remove
        model_input = {
            'label_input': tf.expand_dims(L_oh, 0),
            'time_input': t_tensor,
            'image_input': current
        }
        predicted_noise = model(model_input)

        # Update current image using Equation 18.5-like logic
        a_t = alpha[t]
        b_t = 1 - a_t
        s_t = sigma[t]

        current = (1 / np.sqrt(1 - b_t)) * (current - ((1 - a_t) / np.sqrt(a_t)) * predicted_noise)

        if t > 0:
            current += s_t * tf.random.normal(shape=current.shape)

        # Store current step for plotting
        image_np = convert_image(current[0].numpy())  # shape: HWC
        frames.append(image_np)

    # === PLOT ===
    ncols = 5
    nrows = int(np.ceil(len(frames) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 3 * nrows))
    axes = axes.flat

    for idx in range(nrows * ncols):
        ax = axes[idx]
        if idx < len(frames):
            ax.imshow(frames[idx])
            ax.set_title(f"Step {nsteps - 1 - idx}")
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

    return frames

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
    net_dir = "./models/net_1/"

    #######################
    #     Load Models     #
    #######################
    
    models = []

    for i in range(nb_rotation):
        try:
            model = load_trained_model(net_dir, f"rot_0{i}")
            models.append(model)
        except Exception as e:
            print(f"Error loading shallow model: {e}")

    #######################
    #       Plotting      #
    #######################

    _, alpha, sigma = compute_beta_alpha(
        nsteps=args.nsteps,
        beta_start=0.0001,
        beta_end=0.02
    )

    # Example of prediction from a model
    prediction_example_from_a_model(args, models[0], 0, timestamps=10, num_examples=2, filename="figure_2.png")

    # generate_figure2(models[0], alpha, sigma, patch_size=256, nsteps=50, seed=None, save_path='figure2.png')


