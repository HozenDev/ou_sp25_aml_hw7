"""
Advanced Machine Learning, 2025, HW4

Author: Enzo B. Durel (enzo.durel@gmail.com)

Plotting script to analyse models results and performances.
"""

from chesapeake_loader4 import create_diffusion_dataset, create_single_dataset, create_diffusion_example
from diffusion_tools import compute_beta_alpha, convert_image, compute_beta_alpha2
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
import keras

from sklearn.metrics import confusion_matrix
from parser import check_args, create_parser
from diffusion_tools import PositionEncoder
    
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
    model = tf.keras.models.load_model(model_path, custom_objects={'PositionEncoder': PositionEncoder,
                                                                   'ExpandDims': keras.src.ops.numpy.ExpandDims,
                                                                   'Tile': keras.src.ops.numpy.Tile,
                                                                   'mse': 'mse'})

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

def predict_example(args, model):
    # Load validation data
    ds = create_single_dataset(base_dir=args.dataset,
                               full_sat=False,
                               partition='valid',
                               patch_size=256,
                               fold=0,
                               cache_path=None,
                               repeat=True,
                               batch_size=8,
                               prefetch=2,
                               num_parallel_calls=4)
    
    I, L = next(iter(ds))
    print(I.numpy().shape, L.numpy().shape)

    timesteps = 10
    beta, alpha, gamma = compute_beta_alpha2(timesteps, 0.0001, 0.02, 0, 0.1)

    # Start from random noise
    Z = np.random.normal(loc=0, scale=1.0, size=(args.batch, 256, 256, 3)).astype(np.float32)
    Zs = []

    # Reversed diffusion steps
    for ts in reversed(range(timesteps)):
        t_tensor = tf.constant(ts * np.ones((args.batch, 1)), dtype=tf.int32)

        # Predict noise
        delta = model.predict({
            'image_input': tf.convert_to_tensor(Z),
            'label_input': L,
            'time_input': t_tensor
        }, verbose=0)

        b, a, g = beta[ts], alpha[ts], gamma[ts]
        Z = Z / np.sqrt(1 - b) - delta * b / (np.sqrt(1 - a) * np.sqrt(1 - b))

        if ts > 0:
            Z += g * np.random.normal(size=Z.shape).astype(np.float32)

        Zs.append(Z.copy())

    # Visualization
    i = 0  # show the first example
    cols = min(10, len(Zs))
    rows = (len(Zs) + cols - 1) // cols + 2  # +2 for label and noised input

    fig, axs = plt.subplots(rows, cols, figsize=(20, 2 * rows))

    axs[0, 0].imshow(np.argmax(L[i], axis=-1), vmin=0, vmax=6)
    axs[0, 0].set_title("Label")
    axs[0, 1].imshow(convert_image(Z[i]))
    axs[0, 1].set_title("Input")

    for j, z in enumerate(Zs):
        row, col = divmod(j, cols)
        axs[row + 1, col].imshow(convert_image(z[i]))
        axs[row + 1, col].set_title(f"Step {timesteps - 1 - j}")
        axs[row + 1, col].axis('off')

    for ax in axs.flatten():
        ax.set_xticks([]), ax.set_yticks([])

    plt.tight_layout()
    plt.savefig("figures/figure_2.png")
    plt.close()

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

    nsteps = 10

    alpha_tf = tf.constant(alpha_np, dtype=tf.float32)
    patch_size = 256

    fig, axes = plt.subplots(nsteps + 1, num_examples, figsize=(4 * num_examples, 3 * (nsteps + 1)))

    for col in range(num_examples):
        try:
            I, L = next(dataset.as_numpy_iterator())
        except StopIteration:
            break

        label_tensor = I['label_input']
        time_tensor = I['time_input']
        image_tensor = I['image_input']
        noise = L;
        
        for t in range(nsteps):
            t_tensor = tf.constant([t], dtype=tf.int32)
            _, _, noised_image, _ = create_diffusion_example(image_tensor, label_tensor, patch_size, alpha_tf, t_tensor)
            time_input = t * np.ones(shape=(I['image_input'].shape[0], 1))

            # Predict noise
            model_inputs = {
                'label_input': label_tensor,
                'image_input': noised_image, 
                'time_input': time_input,
            }

            # for k, v in model_inputs.items():
            #    print(f"{k}:", v.shape)
            
            predicted_noise = model.predict(x=model_inputs, verbose=0)[0]

            a_t = tf.gather(alpha_tf, t_tensor)[0]
            sqrt_a = tf.sqrt(a_t)
            sqrt_1_a = tf.sqrt(1 - a_t)

            # Denoise prediction
            pred_denoised = (noised_image - sqrt_1_a * predicted_noise) / sqrt_a
            pred_denoised = convert_image(pred_denoised.numpy())

            ax = axes[t, col] if num_examples > 1 else axes[t]
            ax.imshow(pred_denoised[col,:,:,:])
            if col == 0:
                ax.set_ylabel(f"Step {t}", fontsize=10)
            ax.axis('off')

        # At the end: show true denoised
        t_final = tf.constant([nsteps - 1], dtype=tf.int32)
        _, _, noised_image_final, true_noise_final = create_diffusion_example(image_tensor, L, patch_size, alpha_tf, t_final)
        a_final = tf.gather(alpha_tf, t_final)[0]
        sqrt_a_final = tf.sqrt(a_final)
        sqrt_1_a_final = tf.sqrt(1 - a_final)
        true_denoised = (noised_image_final - sqrt_1_a_final * true_noise_final) / sqrt_a_final
        true_denoised = convert_image(true_denoised.numpy())

        ax = axes[-1, col] if num_examples > 1 else axes[-1]
        ax.imshow(true_denoised[col,:,:,:])
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
    # prediction_example_from_a_model(args, models[0], 0, timestamps=10, num_examples=2, filename="figure_2.png")
    predict_example(args, models[0])

    # generate_figure2(models[0], alpha, sigma, patch_size=256, nsteps=50, seed=None, save_path='figure2.png')


