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
    model = keras.models.load_model(model_path, custom_objects={'PositionEncoder': PositionEncoder,
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
    timesteps = 10
    beta, alpha, gamma = compute_beta_alpha2(timesteps, 0.0001, 0.02, 0, 0.1)

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

    # Start from random noise
    Z = np.random.normal(loc=0, scale=1.0, size=(args.batch, 256, 256, 3)).astype(np.float32)
    Zs = []

    label_input = tf.cast(L, tf.int32)
    label_input = tf.one_hot(label_input, depth=7)

    # Reversed diffusion steps
    for ts in reversed(range(timesteps)):
        t_tensor = tf.constant(ts * np.ones((args.batch, 1)), dtype=tf.int32)

        # Predict noise
        delta = model.predict({
            'image_input': tf.convert_to_tensor(Z),
            'label_input': label_input,
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

    axs[0, 0].imshow(np.argmax(label_input[i], axis=-1), vmin=0, vmax=6)
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

def generate_figure2(model,
                     dataset_dir: str,
                     save_dir: str = '.',
                     nsteps: int = 50,
                     beta_start: float = 0.0001,
                     beta_end: float = 0.02,
                     fold: int = 0,
                     num_examples: int = 2,
                     patch_size: int = 256):
    """
    Generate Figure 2: Denoising sequence with original and label images.

    :param model_path: Path to the saved .keras model.
    :param dataset_dir: Base directory to Chesapeake dataset.
    :param save_dir: Directory where the figures will be saved.
    :param nsteps: Number of diffusion steps.
    :param beta_start: Starting value for beta.
    :param beta_end: Ending value for beta.
    :param fold: Fold index to load validation examples from.
    :param num_examples: Number of examples to visualize.
    :param patch_size: Height/width of input patches.
    """

    _, alpha, sigma = compute_beta_alpha(nsteps, beta_start, beta_end)
    alpha_tf = tf.constant(alpha, dtype=tf.float32)

    ds = create_single_dataset(
        base_dir=dataset_dir,
        full_sat=True,  # Use full_sat=True to get original image (including RGB)
        patch_size=patch_size,
        partition='train',
        fold=fold,
        filt='*9',
        cache_path=None,
        repeat=False,
        shuffle=None,
        batch_size=1,
        prefetch=1,
        num_parallel_calls=1
    )

    examples = list(ds.take(num_examples))

    def denoise(label_img):
        """Perform reverse diffusion on a label image."""
        label_onehot = tf.one_hot(label_img, 7)
        x = tf.random.normal(shape=(patch_size, patch_size, 3))
        images = []

        for t in reversed(range(nsteps)):
            t_tensor = tf.convert_to_tensor([[t]])
            pe = PositionEncoder(max_steps=nsteps, max_dims=30)
            t_embed = pe(t_tensor)
            t_embed_image = keras.ops.expand_dims(keras.ops.expand_dims(t_embed, axis=1), axis=1)
            t_embed_image = keras.ops.tile(t_embed_image, [1, patch_size, patch_size, 1])

            inputs = {
                'label_input': keras.ops.expand_dims(label_onehot, axis=0),
                'time_input': t_tensor,
                'image_input': keras.ops.expand_dims(x, axis=0)
            }

            predicted_noise = model(inputs, training=False)[0]
            a_t = alpha_tf[t].numpy()
            s_t = sigma[t]

            x = (x - (1 - a_t) ** 0.5 * predicted_noise) / (a_t ** 0.5)
            x = x + s_t * tf.random.normal(shape=x.shape)

            images.append(convert_image(x.numpy()))

        return images

    def label_to_rgb(label_img):
        """Map class indices to RGB for visualization."""
        colormap = np.array([
            [0, 0, 0],           # 0: No class (black)
            [0, 0, 255],         # 1: Water (blue)
            [34, 139, 34],       # 2: Forest (green)
            [210, 180, 140],     # 3: Field (tan)
            [165, 42, 42],       # 4: Barren (brown)
            [128, 128, 128],     # 5: Impervious other (gray)
            [255, 255, 0]        # 6: Road (yellow)
        ], dtype=np.uint8)
        return colormap[label_img]

    for i, (img, label) in enumerate(examples):
        rgb_image = img[0, :, :, :3].numpy() 
        label_img = label[0].numpy()
        label_rgb = label_to_rgb(label_img)

        denoised_images = denoise(label_img)

        fig, axs = plt.subplots(1, 10, figsize=(22, 5))
        axs = axs.flatten()

        axs[0].imshow(rgb_image)
        axs[0].set_title("Original")
        axs[0].axis('off')

        axs[1].imshow(label_rgb)
        axs[1].set_title("Label")
        axs[1].axis('off')

        nb_images = 8
        each_step = max(nsteps // nb_images, 1)
        sampled_images = denoised_images[::each_step][:nb_images]
        for j, im in enumerate(sampled_images):
            axs[2 + j].imshow(im)
            axs[2 + j].set_title(f"t={nsteps - 1 - j * each_step}")
            axs[2 + j].axis('off')

        plt.suptitle(f"Figure 2 - Sample {i+1}: Denoising Sequence")
        plt.tight_layout()
        out_path = os.path.join(save_dir, f"figure2_sample{i+1}.png")
        plt.savefig(out_path)
        plt.show()
        print(f"Saved: {out_path}")



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
    # predict_example(args, models[0])
    generate_figure2(
        model=models[0],
        dataset_dir=args.dataset,
        save_dir="./figures",
        nsteps=15,
        beta_start=0.0001,
        beta_end=0.02,
        fold=0,
        num_examples=2,
        patch_size=256
    )

    # generate_figure2(models[0], alpha, sigma, patch_size=256, nsteps=50, seed=None, save_path='figure2.png')


