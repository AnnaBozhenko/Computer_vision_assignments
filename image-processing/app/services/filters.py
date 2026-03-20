import numpy as np 
import cv2 
from scipy.ndimage import binary_dilation

def generate_gaussian_kernel(kernel_size, std):
    ax = np.arange(1 - kernel_size//2-1, kernel_size//2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2)/(2*std))
    kernel = kernel / kernel.sum()
    return kernel

def convert_2d_to_1d(image, window_size=3):
    """image: (Height, Width)
    outputs array of flattened windows across each image's pixel, with padded on edge pixels to make them centered.
    Note that flattened window is along row's axis (0)
    
    e.g. 
    with 3x3 window each pixel contains 9 sized array, so output is (9, Height * Width)"""
    pad = window_size // 2
    pads = tuple([(0, 0)]*(image.ndim == 3) + [(pad, pad), (pad, pad)])
    padded_img = np.pad(image, pads)
    i0 = np.repeat(np.arange(window_size), window_size)
    j0 = np.tile(np.arange(window_size), window_size)

    img_h, img_w = image.shape[-2:]
    i1 = np.repeat(np.arange(img_h), img_w)
    j1 = np.tile(np.arange(img_w), img_h)
    i=i0.reshape(-1,1)+i1.reshape(1,-1)
    j=j0.reshape(-1,1)+j1.reshape(1,-1)
    patches=padded_img[:, i,j] if image.ndim ==3 else padded_img[i,j]
    return patches


def convolve2d(img: np.ndarray, kernel):
    """img's shape: (channels, height, width)
    convolution with stride = 1"""
    img_h, img_w, channels_n = img.shape
    img_dims_swapped = img.transpose(2, 0, 1) # (h, w, c) -> (c, h, w)
    
    patches=convert_2d_to_1d(img_dims_swapped, window_size=kernel.shape[0]) 
    weights=kernel.reshape(-1)        

    out = np.einsum('k,ckn->cn', weights, patches) # [weight | weight's shape = (25,)] * [patches | patches's shape = (3, 25, 699392)] == stack([weight @ patches[i, j, c] for c in range(channels_n)])
    out = out.reshape(channels_n, img_h, img_w)
    out = out.transpose(1, 2, 0) # (c, h, w) -> (h, w, c)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

"""
kernel_size=5
std=5
kernel = generate_gaussian_kernel(kernel_size, std)

blurred_img = convolve2d(img, kernel)
"""

def sharpen_image(image_path: str):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    sharpness_kernel = np.ones(shape=(3, 3)) * (-1)
    sharpness_kernel[1, 1] = 9
    
    sharped_image = convolve2d(image, sharpness_kernel)
    return sharped_image


def median_filter(image_path: str, kernel_size):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_h, img_w, channels_n = img.shape
    img_dims_swapped = img.transpose(2, 0, 1) # (h, w, c) -> (c, h, w)

    patches=convert_2d_to_1d(img_dims_swapped, window_size=kernel_size)
    out = np.stack([np.median(patches[c, ...], axis=0) for c in range(channels_n)])
    out = out.reshape(channels_n, img_h, img_w)
    out = out.transpose(1, 2, 0) # (c, h, w) -> (h, w, c)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out

# out = median_filter(blurred_img, kernel_size)

def generate_erosion_mask(kernel_size=5):
    """kernel_size must be odd"""
    canvas = np.zeros((kernel_size, kernel_size))
    canvas[kernel_size//2, kernel_size//2] = 1
    return binary_dilation( canvas, iterations=int((kernel_size-1)/2))


def morphology_filter(image_path: str, kernel_size, erosion=True):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img_h, img_w, channels_n = img.shape
    img_dims_swapped = img.transpose(2, 0, 1) # (h, w, c) -> (c, h, w)
    kernel = generate_erosion_mask(kernel_size=kernel_size)
    mask = kernel.reshape(-1).astype(bool)

    kernel_h, kernel_w = kernel.shape
    pad = kernel_h // 2 # kernel is a square
    pad_value = 255 if erosion else 0
    padded_img = np.pad(img_dims_swapped, ((0, 0), (pad, pad), (pad, pad)), constant_values=pad_value)
    
    i0 = np.repeat(np.arange(kernel_h), kernel_w)
    j0 = np.tile(np.arange(kernel_w), kernel_h)
    i1 = np.repeat(np.arange(img_h), img_w)
    j1 = np.tile(np.arange(img_w), img_h)
    i=i0.reshape(-1,1)+i1.reshape(1,-1)
    j=j0.reshape(-1,1)+j1.reshape(1,-1)
    patches=padded_img[:,i,j]  
    masked_image = patches[:, mask, :]

    out = masked_image.min(axis=1) if erosion else masked_image.max(axis=1)
    out = out.reshape(channels_n, img_h, img_w)
    out = out.transpose(1,2,0)
    out = np.clip(out, 0, 255).astype(np.uint8)
    return out
    

# eroded_image = morphology_filter(img, 3, erosion=True)
# diluted_image = morphology_filter(img, 3, erosion=False)

dx_kernel = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
])

dy_kernel = dx_kernel.copy().T


def get_edge(theta):
    if theta < 22.5 or 157.5 <= theta < 180.:
        return 0
    if 22.5 <= theta < 67.5:
        return 45
    if 67.5 <= theta < 112.5:
        return 90
    else:
        return 135


def get_edge_direction(theta): 
    """for edge in range [0, ... , 180] output corresponding 
    for each 3x3 window, of image (assume edging windows are padded), extract neighbors by mask, depending the edge orientation:
    0 -> [[0, 0, 0], [1, 0, 1], [0, 0, 0]] # east, west neighbors
    45 -> [[0, 0, 1], [0, 0, 0], [1, 0, 0]] # north-east, south-west
    90 -> [[0, 1, 0], [0, 0, 0], [0, 1, 0]] # north, south
    135 -> [[1, 0, 0], [0, 0, 0], [0, 0, 1]] # noth-west, south-east
    """
    if theta == 0:
        return np.array([0, 0, 0, 1, 0, 1, 0, 0, 0])
    if theta == 45:
        return np.array([0, 0, 1, 0, 0, 0, 1, 0, 0])
    if theta == 90:
        return np.array([0, 1, 0, 0, 0, 0, 0, 1, 0])
    else:
        return np.array([1, 0, 0, 0, 0, 0, 0, 0, 1])
    

def hysteresis(strong, weak):
    result = strong.copy()
    
    # 8-connected structure
    structure = np.ones((3,3), dtype=bool)
    
    while True:
        # expand current strong region
        expanded = binary_dilation(result, structure)
        
        # new strong pixels are weak ones connected to current strong
        new = expanded & weak & (~result)
        
        if not new.any():
            break
        
        result |= new
    
    return result.astype(np.uint8) * 255


def canny_edge(image_path: str, dx_kernel=dx_kernel, dy_kernel=dy_kernel, sigma=1.5, gaussian_window=(5, 5), low_threshold = 0.3, high_threshold=0.7):
    image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_image = cv2.GaussianBlur(gray_image, gaussian_window, sigma)

    img_h, img_w = gray_image.shape

    patches=convert_2d_to_1d(gray_image, window_size=3) 
 
    dx_weights=dx_kernel.reshape(-1)        
    Gx = np.einsum('k,kn->n', dx_weights, patches) # [weight | weight's shape = (25,)] * [patches | patches's shape = (3, 25, 699392)] == stack([weight @ patches[i, j, c] for c in range(channels_n)])
    Gx = Gx.reshape(img_h, img_w)

    dx_weights = dy_kernel.reshape(-1)
    dy_out = np.einsum('k,kn->n', dx_weights, patches) 
    Gy = dy_out.reshape(img_h, img_w)

    magnitude = np.sqrt(Gx**2 + Gy**2)
    magnitude = np.clip(magnitude, 0, 255).astype(np.uint8)
    orientation = np.arctan2(Gy, Gx) * (180/np.pi) % 180

    orientation_patches = orientation.flatten()
    edges = np.vectorize(get_edge)(orientation_patches)

    masks_patches = np.vstack([get_edge_direction(e) for e in edges])
    magnitude_patches = convert_2d_to_1d(magnitude, window_size=3).T
    
    center = magnitude_patches[:, 4]                    # (N,)
    neighbor_vals = np.where(masks_patches, magnitude_patches, -np.inf)
    neighbor_max = neighbor_vals.max(axis=1) # (N,)

    suppressed = np.where(center >= neighbor_max, center, 0)
    suppressed_img = suppressed.reshape(img_h, img_w)

    max_magn = suppressed_img.max()
    low_thr = max_magn*low_threshold
    upper_thr = max_magn*high_threshold

    strong_edges = suppressed_img >= upper_thr 
    weak_edges = (suppressed_img >= low_thr) & (suppressed_img < upper_thr)

    final_image = hysteresis(strong_edges, weak_edges)
    return final_image
