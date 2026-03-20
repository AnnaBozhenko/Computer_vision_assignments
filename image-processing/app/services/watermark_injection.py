import cv2

import numpy as np

def preprocess_watermark(filepath: str):
    watermark = cv2.imread(filepath)
    watermark = cv2.cvtColor(watermark, cv2.COLOR_BGR2RGB)
    watermark = cv2.cvtColor(watermark, cv2.COLOR_RGB2GRAY)
    watermark = watermark < 127
    return watermark


def preprocess_watermark(filepath: str):
    """returns (H, W) (image-derived size) array, 
    converts image to black-white mask, with extracted image contours (black-colored pixels)"""
    watermark = cv2.imread(filepath)
    watermark = cv2.cvtColor(watermark, cv2.COLOR_BGR2RGB)
    watermark = cv2.cvtColor(watermark, cv2.COLOR_RGB2GRAY)
    blurred_img = cv2.GaussianBlur(watermark, (5, 5), 40)
    edges = cv2.Canny(blurred_img, threshold1=140, threshold2=170)
    edges = 255*(edges == 0)
    return edges


def fit_watermark_to_image(watermark: np.ndarray, image: np.ndarray):
    """fits shape"""
    w_h, w_w = watermark.shape
    img_h, img_w = image.shape
    fitted_watermark_in_height = np.vstack(
        [watermark for _ in range(img_h // w_h)] + \
        [watermark[:(img_h % w_h), :]] 
        if img_h // w_h > 0 else [watermark[:img_h]]
    )
    fitted_watermark = np.hstack(
        [fitted_watermark_in_height for _ in range(img_w // w_w)] + \
        [fitted_watermark_in_height[:, :(img_w % w_w)]] 
        if img_w // w_w > 0 else [fitted_watermark_in_height[:, :img_w]])
    return fitted_watermark


def inject_watermark(image_path, watermark_path, level=0):
    """image shape: (H, W), watermark: (w_h, w_w)
    inserts given watermark, fitted to image shape, into blue channel"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    watermark = preprocess_watermark(watermark_path)
    channel = image[..., 2]
    channel_lsb_cleared = channel & ~np.uint8(1 << level)

    watermark = fit_watermark_to_image(watermark, channel)
    watermark_mask = (watermark == 0).astype(np.uint8)
    watermarked_channel = channel_lsb_cleared | (watermark_mask << level)
    watermarked_image = cv2.merge([image[..., 0], image[..., 1], watermarked_channel])
    return watermarked_image


def extract_watermark(watermarked_image_path, level=0):
    """
    watermarked_image is (R, G, B) iamge.
    extracts watermark, fitted to image size, from the blue channel"""
    watermarked_image = cv2.imread(watermarked_image_path)
    watermarked_image = cv2.cvtColor(watermarked_image, cv2.COLOR_BGR2RGB)
    watermarked_image_blue = watermarked_image[..., 2]
    watermark = ((watermarked_image_blue >> level) & 1).astype(np.uint8) * 255
    watermark = (watermark == 0)*255
    return watermark

# level = 0 # 0, ... , 7
# image = cv2.imread(...)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
# watermark = preprocess_watermark("data/watermark2.jpg")
# watermarked_image = inject_watermark(image, watermark, level)

# extracted = extract_watermark(watermarked_image, level)
