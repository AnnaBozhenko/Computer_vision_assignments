import cv2 
import numpy as np

def get_img(path: str) -> np.ndarray:
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    return img