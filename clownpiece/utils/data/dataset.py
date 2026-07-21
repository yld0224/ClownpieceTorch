from typing import Callable, List, Any, Union
import os
from PIL import Image
import numpy as np
import csv

from clownpiece.tensor import Tensor

class Dataset:
  
  def __init__(self):
    pass

  def __getitem__(self, index):
    """
    Returns the item at the given index.
    """
    raise NotImplementedError("Dataset __getitem__ method not implemented")
  
  def __len__(self):
    """
    Returns the total number of item
    """
    raise NotImplementedError("Dataset __len__ method not implemented")
  
"""
CSV
"""
  
class CSVDataset(Dataset):

    file_path: str
    data: List[Any]
    transform: Callable

    def __init__(self, file_path: str, transform: Callable = None):
        self.file_path = file_path
        self.transform = transform
        self.data = []
        self.load_data()

    def load_data(self):
        with open(self.file_path, "r", newline="", encoding="utf-8") as file:
            reader = csv.reader(file)
            for row in reader:
                if self.transform is not None:
                    row = self.transform(row)
                self.data.append(row)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

"""
Image
"""

class ImageDataset(Dataset):

    file_path: str
    data: List[Union[np.ndarray, Tensor]]
    labels: List[int]
    transform: Callable
    class_to_idx: dict[str, int]

    def __init__(self, file_path: str, transform: Callable = None):
        self.file_path = file_path
        self.transform = transform
        self.data = []
        self.labels = []
        self.class_to_idx = {}
        self.load_data()

    def load_data(self):
        class_names = []
        for name in os.listdir(self.file_path):
            class_path = os.path.join(self.file_path, name)
            if os.path.isdir(class_path):
                class_names.append(name)
        class_names.sort()
        self.class_to_idx = {name : index for index, name in enumerate(class_names)}
        for class_name in class_names:
            label_id = self.class_to_idx[class_name]
            class_path = os.path.join(self.file_path, class_name)
            for filename in sorted(os.listdir(class_path)):
                image_path = os.path.join(class_path, filename)
                if not os.path.isfile(image_path):
                    continue
                with Image.open(image_path) as image:
                    image_array = np.array(image)
                if self.transform is not None:
                    image_array = self.transform(image_array)
                self.data.append(image_array)
                self.labels.append(label_id)

    def __getitem__(self, index):
        return (self.data[index], self.labels[index])

    def __len__(self):
        return len(self.data)
  
"""
Image Transforms
"""


def sequential_transform(*trans):
    def transform(data):
        for tran in trans:
            data = tran(data)
        return data
    return transform

def resize_transform(size):
    if isinstance(size, int):
        height = size
        width = size
    else:
        height, width = size
    def transform(image: np.ndarray) -> np.ndarray:
        image = Image.fromarray(image)
        resized_image = image.resize((width, height), Image.Resampling.BILINEAR)
        return np.array(resized_image)
    return transform

def normalize_transform(mean, std):
    mean_arr = np.asarray(mean, dtype=float)
    std_arr = np.asarray(std, dtype=float)
    def transform(image: np.ndarray) -> np.ndarray:
        image = image.astype(np.float32) / 255.0
        return (image - mean_arr) / std_arr
    return transform

def to_tensor_transform():
    def transform(image):
        return Tensor(image.tolist())
    return transform