from abc import abstractmethod, ABC
from os import PathLike
from typing import Tuple, Dict

import keras
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import Image
from keras import ops


class COCO(ABC):
    def __init__(self, image_shape: Tuple[int, int, int], directory: PathLike | str):
        self.image_shape = image_shape
        self.num_labels = 80
        self.directory = directory
        self.categories = {
            0: 'person',
            1: 'bicycle',
            2: 'car',
            3: 'motorcycle',
            4: 'airplane',
            5: 'bus',
            6: 'train',
            7: 'truck',
            8: 'boat',
            9: 'traffic light',
            10: 'fire hydrant',
            11: 'stop sign',
            12: 'parking meter',
            13: 'bench',
            14: 'bird',
            15: 'cat',
            16: 'dog',
            17: 'horse',
            18: 'sheep',
            19: 'cow',
            20: 'elephant',
            21: 'bear',
            22: 'zebra',
            23: 'giraffe',
            24: 'backpack',
            25: 'umbrella',
            26: 'handbag',
            27: 'tie',
            28: 'suitcase',
            29: 'frisbee',
            30: 'skis',
            31: 'snowboard',
            32: 'sports ball',
            33: 'kite',
            34: 'baseball bat',
            35: 'baseball glove',
            36: 'skateboard',
            37: 'surfboard',
            38: 'tennis racket',
            39: 'bottle',
            40: 'wine glass',
            41: 'cup',
            42: 'fork',
            43: 'knife',
            44: 'spoon',
            45: 'bowl',
            46: 'banana',
            47: 'apple',
            48: 'sandwich',
            49: 'orange',
            50: 'broccoli',
            51: 'carrot',
            52: 'hot dog',
            53: 'pizza',
            54: 'donut',
            55: 'cake',
            56: 'chair',
            57: 'couch',
            58: 'potted plant',
            59: 'bed',
            60: 'dining table',
            61: 'toilet',
            62: 'tv',
            63: 'laptop',
            64: 'mouse',
            65: 'remote',
            66: 'keyboard',
            67: 'cell phone',
            68: 'microwave',
            69: 'oven',
            70: 'toaster',
            71: 'sink',
            72: 'refrigerator',
            73: 'book',
            74: 'clock',
            75: 'vase',
            76: 'scissors',
            77: 'teddy bear',
            78: 'hair drier',
            79: 'toothbrush'
        }

    @property
    def width(self) -> int:
        return self.image_shape[0]

    @property
    def height(self) -> int:
        return self.image_shape[1]

    @property
    def channels(self) -> int:
        return self.image_shape[2]

    @property
    def image_size(self) -> Tuple[int, int]:
        return self.width, self.height

    def input_shape(self, batch_size: int | None = None) -> Tuple[int, int, int, int]:
        return (batch_size,) + self.image_shape

    def input_sample(self, batch_size: int = 32) -> keras.KerasTensor:
        return keras.Input(batch_shape=self.input_shape(batch_size=batch_size))

    def train_dataset(self, batch_size: int, shuffle: bool = True, download: bool = False) -> tf.data.Dataset:
        dataset = tfds.load(name='coco/2017', split='train', data_dir=self.directory, shuffle_files=shuffle,
                            download=download, with_info=False)
        return self._configure_dataset(dataset=dataset, batch_size=batch_size)

    def validation_dataset(self, batch_size: int, download: bool = False) -> tf.data.Dataset:
        dataset = tfds.load(name='coco/2017', split='validation', data_dir=self.directory, shuffle_files=False,
                            download=download, with_info=False)
        return self._configure_dataset(dataset=dataset, batch_size=batch_size)

    def test_dataset(self, batch_size: int, download: bool = False) -> tf.data.Dataset:
        dataset = tfds.load(name='coco/2017', split='test', data_dir=self.directory, shuffle_files=False,
                            download=download, with_info=False)
        return self._configure_dataset(dataset=dataset, batch_size=batch_size)

    @staticmethod
    def image_from_tensor(image: tf.Tensor) -> Image:
        return COCO.image_from_numpy(image.numpy())

    @staticmethod
    def image_from_numpy(image: np.ndarray) -> Image:
        return Image.fromarray(image.astype(np.uint8), mode='RGB')

    @abstractmethod
    def _configure_dataset(self, dataset: tf.data.Dataset, batch_size: int) -> tf.data.Dataset:
        raise NotImplementedError

    def to_categories(self, labels: tf.Tensor | np.ndarray, threshold: float = 0.5) -> Dict[int, str]:
        if isinstance(labels, tf.Tensor):
            labels = labels.numpy()
        return {label: self.categories[label] for label in map(int, np.argwhere(labels > threshold))}

    def prevalence(self, dataset: str = 'train', batch_size: int = 128) -> np.ndarray:
        samples = 0
        counts = np.zeros(shape=(self.num_labels,))
        dataset = (self.train_dataset(batch_size=batch_size) if dataset == 'train' else
                   self.validation_dataset(batch_size=batch_size))
        for _, labels in dataset:
            counts += np.sum(labels.numpy(), axis=0)
            samples += batch_size
        return counts / samples


class COCOClassification(COCO):
    def _configure_dataset(self, dataset: tf.data.Dataset, batch_size: int) -> tf.data.Dataset:
        def as_supervised(sample):
            return sample['image'], sample['objects']['label']

        def to_fixed_size(image, label):
            return ops.image.resize(image, size=self.image_size, interpolation='lanczos3', antialias=True), label

        def to_multi_hot(image, label):
            return (image, ops.zeros(shape=(self.num_labels,), dtype='float32') if ops.size(label) == 0 else
            ops.multi_hot(label, num_tokens=self.num_labels))

        def to_float32(image, label):
            return ops.cast(image, dtype='float32'), ops.cast(label, dtype='float32')

        dataset = dataset.map(as_supervised)
        dataset = dataset.map(to_fixed_size)
        dataset = dataset.map(to_multi_hot)
        dataset = dataset.map(to_float32)
        dataset = dataset.batch(batch_size=batch_size, num_parallel_calls=tf.data.AUTOTUNE, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset


class COCOObjectDetection(COCO):
    def __init__(self, image_shape: Tuple[int, int, int], directory: str, num_queries: int = 100):
        super().__init__(image_shape=image_shape, directory=directory)
        self.num_queries = num_queries

    def _configure_dataset(self, dataset: tf.data.Dataset, batch_size: int) -> tf.data.Dataset:
        def as_supervised(sample):
            return sample['image'], {'label': sample['objects']['label'], 'bounding_box': sample['objects']['bbox']}

        def to_fixed_size(image, label):
            padding = self.num_queries - ops.size(label['label'])
            return (ops.image.resize(image, size=self.image_size, interpolation='lanczos3', antialias=True),
                    {'label': ops.pad(label['label'], (0, padding), 'constant', self.num_labels),
                     'bounding_box': ops.pad(label['bounding_box'], ((0, padding), (0, 0)), 'constant', 0)})

        def preprocess_input(image, label):
            return keras.applications.resnet.preprocess_input(image), label

        def to_one_hot(image, label):
            labels = ops.zeros(shape=(self.num_labels + 1,), dtype='float32') if ops.size(label['label']) == 0 else (
                ops.one_hot(label['label'], num_classes=self.num_labels + 1))
            return image, {'label': labels, 'bounding_box': label['bounding_box']}

        def to_float32(image, label):
            return ops.cast(image, dtype='float32'), {'label': ops.cast(label['label'], dtype='float32'),
                                                      'bounding_box': ops.cast(label['bounding_box'], dtype='float32')}

        def ensure_shape(image, label):
            image_shape = (batch_size, *self.image_shape)
            label_shape = (batch_size, self.num_queries, self.num_labels + 1)
            bounding_box_shape = (batch_size, self.num_queries, 4)
            return (tf.ensure_shape(image, shape=image_shape),
                    {'label': tf.ensure_shape(label['label'], shape=label_shape),
                     'bounding_box': tf.ensure_shape(label['bounding_box'], shape=bounding_box_shape)})

        dataset = dataset.map(as_supervised)
        dataset = dataset.map(to_fixed_size)
        dataset = dataset.map(preprocess_input)
        dataset = dataset.map(to_one_hot)
        dataset = dataset.map(to_float32)
        dataset = dataset.batch(batch_size=batch_size, num_parallel_calls=tf.data.AUTOTUNE, drop_remainder=True)
        dataset = dataset.map(ensure_shape)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset
