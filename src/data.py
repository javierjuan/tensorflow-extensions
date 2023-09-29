import json
import logging
import sys
from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path
from typing import Union, Dict, Optional, Any, Sequence, Tuple, List
import numpy as np

import tensorflow as tf


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


class DatasetManagerBase(ABC):
    def __init__(self, name, **kwargs):
        self.logger = self._initialize_logger(name=name)

    @classmethod
    def from_json(cls, inputs: Union[Dict, PathLike, str]):
        if isinstance(inputs, (PathLike, str)):
            if Path(inputs).is_file():
                with open(inputs, 'r') as file:
                    inputs = json.load(file)
            else:
                inputs = json.loads(inputs)
        return cls(**inputs)

    def to_json(self, file_path: Optional[Union[PathLike, str]] = None, indent: int = 4) -> str:
        output = json.dumps(self.to_dict(), indent=indent)
        if file_path is not None:
            with open(file_path, 'w') as file:
                file.write(output)
        return output

    def to_dict(self) -> Dict[str, Any]:
        return self.__dict__.copy()

    @staticmethod
    def _initialize_logger(name: str, handlers: Optional[Sequence[logging.Handler]] = None) -> logging.Logger:
        logger = logging.getLogger(name=name)
        logger.setLevel(logging.INFO)
        add_stream_handler = True
        if handlers is not None:
            for handler in handlers:
                add_stream_handler &= not isinstance(handler, logging.StreamHandler)
                logger.addHandler(handler)
        if add_stream_handler:
            handler = logging.StreamHandler(stream=sys.stdout)
            handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s: %(message)s'))
            logger.addHandler(handler)
        return logger

    def datasets_from_directory(self, directory: Union[PathLike, str], extension: str = '.tfrecords',
                                verbose: bool = True) -> List[tf.data.Dataset]:
        file_paths = sorted(Path(directory).glob('*' + extension))
        if not file_paths:
            raise ValueError(f'Empty directory: {directory}')
        datasets = []
        for file_path in file_paths:
            if not file_path.is_file():
                raise FileNotFoundError(f'File {file_path} is not a valid file')
            if verbose:
                self.logger.info(f'Loading dataset: {file_path}...')
            datasets.append(tf.data.TFRecordDataset(filenames=file_path))
        return datasets

    def sample_from_datasets(self, datasets: Sequence[tf.data.Dataset], weights: Optional[Sequence[float]] = None,
                             verbose: bool = True) -> tf.data.Dataset:
        if len(datasets) == 1:
            return datasets[0]
        if weights is None:
            if verbose:
                self.logger.info(f'Uniform sampling from {datasets}')
            choice_dataset = tf.data.Dataset.range(len(datasets)).repeat()
            return tf.data.Dataset.choose_from_datasets(datasets=datasets, choice_dataset=choice_dataset)
        else:
            if len(datasets) != len(weights):
                raise ValueError('`weights` must have the same number of elements than `datasets`')
            if sum(weights) != 1:
                raise ValueError('`weights` must sum 1')
            if verbose:
                self.logger.info('\n'.join(f'Sampling from dataset {dataset} with weight {weight}' for dataset, weight
                                           in zip(datasets, weights)))
            return tf.data.Dataset.sample_from_datasets(datasets=datasets, weights=weights)

    @abstractmethod
    def train_dataset(self, *args, batch_size: int, shuffle_size: int = 128, repeat: bool = True, **kwargs) \
            -> tf.data.Dataset:
        raise NotImplementedError('This method must be implemented by all subclasses')

    @abstractmethod
    def validation_dataset(self, *args, batch_size: int, **kwargs) -> tf.data.Dataset:
        raise NotImplementedError('This method must be implemented by all subclasses')

    @abstractmethod
    def predict_dataset(self, *args, batch_size: int, **kwargs) -> tf.data.Dataset:
        raise NotImplementedError('This method must be implemented by all subclasses')

    @abstractmethod
    def train_example(self, **args) -> tf.train.Example:
        raise NotImplementedError('This method must be implemented by all subclasses')

    @tf.function
    @abstractmethod
    def parse_tfrecord(self, serialized) -> Tuple[Any, Any]:
        raise NotImplementedError('This method must be implemented by all subclasses')

    @abstractmethod
    def input_shape(self, batch_size: int):
        raise NotImplementedError('This method must be implemented by all subclasses')

    @staticmethod
    def iterate_dataset(dataset: tf.data.Dataset):
        for x, y in dataset:
            yield tuple(np.squeeze(z.numpy()) if z is not None else z for z in x), np.squeeze(y.numpy())

    @staticmethod
    def count_dataset_elements(dataset: tf.data.Dataset) -> int:
        count = 0
        for _ in dataset:
            count += 1
        return count

    @staticmethod
    def count_datasets_elements(datasets: Sequence[tf.data.Dataset]) -> List[int]:
        return [DatasetManagerBase.count_dataset_elements(dataset) for dataset in datasets]
