import logging
from abc import ABC
from os import PathLike
from pathlib import Path
from typing import List

import tensorflow as tf
import tensorflow_datasets as tfds
from keras_nlp import layers
from keras_nlp import tokenizers


class WikiText(ABC):
    name: str = None
    pad_token: str = '[PAD]'
    oov_token: str = '[UNK]'
    start_sequence_token: str = '[BOS]'
    end_sequence_token: str = '[EOS]'

    def __init__(self, tokenizer: tokenizers.Tokenizer, directory: PathLike | str, sequence_length: int):
        self.tokenizer = tokenizer
        self.directory = directory
        self.sequence_length = sequence_length

        self.logger = logging.getLogger(name=self.__class__.__name__)

    @property
    def vocabulary(self) -> List[str]:
        return self.tokenizer.get_vocabulary()

    @property
    def vocabulary_size(self) -> int:
        return self.tokenizer.vocabulary_size()

    @staticmethod
    def _configure_vocabulary_dataset(dataset: tf.data.Dataset, min_tokens: int) -> tf.data.Dataset:
        def extract_text(sample):
            return sample['text']

        def is_valid(sample):
            return len(tf.strings.split(sample, sep=' ')) > min_tokens

        dataset = dataset.map(extract_text)
        dataset = dataset.filter(is_valid)
        return dataset

    def _configure_dataset(self, dataset: tf.data.Dataset, min_tokens: int, batch_size: int,
                           buffer_size: int = 512) -> tf.data.Dataset:
        def as_supervised(sample):
            packer = layers.StartEndPacker(sequence_length=self.sequence_length,
                                           start_value=self.tokenizer.token_to_id(self.start_sequence_token),
                                           end_value=self.tokenizer.token_to_id(self.end_sequence_token))
            outputs = self.tokenizer(sample)
            return packer(outputs), outputs

        dataset = self._configure_vocabulary_dataset(dataset=dataset, min_tokens=min_tokens)
        dataset = dataset.map(as_supervised)
        dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.batch(batch_size=batch_size, num_parallel_calls=tf.data.AUTOTUNE, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    @classmethod
    def compute_word_piece_vocabulary(cls, directory: PathLike | str, vocabulary_size: int, lowercase: bool = True,
                                      strip_accents: bool = True, min_tokens: int = 5,
                                      file_path: PathLike | str | None = None) -> List[str]:
        reserved_tokens = [cls.pad_token, cls.oov_token]
        reserved_tokens.extend([cls.start_sequence_token] if cls.start_sequence_token is not None else [])
        reserved_tokens.extend([cls.end_sequence_token] if cls.end_sequence_token is not None else [])
        vocabulary = tokenizers.compute_word_piece_vocabulary(
            data=WikiText2.vocabulary_dataset(directory=directory, min_tokens=min_tokens),
            vocabulary_size=vocabulary_size, lowercase=lowercase, strip_accents=strip_accents,
            reserved_tokens=reserved_tokens, vocabulary_output_file=str(file_path))
        return vocabulary

    @staticmethod
    def load_vocabulary(file_path: PathLike | str) -> List[str]:
        with open(Path(file_path), 'r') as file:
            return file.readlines()

    @classmethod
    def vocabulary_dataset(cls, directory: PathLike | str, min_tokens: int = 5,
                           download: bool = False) -> tf.data.Dataset:
        dataset = tfds.load(name=f'huggingface:wikitext/{cls.name}', split='train', data_dir=directory,
                            shuffle_files=False, download=download, with_info=False)
        return WikiText._configure_vocabulary_dataset(dataset=dataset, min_tokens=min_tokens)

    def train_dataset(self, batch_size: int, min_tokens: int = 5, buffer_size: int = 512,
                      download: bool = False) -> tf.data.Dataset:
        dataset = tfds.load(name=f'huggingface:wikitext/{WikiText.name}', split='train', data_dir=self.directory,
                            shuffle_files=False, download=download, with_info=False)
        return self._configure_dataset(dataset=dataset, min_tokens=min_tokens,
                                       buffer_size=buffer_size, batch_size=batch_size)

    def validation_dataset(self, batch_size: int, min_length: int = 5, buffer_size: int = 512,
                           download: bool = False) -> tf.data.Dataset:
        dataset = tfds.load(name=f'huggingface:wikitext/{WikiText.name}', split='validation',
                            data_dir=self.directory, shuffle_files=False, download=download, with_info=False)
        return self._configure_dataset(dataset=dataset, min_tokens=min_length,
                                       buffer_size=buffer_size, batch_size=batch_size)

    def test_dataset(self, batch_size: int, min_length: int = 5, buffer_size: int = 512,
                     download: bool = False) -> tf.data.Dataset:
        dataset = tfds.load(name=f'huggingface:wikitext/{WikiText.name}', split='test', data_dir=self.directory,
                            shuffle_files=False, download=download, with_info=False)
        return self._configure_dataset(dataset=dataset, min_tokens=min_length,
                                       buffer_size=buffer_size, batch_size=batch_size)


class WikiText2(WikiText):
    name = 'wikitext-2-v1'

    def __init__(self, tokenizer: tokenizers.Tokenizer, directory: PathLike | str, sequence_length: int):
        super().__init__(tokenizer=tokenizer, directory=directory, sequence_length=sequence_length)


class WikiText103(WikiText):
    name = 'wikitext-103-v1'

    def __init__(self, tokenizer: tokenizers.Tokenizer, directory: PathLike | str, sequence_length: int):
        super().__init__(tokenizer=tokenizer, directory=directory, sequence_length=sequence_length)
