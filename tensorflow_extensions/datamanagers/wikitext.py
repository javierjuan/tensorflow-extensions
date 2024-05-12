import logging
from abc import ABC
from os import PathLike
from pathlib import Path
from typing import List, Tuple

import keras
import tensorflow as tf
from datasets import load_dataset, Dataset
from tokenizers import Tokenizer, pre_tokenizers, normalizers
from tokenizers.models import BPE, WordPiece
from tokenizers.normalizers import NFD, Lowercase, StripAccents
from tokenizers.pre_tokenizers import Whitespace, Punctuation
from tokenizers.processors import TemplateProcessing
from tokenizers.trainers import BpeTrainer, Trainer, WordPieceTrainer


class WikiText(ABC):
    name: str = None
    pad_token: str = '[PAD]'
    oov_token: str = '[UNK]'
    mask_token: str = '[MASK]'
    start_token: str = '[BOS]'
    end_token: str = None
    sep_token: str = '[SEP]'

    def __init__(self, tokenizer: Tokenizer, sequence_length: int):
        self.tokenizer = tokenizer
        self.sequence_length = sequence_length
        self.logger = logging.getLogger(name=self.__class__.__name__)

    def input_shape(self, batch_size: int | None = None) -> Tuple[int, int]:
        return batch_size, self.sequence_length

    def input_sample(self, batch_size: int = 32) -> keras.KerasTensor:
        return keras.Input(batch_shape=self.input_shape(batch_size=batch_size))

    @property
    def vocabulary(self) -> List[str]:
        return list(self.tokenizer.get_vocab())

    @property
    def vocabulary_size(self) -> int:
        return self.tokenizer.get_vocab_size()

    @property
    def oov_id(self) -> int | None:
        return self.tokenizer.token_to_id(self.oov_token)

    @property
    def pad_id(self) -> int | None:
        return self.tokenizer.token_to_id(self.pad_token)

    @property
    def mask_id(self) -> int | None:
        return self.tokenizer.token_to_id(self.mask_token)

    @property
    def start_id(self) -> int | None:
        return self.tokenizer.token_to_id(self.start_token)

    @property
    def end_id(self) -> int | None:
        return self.tokenizer.token_to_id(self.end_token)

    @property
    def sep_id(self) -> int | None:
        return self.tokenizer.token_to_id(self.sep_token)

    @classmethod
    def special_tokens(cls) -> List[str]:
        special_tokens = [cls.pad_token, cls.oov_token, cls.mask_token]
        special_tokens.extend([cls.start_token] if cls.start_token is not None else [])
        special_tokens.extend([cls.end_token] if cls.end_token is not None else [])
        special_tokens.extend([cls.sep_token] if cls.sep_token is not None else [])
        return special_tokens

    @classmethod
    def train_tokenizer(cls, tokenizer: Tokenizer, trainer: Trainer, batch_size: int = 1024,
                        file_path: PathLike | None = None) -> Tokenizer:
        def iterator():
            for i in range(0, len(dataset), batch_size):
                yield dataset[i: i + batch_size]['text']

        tokenizer.pre_tokenizer = pre_tokenizers.Sequence([Whitespace(), Punctuation()])
        tokenizer.normalizer = normalizers.Sequence([NFD(), Lowercase(), StripAccents()])
        dataset = load_dataset(path='wikitext', name=cls.name, split='train+test+validation')
        tokenizer.train_from_iterator(iterator(), trainer=trainer, length=len(dataset))
        tokenizer.post_processor = TemplateProcessing(
            single=f'{cls.start_token} $A',
            pair=f'{cls.start_token} $A {cls.sep_token} {cls.start_token} $B',
            special_tokens=[(cls.start_token, tokenizer.token_to_id(cls.start_token)),
                            (cls.sep_token, tokenizer.token_to_id(cls.sep_token))])
        if file_path is not None:
            tokenizer.save(str(file_path))
        return tokenizer

    @classmethod
    def train_wordpiece_tokenizer(cls, vocabulary_size: int, show_progress: bool = True, batch_size: int = 1024,
                                  file_path: PathLike | None = None) -> Tokenizer:
        tokenizer = Tokenizer(model=WordPiece(unk_token=cls.oov_token))
        trainer = WordPieceTrainer(vocab_size=vocabulary_size, special_tokens=cls.special_tokens(),
                                   show_progress=show_progress)
        return cls.train_tokenizer(tokenizer=tokenizer, trainer=trainer, batch_size=batch_size, file_path=file_path)

    @classmethod
    def train_bpe_tokenizer(cls, vocabulary_size: int, show_progress: bool = True, batch_size: int = 1024,
                            file_path: PathLike | None = None) -> Tokenizer:
        tokenizer = Tokenizer(model=BPE(unk_token=cls.oov_token))
        trainer = BpeTrainer(vocab_size=vocabulary_size, special_tokens=cls.special_tokens(),
                             show_progress=show_progress)
        return cls.train_tokenizer(tokenizer=tokenizer, trainer=trainer, batch_size=batch_size, file_path=file_path)

    def _configure_dataset(self, dataset: Dataset, batch_size: int, buffer_size: int = 512) -> tf.data.Dataset:
        def tokenization(samples):
            encodings = self.tokenizer.encode_batch(samples['text'])
            inputs, attention_mask, special_tokens_mask = [], [], []
            for encoding in encodings:
                inputs.append(encoding.ids)
                attention_mask.append(encoding.attention_mask)
                special_tokens_mask.append(encoding.special_tokens_mask)
            return {'inputs': inputs, 'attention_mask': attention_mask, 'special_tokens_mask': special_tokens_mask}

        def concatenate(samples):
            samples = {key: sum(samples[key], []) for key in samples.keys()}
            samples['labels'] = samples['inputs'].copy()[1:] + [self.pad_id]
            total_length = len(samples[list(samples.keys())[0]])
            total_length = (total_length // self.sequence_length) * self.sequence_length
            return {key: [value[i: i + self.sequence_length] for i in range(0, total_length, self.sequence_length)]
                    for key, value in samples.items()}

        self.tokenizer.no_padding()
        dataset = dataset.map(tokenization, batched=True, remove_columns=['text'], num_proc=8)
        dataset = dataset.map(concatenate, batched=True, num_proc=8)
        dataset = dataset.to_tf_dataset(columns='inputs', label_cols='labels')
        dataset = dataset.shuffle(buffer_size=buffer_size)
        dataset = dataset.batch(batch_size=batch_size, num_parallel_calls=tf.data.AUTOTUNE, drop_remainder=True)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
        return dataset

    @staticmethod
    def load_vocabulary(file_path: PathLike | str) -> List[str]:
        with open(Path(file_path), 'r') as file:
            return file.read().splitlines()

    def train_dataset(self, batch_size: int, buffer_size: int = 512) -> tf.data.Dataset:
        dataset = load_dataset(path='wikitext', name=self.name, split='train')
        return self._configure_dataset(dataset=dataset, buffer_size=buffer_size, batch_size=batch_size)

    def validation_dataset(self, batch_size: int, buffer_size: int = 512) -> tf.data.Dataset:
        dataset = load_dataset(path='wikitext', name=self.name, split='validation')
        return self._configure_dataset(dataset=dataset, buffer_size=buffer_size, batch_size=batch_size)

    def test_dataset(self, batch_size: int, buffer_size: int = 512) -> tf.data.Dataset:
        dataset = load_dataset(path='wikitext', name=self.name, split='test')
        return self._configure_dataset(dataset=dataset, buffer_size=buffer_size, batch_size=batch_size)


class WikiText2(WikiText):
    name = 'wikitext-2-v1'


class WikiText2Raw(WikiText):
    name = 'wikitext-2-raw-v1'


class WikiText103(WikiText):
    name = 'wikitext-103-v1'


class WikiText103Raw(WikiText):
    name = 'wikitext-103-raw-v1'
