import os
import shutil
from pathlib import Path

import keras
import keras_nlp
from keras_nlp import tokenizers

from tensorflow_extensions.callbacks.text_generator import TopPTextGenerator
from tensorflow_extensions.datasets import WikiText2
from tensorflow_extensions.models import GPT

# keras.mixed_precision.set_global_policy('mixed_float16')

p = 10
epochs = 10
min_tokens = 5
batch_size = 64
lowercase = True
buffer_size = 512
model_name = 'gpt'
strip_accents = True
learning_rate = 1e-3
vocabulary_size = 12000
dataset_directory = Path(os.environ['DATASET_DIRECTORY'])
workspace_directory = Path(os.environ['WORKSPACE_DIRECTORY'])
vocabulary_file_path = dataset_directory / 'vocabulary.txt'

if vocabulary_file_path.exists():
    vocabulary = WikiText2.load_vocabulary(file_path=vocabulary_file_path)
else:
    vocabulary = WikiText2.compute_word_piece_vocabulary(
        directory=dataset_directory, vocabulary_size=vocabulary_size, min_tokens=min_tokens, lowercase=lowercase,
        strip_accents=strip_accents, file_path=vocabulary_file_path)

wikitext = WikiText2(tokenizer=tokenizers.WordPieceTokenizer(
    vocabulary=vocabulary, lowercase=lowercase, strip_accents=strip_accents, oov_token=WikiText2.oov_token),
    directory=dataset_directory, sequence_length=128)

model = GPT(tokenizer=wikitext.tokenizer, packer=wikitext.packer, embedding_dimension=512,
            units=[256] * 6, num_heads=6, sequence_length=wikitext.sequence_length)

loss = keras.losses.SparseCategoricalCrossentropy()
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

logs_directory = workspace_directory / 'logs' / f'{model_name}'
if logs_directory.is_dir():
    shutil.rmtree(logs_directory)
logs_directory.mkdir(parents=True, exist_ok=True)


metrics = [keras_nlp.metrics.Perplexity(mask_token_id=wikitext.pad_token_id)]
callbacks = [keras.callbacks.TensorBoard(log_dir=logs_directory), TopPTextGenerator(p=0.5)]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics, run_eagerly=False)
model.build(wikitext.input_shape(batch_size=batch_size))
model.summary()
model.fit(x=wikitext.train_dataset(batch_size=batch_size, buffer_size=buffer_size),
          validation_data=wikitext.validation_dataset(batch_size=batch_size), epochs=epochs, callbacks=callbacks)

model_directory = workspace_directory / 'models' / f'{model_name}'
if model_directory.is_dir():
    shutil.rmtree(logs_directory)
model_directory.mkdir(parents=True, exist_ok=True)
model.save(model_directory / f'{model_name}.keras')
