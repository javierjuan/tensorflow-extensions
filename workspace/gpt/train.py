import os
import shutil
from pathlib import Path

import keras
import keras_nlp
from dotenv import load_dotenv
from tokenizers import Tokenizer

from tensorflow_extensions.callbacks.text_generator import TopPTextGenerator
from tensorflow_extensions.datamanagers import WikiText103Raw
from tensorflow_extensions.models import GPT

load_dotenv()

keras.mixed_precision.set_global_policy('mixed_float16')

epochs = 30
batch_size = 64
buffer_size = 512
model_name = 'gpt'
learning_rate = 1e-3
vocabulary_size = 30000
dataset_directory = Path(os.environ['DATASET_DIRECTORY'])
workspace_directory = Path(os.environ['WORKSPACE_DIRECTORY'])
vocabulary_file_path = dataset_directory / 'vocabulary.txt'

tokenizer_path = Path(dataset_directory) / 'tokenizer.json'
if tokenizer_path.exists():
    tokenizer = Tokenizer.from_file(str(tokenizer_path))
else:
    tokenizer = WikiText103Raw.train_bpe_tokenizer(vocabulary_size=vocabulary_size, file_path=tokenizer_path)
wikitext = WikiText103Raw(tokenizer=tokenizer, sequence_length=256)

model = GPT(units=[512] * 8, num_heads=8, sequence_length=wikitext.sequence_length, embedding_dimension=768,
            vocabulary_size=wikitext.vocabulary_size, rate=0.1)

loss = keras.losses.SparseCategoricalCrossentropy(ignore_class=wikitext.pad_id)
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)

logs_directory = workspace_directory / 'logs' / f'{model_name}'
if logs_directory.is_dir():
    shutil.rmtree(logs_directory)
logs_directory.mkdir(parents=True, exist_ok=True)

metrics = [keras.metrics.SparseCategoricalAccuracy(), keras_nlp.metrics.Perplexity(mask_token_id=wikitext.pad_id)]
callbacks = [keras.callbacks.TensorBoard(log_dir=logs_directory),
             TopPTextGenerator(tokenizer=wikitext.tokenizer, p=0.7, sequence_length=wikitext.sequence_length,
                               pad_token=wikitext.pad_token)]

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
