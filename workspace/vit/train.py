import os
import shutil
from pathlib import Path

import keras

from tensorflow_extensions.datasets import COCOClassification
from tensorflow_extensions.models import ViT

# keras.mixed_precision.set_global_policy('mixed_float16')

epochs = 10
shuffle = True
batch_size = 32
model_name = 'vit'
learning_rate = 1e-3
image_shape = (240, 240, 3)
dataset_directory = Path(os.environ['DATASET_DIRECTORY'])
workspace_directory = Path(os.environ['WORKSPACE_DIRECTORY'])

coco = COCOClassification(image_shape=image_shape, directory=dataset_directory)
model = ViT(patch_size=(16, 16), num_labels=coco.num_labels, embedding_dimension=1024, encoder_units=[2048] * 3,
            encoder_num_heads=8, mlp_units=[512, 128], epsilon=1e-6, name=model_name)

loss = keras.losses.BinaryCrossentropy()
optimizer = keras.optimizers.AdamW(learning_rate=learning_rate)

logs_directory = workspace_directory / 'logs' / f'{model_name}'
if logs_directory.is_dir():
    shutil.rmtree(logs_directory)
logs_directory.mkdir(parents=True, exist_ok=True)

model_directory = workspace_directory / 'models' / f'{model_name}.keras'
metrics = [keras.metrics.Precision(), keras.metrics.Recall()]
callbacks = [keras.callbacks.TensorBoard(log_dir=logs_directory)]

model.compile(optimizer=optimizer, loss=loss, metrics=metrics, run_eagerly=False)
model.build(coco.input_shape(batch_size=batch_size))
model.summary()
model.fit(x=coco.train_dataset(batch_size=batch_size, shuffle=shuffle),
          validation_data=coco.validation_dataset(batch_size=batch_size), epochs=epochs, callbacks=callbacks)
model.save(model_directory)
