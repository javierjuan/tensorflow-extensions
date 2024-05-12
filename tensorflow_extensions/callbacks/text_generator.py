import keras
import keras_nlp
from keras import ops
from tokenizers import Tokenizer


class TextGenerator(keras.callbacks.Callback):
    def __init__(self, tokenizer: Tokenizer, sampler: keras_nlp.samplers.Sampler, sequence_length: int,
                 pad_token: str = '[PAD]'):
        super().__init__()
        self.tokenizer = tokenizer
        self.sampler = sampler
        self.sequence_length = sequence_length
        self.pad_id = int(self.tokenizer.encode(pad_token).ids[0])
        self.pad_token = pad_token

    def next(self, prompt, cache, index):
        logits = self.model(prompt)[:, index - 1, :]
        return logits, None, cache

    def on_epoch_end(self, epoch, logs=None):
        self.tokenizer.enable_padding(pad_id=self.pad_id, pad_token=self.pad_token, length=self.sequence_length)
        prompt = ops.expand_dims(ops.convert_to_tensor(self.tokenizer.encode('').ids), axis=0)
        tokens = self.sampler(next=self.next, prompt=prompt, index=1)
        text = self.tokenizer.decode(ops.reshape(tokens, -1))
        self.tokenizer.no_padding()
        print(f"\nGenerated text: \n{text}\n")


class TopKTextGenerator(TextGenerator):
    def __init__(self, tokenizer: Tokenizer, k: int, sequence_length: int, pad_token: str = '[PAD]',
                 seed: int | None = None):
        super().__init__(tokenizer=tokenizer, sampler=keras_nlp.samplers.TopKSampler(k=k, seed=seed),
                         sequence_length=sequence_length, pad_token=pad_token)


class TopPTextGenerator(TextGenerator):
    def __init__(self, tokenizer: Tokenizer, p: float, sequence_length: int, pad_token: str = '[PAD]',
                 seed: int | None = None):
        super().__init__(tokenizer=tokenizer, sampler=keras_nlp.samplers.TopPSampler(p=p, seed=seed),
                         sequence_length=sequence_length, pad_token=pad_token)
