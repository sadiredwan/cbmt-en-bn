import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text
import pathlib


def load_data(path):
    text = path.read_text(encoding='utf-8')
    lines = text.splitlines()
    pairs = [line.split('\t') for line in lines]
    inp = [inp for targ, inp in pairs]
    targ = [targ for targ, inp in pairs]
    return targ, inp


def preproc_en(text):
    text = tf_text.normalize_utf8(text, 'NFKD')
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    text = tf.strings.strip(text)
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text


def preproc_bn(text):
    text = tf_text.normalize_utf8(text, 'NFKC')
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text


class ShapeChecker():
    def __init__(self):
        self.shapes = {}
    
    def __call__(self, tensor, names, broadcast=False):
        if not tf.executing_eagerly():
            return
        
        if isinstance(names, str):
            names = (names,)
        
        shape = tf.shape(tensor)
        rank = tf.rank(tensor)
        if rank != len(names):
            raise ValueError(f'Rank mismatch:\n'
                             f'found {rank}: {shape.numpy()}\n'
                             f'expected {len(names)}: {names}\n')
        for i, name in enumerate(names):
            if isinstance(name, int):
                old_dim = name
            else:
                old_dim = self.shapes.get(name, None)
            new_dim = shape[i]
            if (broadcast and new_dim == 1):
                continue
            if old_dim is None:
                self.shapes[name] = new_dim
                continue
            if new_dim != old_dim:
                raise ValueError(f"Shape mismatch for dimension: '{name}'\n"
                                 f"found: {new_dim}\n"
                                 f"expected: {old_dim}\n")


def build_vocab(datapath):
    path_to_file = pathlib.Path(datapath)
    inp, targ = load_data(path_to_file)
    BUFFER_SIZE = len(inp)
    BATCH_SIZE = 64
    dataset = tf.data.Dataset.from_tensor_slices((inp, targ)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    max_vocab_size = 10000
    input_text_processor = tf.keras.layers.TextVectorization(
        standardize=preproc_en,
        max_tokens=max_vocab_size)
    output_text_processor = tf.keras.layers.TextVectorization(
        standardize=preproc_bn,
        max_tokens=max_vocab_size)
    input_text_processor.adapt(inp)
    output_text_processor.adapt(targ)
    input_vocab = np.array(input_text_processor.get_vocabulary())
    output_vocab = np.array(output_text_processor.get_vocabulary())
    return dataset, input_text_processor, output_text_processor, input_vocab, output_vocab
