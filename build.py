import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from model import *
import tensorflow as tf


embedding_dim = 256
units = 1024

dataset,\
    input_text_processor,\
        output_text_processor,\
            input_vocab, output_vocab = build_vocab('data/anki/bn-en.txt')

decoder = Decoder(output_text_processor.vocabulary_size(),
                  embedding_dim,
                  units)

train_translator = TrainTranslator(embedding_dim,
                                   units,
                                   input_text_processor=input_text_processor,
                                   output_text_processor=output_text_processor)

train_translator.compile(optimizer=tf.optimizers.Adam(), loss=MaskedLoss())

batch_loss = BatchLogs('batch_loss')

train_translator.fit(dataset, epochs=int(sys.argv[1]), callbacks=[batch_loss])

translator = Translator(encoder=train_translator.encoder,
                        decoder=train_translator.decoder,
                        input_text_processor=input_text_processor,
                        output_text_processor=output_text_processor)
