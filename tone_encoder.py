# -*- coding: utf-8 -*-
"""
Created on Tue May 12 15:33:31 2020

@author: Henry Ives

Utility functions for the tone_encoder neural network
"""

import os
import datetime
import copy

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import simpleaudio as sa

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, concatenate
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Conv1D, MaxPooling1D, UpSampling1D
from tensorflow.keras.callbacks import TensorBoard

try:
    import google.colab
    IN_COLAB = True
    from google.colab import auth
except:
    IN_COLAB = False


# DOWNLOAD AND PREPARE DATA -- roughly equivalent to tfds.load

# Can use different splits depending on system abilities. Test < Valid < Train
    
def obtain_datasets(download=False, data_dir='./Overflow', download_dir='./Overflow/downloads', local_train=False):
    """Return Dataset objects and info for Nsynth dataset
    Uses Google Cloud Storage instance if in Colab and available.
    Will search for local copy in download_dir, or download into data_dir + '/nsynth'
    nsynth_ds uses train split in Colab, valid split otherwise (unless local_train is True)
    test_ds uses test split.

    Return - nsynth_ds, test_ds, nsynth_info, train_split
    """

    if IN_COLAB and tfds.is_dataset_on_gcs('nsynth'):
        
        nsynth_ds, nsynth_info = tfds.load(
                                'nsynth', split='train', download=False,
                                with_info=True, try_gcs=True)
        test_ds, _ = tfds.load(
                    'nsynth', split='test', download=False,
                    with_info=True, try_gcs=True)
        train_split = 'train'

    else:

        nsynth_builder = tfds.builder('nsynth', data_dir=data_dir) # , **builder_kwargs

        if download:
            certain = input('Are you sure you want to download 70 GiB dataset? ')
            if certain == 'yes' or certain == 'y':
                tfds_config = tfds.download.DownloadConfig(
                    extract_dir=download_dir, manual_dir=download_dir,
                    download_mode=tfds.GenerateMode.REUSE_DATASET_IF_EXISTS, compute_stats=None, max_examples_per_split=None,
                    register_checksums=False, beam_runner=None, beam_options=None)
                nsynth_builder.download_and_prepare(download_dir=download_dir, download_config=tfds_config)    

        local_split = 'train' if local_train else 'valid'
        nsynth_ds = nsynth_builder.as_dataset(split=local_split, as_supervised=False)
        test_ds = nsynth_builder.as_dataset(split='test', as_supervised=False)
        nsynth_info = nsynth_builder.info
        train_split = local_split

    return nsynth_ds, test_ds, nsynth_info, train_split


#  Many thanks to ibab/tensorflow-wavenet (https://github.com/ibab/tensorflow-wavenet/)

def _mu_law_encode(audio, quantization_channels=255):
    '''Quantize waveform amplitudes.
    '''
    with tf.name_scope('encode'):
        mu = tf.cast(quantization_channels - 1, dtype='float32')
        # Perform mu-law companding transformation (ITU-T, 1988).
        # Minimum operation is here to deal with rare large amplitudes caused
        # by resampling.
        safe_audio_abs = tf.minimum(tf.abs(audio), 1.0)
        magnitude = tf.math.log1p(mu * safe_audio_abs) / tf.math.log1p(mu)
        signal = tf.sign(audio) * magnitude
        # Quantize signal to the specified number of levels.
        return tf.cast((signal + 1) / 2 * mu + 0.5, dtype='int16')


def _mu_law_decode(output, quantization_channels=255):
    '''Recover waveform from quantized values.
    Clips at magnitude of 1 (for speaker safety)
    '''
    with tf.name_scope('decode'):
        mu = quantization_channels - 1
        # Map values back to [-1, 1].
        signal = 2 * (tf.cast(output, dtype='float32') / mu) - 1
        # Perform inverse of mu-law transformation.
        magnitude = (1 / mu) * ((1 + mu)**abs(signal) - 1)
        return tf.sign(signal) * magnitude

def _mu_law_decode_clip(output, quantization_channels=255):
    '''Recover waveform from quantized values.
    Clips at magnitude of 1 (for speaker safety)
    '''
    with tf.name_scope('decode'):
        mu = quantization_channels - 1
        # Map values back to [-1, 1].
        signal = 2 * (tf.cast(output, dtype='float32') / mu) - 1
        # Perform inverse of mu-law transformation.
        magnitude = (1 / mu) * ((1 + mu)**abs(signal) - 1)
        # Clip magnitudes above 1
        magnitude = tf.math.minimum(magnitude, 1)
        return tf.sign(signal) * magnitude

def _mu_law_decode_prenorm(output, quantization_channels=255):
    '''Recover waveform from quantized values.
    Normalizes to quantization_channels before decoding
    '''
    with tf.name_scope('decode'):
        mu = quantization_channels - 1
        # Map values back to [-1 = 0, 1 = quantization_channels].
        signal = 2 * (tf.cast(output, dtype='float32') / mu) - 1
        # Normalize values to [-1, 1].
        signal = signal / tf.reduce_max(tf.abs(signal))
        # Perform inverse of mu-law transformation.
        magnitude = (1 / mu) * ((1 + mu)**abs(signal) - 1)
        return tf.sign(signal) * magnitude

def _mu_law_decode_postnorm(output, quantization_channels=255):
    '''Recover waveform from quantized values.
    Normalizes to quantization_channels after decoding
    '''
    with tf.name_scope('decode'):
        mu = quantization_channels - 1
        # Map values back to [-1 = 0, 1 = quantization_channels].
        signal = 2 * (tf.cast(output, dtype='float32') / mu) - 1
        # Normalize values to [-1, 1].
        # Perform inverse of mu-law transformation.
        magnitude = (1 / mu) * ((1 + mu)**abs(signal) - 1)
        signal = tf.sign(signal) * magnitude
        signal = signal / tf.reduce_max(tf.abs(signal))
        return signal

def get_elem(ds, index=0):
    """Return ds element from index as list
    """
    return list(ds.take(index+1).as_numpy_iterator())[index]

def play_audio(element_audio, sample_width):
    """Play audio from array-style audio given sample width 
    """
    NUM_CHANNELS = 1
    SAMPLE_RATE = 16000
    temp_player = sa.play_buffer(element_audio, NUM_CHANNELS, sample_width, SAMPLE_RATE)
    temp_player.wait_done()
    
def show_audio_from_element(element_audio, encode=False, decode=False, play=True, view_range=(0,-1)): 
    """Print info, graph audio, and play audio (if not in Colab)
    """
    sample_width = 4

    if encode:
        element_audio = tf.cast(_mu_law_encode(element_audio), dtype='float32')
        sample_width = 2
    if decode:
        element_audio = _mu_law_decode(tf.cast(element_audio, dtype='float32'))
        sample_width = 4
       
    plt.figure(figsize=(12,4))
    plt.plot(element_audio[view_range[0]:view_range[1]])
    plt.show()
    
    if (not IN_COLAB) and play:
        play_audio(element_audio, sample_width)

def ds_peek(ds, index=0, div=1, show=True, encode=False, decode=False, view_range=(0,-1)):
    """Utility function for peeking into a dataset.
    Prints info, graphs audio, and plays audio
    Returns element at index index of dataset ds
    """

    element = get_elem(ds, index)
    
    if show:
        print(element)
        show_audio_from_element(copy.deepcopy(element['audio'][::]), encode=encode, decode=decode, view_range=view_range)
    
    return element


# Pre-processing functions

# @tf.function  # I think still will be compiled in train_fn context
def _extract_audio(elem):
    """Extract audio, return 32000 samples from transient
    """
    audio_tensor = tf.cast(_mu_law_encode(elem['audio']), dtype='float32')
    max_idx = tf.math.argmax(audio_tensor[:32000])
    if max_idx < 32000:
        audio_tensor = audio_tensor[max_idx : max_idx + 32000]
    else:
        audio_tensor = audio_tensor[max_idx - 32000 : max_idx]
    audio_tensor = tf.expand_dims(audio_tensor, axis=1)
    audio_tensor = tf.ensure_shape(audio_tensor, (32000, 1))
    return audio_tensor

# @tf.function  # I think still will be compiled in train_fn context
def _extract_metadata(elem):
    """Extract metadata from elem dict to Tensor
    """
    metadata_list=[]
    for inst in elem['instrument'].values():
        metadata_list.append(inst)
    metadata_list.append(elem['pitch'])
    metadata_list.append(elem['velocity'])
    for quality in elem['qualities'].values():
        metadata_list.append(int(quality))
    metadata_list.append(0)
    metadata_tensor = tf.convert_to_tensor(metadata_list, dtype='float32')
    metadata_tensor = tf.concat([metadata_tensor, metadata_tensor], axis=0)
    metadata_tensor = tf.expand_dims(metadata_tensor, axis=1)
    metadata_tensor = tf.repeat(metadata_tensor, repeats=8, axis=1)
    metadata_tensor = tf.debugging.check_numerics(metadata_tensor, 'metadata nan or inf', name=None)
    return metadata_tensor

@tf.function
def train_fn(elem):
    """Translate original nsynth dataset to form ((audio, metadata), audio)
    for training the model.
    """
    audio_elem = _extract_audio(elem)
    metadata_elem = _extract_metadata(elem)
    return ((audio_elem, metadata_elem), audio_elem)

def use_fn(elem):
    """Translate original nsynth dataset to form (audio, metadata)
    for using the model.
    """
    audio_elem = _extract_audio(elem)
    audio_elem = tf.expand_dims(audio_elem, axis=0)
    meta_elem = _extract_metadata(elem)
    meta_elem = tf.expand_dims(meta_elem, axis=0)
    return (audio_elem, meta_elem)

def encode_fn(elem):
    """Translate original nsynth dataset to form (audio)
    for using the encoder.
    """
    audio_elem = _extract_audio(elem)
    audio_elem = tf.expand_dims(audio_elem, axis=0)
    return audio_elem


def construct_ToneEncoder():
    """Construct a new instance of the ToneEncoder model and encoder.
    """

    print('Constructing model...')
    audio_shape = (64, 32000, 1)  # (timesteps, channels/features, (*batch))
    meta_shape = (64, 32, 8)
    leaky_relu = LeakyReLU(alpha=0.2)

    # Encoder
    audio_input = Input(batch_shape=audio_shape, dtype='float32', name='audio_input')
    en_conv_0 = Conv1D(44, 122, strides=8, padding='same', activation=leaky_relu, name='en_conv_0')(audio_input)
    en_conv_1 = Conv1D(44, 122, strides=8, padding='same', activation=leaky_relu, name='en_conv_1')(en_conv_0)
    en_pool_1 = MaxPooling1D(2, 2, padding='same', name='en_pool_1')(en_conv_1)
    en_conv_2 = Conv1D(8, 122, strides=8, padding='same', activation=leaky_relu, name='en_conv_2')(en_pool_1)
    encoding = en_conv_2
    
    # Combine encoding with metadata
    meta_input = Input(batch_shape=meta_shape, dtype='float32', name='meta_input')
    merge_0 = concatenate([en_conv_2, meta_input], axis=1, name='merge_0')

    # Decoder
    de_conv_0 = Conv1D(44, 122, padding='same', activation=leaky_relu, name='de_conv_0')(merge_0)
    de_upsample_0 = UpSampling1D(size=50, name='de_upsample_0')(de_conv_0)
    de_conv_1 = Conv1D(44, 122, padding='same', activation=leaky_relu, name='de_conv_1')(de_upsample_0)
    de_upsample_1 = UpSampling1D(size=10, name='de_upsample_1')(de_conv_1)
    de_conv_2 = Conv1D(44, 122, padding='same', activation=leaky_relu, name='de_conv_2')(de_upsample_1)
    audio_output = Conv1D(1, 122, activation=leaky_relu, padding='same', name='audio_output')(de_conv_2)

    model = Model(inputs=[audio_input, meta_input],
                    outputs=[audio_output],
                    name='model')
    print('Model constructed.')
    encoder = Model(inputs=audio_input,
                    outputs=en_conv_2,
                    name='encoder')
    print('Encoder constructed')
    
    print('Compiling model...')
    optimizer = Adam(learning_rate=0.0001)
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    print('Model compiled.')
    # Consider SGD once it's nearly converged

    return model, encoder


def get_ToneEncoder(method='fresh', save_dir='.\\savepoints\\', summary=True, plot=True):
    """Construct or load ToneEncoder.

    method: 
        'full_load': load full model from save_dir
            (mostly deprecated—use for 20200511-061648 and before)
        'checkpoint': construct a new instance of the model and load weights from save_dir
                (either most recent checkpoint or savepoints after 20200511-061648)
        'fresh': construct a new model (loads nothing)

    Return: model, encoder, plot_img
    """
    full_load = ( method == 'full_load' or method == 'load' )
    checkpoint = ( method == 'checkpoint' or method == 'check' )
    fresh = ( method == 'fresh' or method == 'new' or method == '' )
    
    if fresh:
        model, encoder = construct_ToneEncoder()

    elif checkpoint:
        model, encoder = construct_ToneEncoder()
        print('Loading weights from ', save_dir, '...')
        model.load_weights(save_dir)
        print('Weights loaded')

    elif full_load:
        print('Loading model from ', save_dir, '...')
        if not save_dir:
            print("Directory Error.")
            return
        model = tf.keras.models.load_model(save_dir)
        print('Model loaded.')
        try:
            encoder = Model(inputs=model.get_layer('audio_input'),
                            outputs=model.get_layer('en_conv_2'),
                            name='encoder')
            print('Encoder constructed')
        except:
            encoder = None
            print('Encoder loading failed. Try checkpoint method.')
    
    if summary:
        model.summary()
        
    plot_img = None
    if plot:
        plot_img = tf.keras.utils.plot_model(model, "ToneEncoder.png", show_shapes=True)
    
    return model, encoder, plot_img


def datesave_model(model, save_dir):
    """Save model to timestamped directory under save_dir

    Version 1.1 (after 20200511-061648) only save weights
    """
    temp_save_dir = os.path.join(save_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), '')
    print('\nSaving weights in ', temp_save_dir, '...')
    model.save_weights(temp_save_dir)
    print('Weights saved.')

def plot_histories(histories):
    """Plot histories from list of Keras History objects
    """
    #Plot history
    if len(histories) > 1:
        plt.plot(np.concatenate([hist[0].history['loss'] for hist in histories[:]]))
        plt.plot(np.concatenate([hist[0].history['val_loss'] for hist in histories[:]]))
    else:
        plt.plot(histories[0][0].history['loss'])
        plt.plot(histories[0][0].history['val_loss'])
        plt.show()
    print('Last Loss: ', histories[-1][0].history['loss'][-1])
    print('Last Val Loss: ', histories[-1][0].history['val_loss'][-1])

def train_TE(model, train_ds, eval_ds, num_epochs, save_dir, check_dir, log_dir):
    """Train the ToneEncoder
    """
    verbose = 1 if num_epochs <= 10 else 2  # Avoid Keras output buffer limit
    log_dir = os.path.join(log_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),'')

    # Fault tolerance part 1
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=check_dir,
                                                    save_weights_only=True,
                                                    verbose=verbose)
    tb_callback = TensorBoard(log_dir=log_dir,
                            histogram_freq=1,
                            profile_batch=0)
    try:
        # Train the model
        history = model.fit(train_ds,
                epochs=num_epochs,
                callbacks=[cp_callback, tb_callback],
                validation_data=eval_ds)

    finally:  # Fault tolerance part 2
        # Save latest version
        datesave_model(model, save_dir)
    
    return history


def demo_ToneEncoder(model, encoder, demo_elem, write_wav=False, wav_dir='./wavs/', zoom_slice=(0, 1500), decode_method='prenorm'):
    """Display graphs for a model given demo element
    
    decode_method: 'prenorm' normalizes before decoding, 'clip' hard clips after decoding
    """

    nsynth_families = ('Bass', 'Brass', 'Flute', 'Guitar', 'Keyboard', 'Mallet',
                       'Organ', 'Reed', 'String', 'Synth_lead', 'Vocal')
    SAMPLE_RATE = 16000
    elem_name = demo_elem['id'].decode('UTF-8')
    
    print("Demoing", elem_name)
    print('Family:', nsynth_families[demo_elem['instrument']['family']])
    print("Pitch:", demo_elem['pitch'])

    test_input = use_fn(demo_elem)
    test_output = model.predict(test_input)
    encoder_input = encode_fn(demo_elem)
    encoder_output = encoder.predict(encoder_input)

    encoded_input_audio = test_input[0][0]
    encoded_output_audio = test_output[0]

    decoded_input_audio = _mu_law_decode(test_input[0][0])
    if decode_method =='clip':
        decoded_output_audio = _mu_law_decode_clip(test_output[0])
    elif decode_method == 'postnorm':
        decoded_output_audio = _mu_law_decode_postnorm(test_output[0])
    else:
        decoded_output_audio = _mu_law_decode_prenorm(test_output[0])

    plt.figure(figsize=(24,16))

    plt.subplot(4, 2, 1)
    plt.plot(decoded_input_audio)
    plt.title('Input Signal')

    plt.subplot(4, 2, 2)
    plt.specgram(tf.squeeze(decoded_input_audio), 512, 16000, scale='dB')
    plt.title('Input Spectrogram')

    plt.subplot(4, 2, 3)
    plt.plot(decoded_output_audio)
    plt.title('Output Signal')

    plt.subplot(4, 2, 4)
    plt.specgram(tf.squeeze(decoded_output_audio), 512, 16000, scale='dB')
    plt.title('Output Spectrogram')

    plt.subplot(4, 2, 5)
    plt.plot(encoded_input_audio[zoom_slice[0]:zoom_slice[1]])
    plt.plot(encoded_output_audio[zoom_slice[0]:zoom_slice[1]])
    plt.title('Input vs. Output, Zoomed')

    plt.subplot(4, 2, 7)
    plt.plot(encoder_output.squeeze())
    plt.title('Encoding')

    plt.subplot(4, 2, 8)
    plt.imshow(encoder_output.squeeze().T)
    plt.title("Encoding, Bird's Eye View")

    plt.tight_layout()
    plt.show()

    if not IN_COLAB:
        print('Playing input...')
        play_audio(decoded_input_audio, 4)

        print('Playing output...')
        play_audio(decoded_output_audio, 4)

    else:
        print('Cannot play audio in Colab.')
    
    if write_wav:
        in_filename = os.path.join(wav_dir, elem_name + '_input.wav')
        in_wav = tf.audio.encode_wav(decoded_input_audio, SAMPLE_RATE)
        print('Writing input audio to', in_filename, '...')
        tf.io.write_file(in_filename, in_wav)
        print('Input audio written.')
        
        out_filename = os.path.join(wav_dir, elem_name + '_output.wav')
        out_wav = tf.audio.encode_wav(decoded_output_audio, SAMPLE_RATE)
        print('Writing output audio to', out_filename, '...')
        tf.io.write_file(out_filename, out_wav)
        print('Output audio written.')
        

    return decoded_input_audio, decoded_output_audio