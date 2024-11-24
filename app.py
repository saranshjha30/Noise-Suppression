# Importing project dependencies

import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from scipy.io.wavfile import write
import util_functions as ufs
import time
import os
preds = None
# Setting config option for deployment

#st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('Noise-Suppressor')
st.subheader('Removes background-noise from audio samples')

# UI design

nav_choice = st.sidebar.radio('Navigation', ['Home'], index=0)

_param_dict = {}  # Used for getting plot related information
_path_to_model = 'utils/models/NoiseSuppressionModel.h5'  # Path to pre-trained model
_targe_file = 'utils/outputs/preds.wav'  # target file for storing model.output

if nav_choice == 'Home':
    st.image('utils/images/header.jpg', width=450)
    st.info('Upload your audio sample below')
    audio_sample = st.file_uploader('Audio Sample', ['wav'])  # Get audio sample as an input from users
    if audio_sample:
        try:
            prog = st.progress(0)
            model = ufs.load_model(_path_to_model)  # call to the utility module to cache the model
            audio = tf.audio.decode_wav(audio_sample.read(), desired_channels=1)
            # decoding audio waveform by using  tf.audio.decode_wav as a mono sound wave
            _param_dict.update({'audio_sample': audio.audio})
            flag = 1
            for i in range(100):
                time.sleep(0.001)
                prog.progress(i + 1)
            st.info('Uploaded audio sample')
            st.audio(audio_sample)
            with st.spinner('Wait for it...'):
                time.sleep(1)

                os.makedirs('utils/outputs', exist_ok=True)
                try:
                    model = ufs.load_model(_path_to_model)
                    audio = tf.audio.decode_wav(audio_sample.read(), desired_channels=1)
                    audio_data = audio.audio.numpy()
                    print(f"Input audio data shape: {audio_data.shape}, dtype: {audio_data.dtype}")
                    _param_dict.update({'audio_sample': audio.audio})
                    print(f"Original audio shape: {audio_data.shape}")
                    target_length = 16000  # Desired audio length (e.g., 1 second at 16kHz)
                    if audio_data.shape[0] > target_length:
                        audio_data = audio_data[:target_length]
                    elif audio_data.shape[0] < target_length:
                        padding = target_length - audio_data.shape[0]
                        audio_data = np.pad(audio_data, ((0, padding), (0, 0)), mode='constant')
                    audio_data = np.expand_dims(audio_data, axis=0)  # Add batch dimension
                    audio_data = np.expand_dims(audio_data, axis=-1)  # Add channel dimension
                    preds = model.predict(audio_data)
                    print(f"Predicted shape: {preds.shape}, values: {preds}")
                    print(f"Predicted shape before reshaping: {preds.shape}, dtype: {preds.dtype}")
                    preds = tf.reshape(preds, (-1, 1))  # Flatten predictions if needed
                    _param_dict.update({'predicted_outcomes': preds})
                    preds = np.array(preds)
                    print(f"Predicted shape after conversion: {preds.shape}, dtype: {preds.dtype}")
                    if preds.size == 0 or np.all(preds == 0):
                        raise ValueError("Predictions are empty!")
                    preds = np.clip(preds * 32767, -32768, 32767).astype(np.int16)
                    target_file = 'utils/outputs/processed_audio.wav'  # Update with the correct file name/path
                    write(target_file, 44100, preds)  # 44100 Hz sample rate, 16-bit PCM
                    print(f"Written audio to: {target_file}")
                    st.success('Audio after noise removal')
                    st.audio(target_file)
                except Exception as e:
                    st.error(f"An error occurred: {e}")

                #preds = model.predict(tf.expand_dims(audio.audio, 0))  # using this EagerTensor to suppress te noie
                #preds = tf.reshape(preds, (-1, 1))
                #_param_dict.update({'predicted_outcomes': preds})
                #preds = np.array(preds)
                #write(_targe_file, 44100, preds)  # writing the output file to play
            #st.success('Audio after noise removal')
            #st.audio(_targe_file)



            # Visual Representation of model's prediction using sync plots

            prediction_stats = st.checkbox('Prediction Plots')
            noise_rem = st.checkbox('Noise Removal Plots')
            if noise_rem:
                fig, axes = plt.subplots(2, 1, figsize=(10, 6))
                axes[0].plot(np.arange(len(_param_dict['audio_sample'])), _param_dict['audio_sample'], c='r')
                axes[0].set_title('Original audio sample')
                axes[1].plot(np.arange(len(_param_dict['predicted_outcomes'])), _param_dict['predicted_outcomes'],
                             c='b')
                axes[1].set_title('Noise suppressed audio output')
                st.pyplot(fig)

            if prediction_stats:
                fig, ax = plt.subplots(figsize=(10, 6))  # Create a new figure
                ax.plot(np.arange(len(_param_dict['audio_sample'])), _param_dict['audio_sample'], c='r', label='Original audio sample')
                ax.plot(np.arange(len(_param_dict['predicted_outcomes'])), _param_dict['predicted_outcomes'], c='b', label='Noise suppressed audio output')
                ax.legend()
                st.pyplot(fig)


        except Exception as e:
            print(e, type(e))
