import os
import pandas as pd
import numpy as np
import tensorflow as tf
import torchaudio
import soundfile as sf

from pathlib import Path
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import numpy as np
import os

import tensorflow_hub as hub
import tensorflow as tf

import torchaudio
import torch
from torch.utils.data import DataLoader, Dataset
model_path = 'https://kaggle.com/models/google/bird-vocalization-classifier/frameworks/TensorFlow2/variations/bird-vocalization-classifier/versions/4'
model = hub.load(model_path)
model_labels_df = pd.read_csv(hub.resolve(model_path) + "/assets/label.csv")
model_labels = {k: v for k, v in enumerate(model_labels_df.ebird2021)}

AUDIO_PATH = Path('/home/zh/525/unlabeled_soundscapes')
SAMPLE_RATE = 32000
WINDOW = 15 * SAMPLE_RATE

class SoundscapeDataset(Dataset):
    def __init__(self, audio_path):
        self.filepaths = list(audio_path.glob('*.ogg'))
    
    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, i):
        filepath = self.filepaths[i]
        audio = torchaudio.load(str(filepath))[0].numpy()[0]
        return audio, filepath.name


import librosa
import IPython.display as ipd
import librosa.display as lid
import matplotlib.pyplot as plt
cmap = plt.get_cmap('coolwarm')
class CFG:
    seed = 42
    
    # Input image size and batch size
    img_size = [128, 384]
    batch_size = 64
    
    # Audio duration, sample rate, and length
    duration = 15 # second
    sample_rate = 32000
    audio_len = duration*sample_rate
    
    # STFT parameters
    nfft = 2028
    window = 2048
    hop_length = audio_len // (img_size[1] - 1)
    fmin = 20
    fmax = 16000
    
    # Number of epochs, model name
    epochs = 10
    preset = 'efficientnetv2_b2_imagenet'
    
    # Data augmentation parameters
    augment=True

    # Class Labels for BirdCLEF 24
    class_names = sorted(os.listdir('./train_audio/'))
    num_classes = len(class_names)
    class_labels = list(range(num_classes))
    label2name = dict(zip(class_labels, class_names))
    name2label = {v:k for k,v in label2name.items()}

def load_audio(filepath):
    audio, sr = librosa.load(filepath, sr=CFG.sample_rate)
    print(f"load audio:({filepath}) shape: {audio.shape}, sample rate: {sr} , total duration: {audio.shape[0]/sr} seconds")
    return audio, sr

def get_spectrogram(audio):
    spec = librosa.feature.melspectrogram(y=audio, 
                                   sr=CFG.sample_rate, 
                                   n_mels=256,
                                   n_fft=2048,
                                   hop_length=512,
                                   fmax=CFG.fmax,
                                   fmin=CFG.fmin,
                                   )
    spec = librosa.power_to_db(spec, ref=1.0)
    min_ = spec.min()
    max_ = spec.max()
    if max_ != min_:
        spec = (spec - min_)/(max_ - min_)
    return spec

def display_audio_row(row):
        # Caption for viz
    caption = f'Id: {row.filename} | Name: {row.common_name} | Sci.Name: {row.scientific_name} | Rating: {row.rating}'
    # Read audio file
    filepath = "./train_audio/" + row.filename
    audio, sr = load_audio(filepath)
    display_audio(audio, caption)
    
def display_audio(audio, caption=""):
     # Keep fixed length audio
    audio = audio[:CFG.audio_len]
    # Spectrogram from audio
    spec = get_spectrogram(audio)
    # Display audio
    print("# Audio:")
    display(ipd.Audio(audio, rate=CFG.sample_rate))
    print('# Visualization:')
    fig, ax = plt.subplots(2, 1, figsize=(12, 2*3), sharex=True, tight_layout=True)
    fig.suptitle(caption)
    # Waveplot
    lid.waveshow(audio,
                 sr=CFG.sample_rate,
                 ax=ax[0],
                 color= cmap(0.1))
    # Specplot
    lid.specshow(spec, 
                 sr = CFG.sample_rate, 
                 hop_length=512,
                 n_fft=2048,
                 fmin=CFG.fmin,
                 fmax=CFG.fmax,
                 x_axis = 'time', 
                 y_axis = 'mel',
                 cmap = 'coolwarm',
                 ax=ax[1])
    ax[0].set_xlabel('');
    fig.show()

def display_audio_path(filepath):
    audio, sr = load_audio(filepath)
    display_audio(audio)
    
def predict_file(filepath):
    audio = torchaudio.load(str(filepath))[0].numpy()[0]
    
    audio = audio[:CFG.duration*CFG.sample_rate]
    
    # display_audio_path(filepath)

    with tf.device('/gpu:0'): 
        result = model.infer_tf(audio[None, :])
        logits = result[0].numpy()
        embedding = result[1].numpy()
        probabilities = tf.nn.softmax(logits, axis=1)
        predicted_index = tf.argmax(probabilities, axis=1).numpy()[0]
        predicted_label = model_labels[predicted_index]
        predicted_probability = probabilities[0][predicted_index]
        print(f"file: {filepath}, label: {predicted_label}, probability: {predicted_probability}")

class WindowedSoundscapeDataset(Dataset):
    def __init__(self, audio_path):
        self.dataset = SoundscapeDataset(audio_path)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        audio, filename = self.dataset[idx]
        num_clips = (len(audio) + WINDOW - 1) // WINDOW  # 计算需要多少个窗口
        clips = np.array([audio[i:i+WINDOW] if i+WINDOW <= len(audio) else np.concatenate([audio[i:], np.zeros(WINDOW - (len(audio) - i))]) for i in range(0, len(audio), WINDOW)])
        return clips, filename, num_clips

dataloader = DataLoader(WindowedSoundscapeDataset(AUDIO_PATH), batch_size=1, num_workers=os.cpu_count())

results = []
with tf.device('/gpu:0'):
    for clips, filename, num_clips in tqdm(dataloader):
        print(f"clips shape: {clips.shape}, filename: {filename}, num_clips: {num_clips}")
        clips = clips.squeeze(0)
        results_batch = model.infer_tf(clips)  # 批量推断
        
        for i in range(num_clips):
            logits = results_batch[0][i].numpy()
            embedding = results_batch[1][i].numpy()
            probabilities = tf.nn.softmax(logits)
            predicted_index = np.argmax(probabilities)
            predicted_label = model_labels[predicted_index]
            predicted_probability = probabilities[predicted_index]

            if predicted_probability < 0.77:
                continue

            print(f"file: {filename[0]}, clip: {i}, label: {predicted_label}, probability: {predicted_probability}")

            clip_filename = f"{filename[0]}_clip{i}_{predicted_label}_{predicted_probability:.2f}.ogg"
            sf.write("./annotated_audio_15/" + clip_filename, clips[i], CFG.sample_rate, format='OGG') 

# predict_file("/home/zh/525/annotated_audio/48468035.ogg_clip22_grejun2_0.90.ogg")

# 数据加载
# dataloader = DataLoader(SoundscapeDataset(AUDIO_PATH), batch_size=32, num_workers=os.cpu_count())
    
# # 预测并收集结果
# results = []
# with tf.device('/gpu:0'):
#     for audio, filename in tqdm(dataloader):
#         audio = audio[0]
#         file_predictions = []
#         for i in range(0, len(audio), WINDOW):
#             clip = audio[i:i+WINDOW]
#             if len(clip) < WINDOW:
#                 clip = np.concatenate([clip, np.zeros(WINDOW - len(clip))])
#             result = model.infer_tf(clip[None, :])
#             logits = result[0].numpy()
#             embedding = result[1].numpy()
#             probabilities = tf.nn.softmax(logits, axis=1)
#             predicted_index = tf.argmax(probabilities, axis=1).numpy()[0]
#             predicted_label = model_labels[predicted_index]
#             predicted_probability = probabilities[0][predicted_index]

#             print(f"file: {filename}, clip: {i//WINDOW}, label: {predicted_label}, probability: {predicted_probability}")

            # prediction = np.argmax(result[0].numpy(), axis=1)
            # file_predictions.append(prediction)
            
        # predictions = np.concatenate(file_predictions)
        # predicted_labels = [model_labels_df.ebird2021[i] for i in predictions if i in model_labels]
        # results.append({"filename": filename, "predictions": ', '.join(predicted_labels)})
        break

#clip file: ('1239333179.ogg',), clip: 44, label: rewbul, probability: 0.9668254852294922
# clip this file and display it


results_df = pd.DataFrame(results)
results_df.to_csv('bird_predictions.csv', index=False)