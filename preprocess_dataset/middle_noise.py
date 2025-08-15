import torch
import numpy as np
import time
import torchaudio
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
import shutil


def load_audio(filepath):
    audio, sr = torchaudio.load(filepath)
    return (audio, sr)

# def get_all_inst_mp3_file_paths(dir_path):
#     return [str(file) for file in Path(dir_path).glob('*.Instrumental.mp3') if file.is_file()] #NEED CHANGE TO MP3

def noise_gen_gaussian_stereo(range_factor, frame_count, device):
    mean = 0.0
    #portion of values in range = 1 - 1 / range_factor^2
    #value range is 1 here
    std = 1.0 / range_factor
    
    # Gaussian noise: create a random normal distribution that has the same size as the data to add noise to 
    # Genearte noise with same size as that of the data.
    ch_1 = torch.normal(mean=mean, std=std, size=(frame_count,), device=device)
    ch_2 = torch.normal(mean=mean, std=std, size=(frame_count,), device=device)
    return torch.stack((ch_1, ch_2))

#signal weight controls how much of the audio signal we want to keep
def add_noise(audio_data, noise_data, signal_weight):
    audio_data *= signal_weight
    audio_data += noise_data * (1.0 - signal_weight)

def noise_file(file_path, signal_weight, device):
    try:
        # audio_data, sample_rate = load_audio_mono(file_path).to(device)[0]
        audio_data, sample_rate = torchaudio.load(file_path)
        audio_data = audio_data.to(device)
        print(f"[DEBUG] Audio data: {audio_data}, shape:{audio_data.shape}")
        frame_count = audio_data.shape[1]
        #split to 3 segments: start, middle and end
        segment_frame_count = frame_count // 3
        middle_segment_start = segment_frame_count
        middle_segment_end = segment_frame_count * 2
        # print(f"[DEBUG] Middle start: {middle_segment_start}, middle end: {middle_segment_end}")
        audio_data_middle = audio_data[:, middle_segment_start : middle_segment_end]
       
        #range factor of 4 covers the dynmaic range quite well without clipping too much
        noise_data = noise_gen_gaussian_stereo(4, segment_frame_count, device=device)
        add_noise(audio_data_middle, noise_data, signal_weight)
        
        #clip above and below, avoid out of range values
        audio_data_middle.clamp_(-1.0, 1.0)

        print(f"File {file_path} finished noising. Middle segement starts at {middle_segment_start / sample_rate}, ends at {middle_segment_end / sample_rate} ")
        # audio_data = audio_data.to(dtype=torch.float32)
        # print(f"[DEBUG] waveform: {audio_data}")
        output_path = os.path.join(noised_dataset_folder_path, Path(file_path).stem + '.Instrumental.noised.mp3')
        torchaudio.save(output_path, audio_data, sample_rate)
    #FMA dataset has corrupted files. It is normal for a few files to fail the processing.
    except Exception as e:
        print(f"Error processing {file_path}: {e}. Skipping")
        raise



def parallel_noising():
    if __name__ == "__main__":
        #ProcessPoolExecutor is probably faster, but we have file IO with soundfile, which will cause problem
        with ProcessPoolExecutor() as executor:
            for mixture_path, signal_weight in mixture_track_dir_paths_with_signal_weights:
                file_paths = [os.path.join(dataset_folder_path, file.stem + ".Instrumental.mp3") for file in Path(mixture_path).glob('*.mp3') if file.is_file()]
                file_count = len(file_paths)
                executor.map(noise_file, file_paths, [signal_weight] * len(file_paths), [device] * len(file_paths))
            


mixture_track_dir_paths_with_signal_weights = [ ("/vol/bitbucket/al4624/finetune_dataset/fma_large/sep/noise_0.1", 0.9),
                            ("/vol/bitbucket/al4624/finetune_dataset/fma_large/sep/noise_0.3", 0.7),
                            ("/vol/bitbucket/al4624/finetune_dataset/fma_large/sep/noise_0.5", 0.5),
                            ("/vol/bitbucket/al4624/finetune_dataset/fma_large/sep/noise_0.7", 0.3),
                            ("/vol/bitbucket/al4624/finetune_dataset/fma_large/sep/noise_1.0", 0.0)]

dataset_folder_path = "/vol/bitbucket/al4624/finetune_dataset/fma_large_sep"
noised_dataset_folder_path = "/vol/bitbucket/al4624/finetune_dataset/fma_large_noised_inst"
cuda_idx = 0
device = torch.device(f"cuda:{cuda_idx}" if torch.cuda.is_available() else "cpu")
# #read input files as np array, assuming audio is stereo
# file_paths = get_all_inst_mp3_file_paths(dataset_folder_path)


start = time.time()
# for file in file_paths:
#    noise_file(file, 1.0, device)
parallel_noising()
end = time.time()
elapsed = end - start
print(f"Processing took {elapsed:.2f} seconds.")