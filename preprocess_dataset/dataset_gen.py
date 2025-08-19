import pandas as pd
import ast

import os
from pathlib import Path
import shutil

from concurrent.futures import ProcessPoolExecutor

# -------- Load track and genre data once --------
track_df_path = "fma_metadata/tracks.csv"
genre_df_path = "fma_metadata/genres.csv"
track_df = pd.read_csv(track_df_path, skiprows=1, dtype={'Unnamed: 0': str}).drop(index=0)
track_df = track_df.rename(columns={'Unnamed: 0': 'track_id'})[['track_id', 'genre_top', 'genres', 'genres_all']]
track_df['track_id'] = track_df['track_id'].astype(int)
genre_df = pd.read_csv(genre_df_path)

def get_genres_from_id(track_id):
    row = track_df[track_df['track_id'] == track_id]
    genre_ids = ast.literal_eval(row['genres_all'].values[0])
    genre_strs = []
    for genre_id in genre_ids:
        genre_row = genre_df[genre_df['genre_id'] == genre_id]
        genre_strs.append(genre_row['title'].values[0])
    return genre_strs



def get_all_track_paths(dir_path):
    return [str(file) for file in Path(dir_path).glob('*.mp3') if file.is_file()]

def data_gen(mixture_track_path):
    try:
        track_name = Path(mixture_track_path).stem

        noised_track_path = os.path.join(noised_inst_track_dir_path, track_name + '.Instrumental.noised.mp3')

        if (not os.path.exists(noised_track_path)):
            print(f"[ERROR] Noised track does not exist for track {noised_track_path}!")
            return

        inst_track_path = os.path.join(inst_track_dir_path, track_name + '.Instrumental.mp3')

        if (not os.path.exists(inst_track_path)):
            print(f"[ERROR] Noised track does not exist for track {inst_track_path}!")
            return


        #we need 4 files: lyric txt, prompt txt, track mp3, and noised track mp3
        lyrics_str = "[instrumental]" #we deal with instrumental tracks
        
        with open(os.path.join(dataset_path, track_name + '_lyrics.txt'), 'w') as f:
            f.write(lyrics_str)

        genre_strs = get_genres_from_id(int(track_name))
        tag_str = ', '.join(genre_strs)

        with open(os.path.join(dataset_path, track_name + '_prompt.txt'), 'w') as f:
            f.write(tag_str)

        shutil.copyfile(inst_track_path, os.path.join(dataset_path, Path(inst_track_path).name))

        shutil.copyfile(noised_track_path, os.path.join(dataset_path, Path(noised_track_path).name))
    except Exception as e:
        print(f"[ERROR] Error generating data for track {mixture_track_path}: {e}")


# mixture_track_dir_path = "/vol/bitbucket/al4624/finetune_dataset/fma_large/sep/noise_0.1" #10% noise
# inst_track_dir_path="/vol/bitbucket/al4624/finetune_dataset/fma_large_sep"
# noised_inst_track_dir_path="/vol/bitbucket/al4624/finetune_dataset/fma_large_noised_inst"
# dataset_path="/vol/bitbucket/al4624/git_repo/ACE-Step/data"


#testing setup paths
mixture_track_dir_path = "/homes/al4624/Documents/YuE_finetune/finetune_testing_dataset/mixture_audio" #10% noise
inst_track_dir_path="/homes/al4624/Documents/YuE_finetune/finetune_testing_dataset/sep_audio"
noised_inst_track_dir_path="/homes/al4624/Documents/YuE_finetune/finetune_testing_dataset/noised_inst"
dataset_path="/vol/bitbucket/al4624/git_repo/ACE-Step/test_data"

mixture_track_paths = get_all_track_paths(mixture_track_dir_path)

with ProcessPoolExecutor() as executor:
    executor.map(data_gen, mixture_track_paths)

# for inst_track_path in inst_track_paths:
    # data_gen(inst_track_path)



    


    


    

    

