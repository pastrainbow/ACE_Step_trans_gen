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



def get_all_inst_track_paths(dir_path):
    return [str(file) for file in Path(dir_path).glob('*.Instrumental.mp3') if file.is_file()]

def data_gen(track_path):
    try:
        track_num = int(Path(track_path).name.split('.')[0])

        track_name = Path(track_path).stem

        noised_track_path = os.path.join(noised_inst_track_dir_path, track_name + '.noised.mp3')
        print(noised_track_path)
        if (not os.path.exists(noised_track_path)):
            print(f"[ERROR] Noised track does not exist for track {track_path}!")
            return
        
        #we need 4 files: lyric txt, prompt txt, track mp3, and noised track mp3
        lyrics_str = "[instrumental]" #we deal with instrumental tracks
        
        with open(os.path.join(dataset_path, track_name + '_lyrics.txt'), 'w') as f:
            f.write(lyrics_str)

        genre_strs = get_genres_from_id(track_num)
        tag_str = ', '.join(genre_strs)

        with open(os.path.join(dataset_path, track_name + '_prompt.txt'), 'w') as f:
            f.write(tag_str)

        shutil.copyfile(track_path, os.path.join(dataset_path, Path(track_path).name))

        shutil.copyfile(noised_track_path, os.path.join(dataset_path, Path(noised_track_path).name))
    except Exception as e:
        print(f"[ERROR] Error generating data for track {track_path}: {e}")


inst_track_dir_path="/homes/al4624/Documents/YuE_finetune/finetune_testing_dataset/sep_audio"
noised_inst_track_dir_path="/homes/al4624/Documents/YuE_finetune/finetune_testing_dataset/noised_inst"
dataset_path="/vol/bitbucket/al4624/git_repo/ACE-Step/data"


inst_track_paths = get_all_inst_track_paths(inst_track_dir_path)
inst_track_count = len(inst_track_paths)

with ProcessPoolExecutor() as executor:
    executor.map(data_gen, inst_track_paths)

# for inst_track_path in inst_track_paths:
    # data_gen(inst_track_path)



    


    


    

    

