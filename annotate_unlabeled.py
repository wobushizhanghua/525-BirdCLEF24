import pandas as pd
from pydub import AudioSegment
import os
from collections import Counter
from multiprocessing import Pool

csv_file = 'class.csv'
output_dir = './train_audio'
clip_length = 5000  # 每个音频裁剪长度（毫秒）

df = pd.read_csv(csv_file, header=None)

def process_audio(args):
    index, row = args  # 获取行索引和数据
    bird_names = row[1].split(',')
    bird_names = [bird_name.strip() for bird_name in bird_names]
    audio_name = row[0]

    counts = Counter(bird_names)

    most_common_bird, _ = counts.most_common(1)[0]

    first_occurrence_index = bird_names.index(most_common_bird)

    start_time = first_occurrence_index * clip_length
    end_time = start_time + clip_length

    audio = AudioSegment.from_ogg(f"./unlabeled_soundscapes/{audio_name}")

    clip = audio[start_time:end_time]

    bird_dir = os.path.join(output_dir, most_common_bird)

    if not os.path.exists(bird_dir):
        return

    clip_path = os.path.join(bird_dir, audio_name)
    clip.export(clip_path, format='ogg')

    return f'Audio clip for {most_common_bird} saved to {clip_path}'

if __name__ == '__main__':
    with Pool(processes=32) as pool:
        results = pool.map(process_audio, list(df.iterrows()))
        for result in results:
            print(result)
