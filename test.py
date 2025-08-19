from acestep.text2music_dataset import Text2MusicDataset
train_dataset = Text2MusicDataset(
    train=True,
    train_dataset_path="./zh_lora_dataset",
)
print(train_dataset.get_full_features(0))
