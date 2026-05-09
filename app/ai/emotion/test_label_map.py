from app.ai.emotion.label_map import map_ravdess_to_project_label

example = "data/emotion_datasets/ravdess/Actor_01/03-01-03-01-01-01-01.wav"
print(map_ravdess_to_project_label(example))