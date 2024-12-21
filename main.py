import soundata

print(soundata.list_datasets())
dataset = soundata.initialize('urbansound8k', data_home='C:/Users/usama/PycharmProject/Sns/data')

dataset.download()  # download the dataset
dataset.validate()  # validate that all the expected files are there

example_clip = dataset.choice_clip()  # choose a random example clip
print(example_clip)  # see the available data
