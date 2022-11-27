def prepare_dataset(csv_file):
  dataset = pd.read_csv(csv_file)
  dataset = dataset.drop_duplicates(subset=dataset.columns[0])
  dataset = dataset.reset_index(drop=True)
  print("Shape of dataset from {} file: ".format(csv_file), dataset.shape)
  return dataset