def prepare_dataset(csv_file):
  dataset = pd.read_csv(csv_file)
  dataset = dataset.drop_duplicates(subset=dataset.columns[0])
  dataset = dataset.reset_index(drop=True)
  print("Shape of dataset from {} file: ".format(csv_file), dataset.shape)
  return dataset


def extract_diff_common(dataset_1, dataset_2):
  col = dataset_1.columns[0]
  common = list(
                set(dataset_1['{}'.format(col)].values) & set(dataset_2['{}'.format(col)].values)
                )
  extra_1 = list(
                set(dataset_1['{}'.format(col)].values) - set(dataset_2['{}'.format(col)].values)
                )
  extra_2 = list(
              set(dataset_2['{}'.format(col)].values) - set(dataset_1['{}'.format(col)].values)
                )
  print("Common images: ", len(common), 
        "\nDifferent images(dataset_1 only): ", len(extra_1),
        "\nDifferent images(dataset_2 only): ", len(extra_2))
  return common, extra_1, extra_2  