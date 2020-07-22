from data.custom_dataset_data_loader import CustomDatasetDataLoader


def CreateDataLoader(opt, rank):
    data_loader = CustomDatasetDataLoader()
    # print(data_loader.name())
    data_loader.initialize(opt, rank)
    return data_loader
