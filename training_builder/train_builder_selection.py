def get_train_builder_class(config):
    if config['network'] == 'ResNet':
        train_builder_class = ResNetTrainBuilder
    else:
        raise NotImplementedError
    return train_builder_class
