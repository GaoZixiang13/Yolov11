import torch


def load_model(model, model_path):
    print('Load weights {}.'.format(model_path))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path, map_location=device)

    # print(model_dict.keys())
    # print(pretrained_dict.keys())
    # print(pretrained_dict)
    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
    # model_dict.update(pretrained_dict)
    # print(pretrained_dict)
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)