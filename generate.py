import os
import time
import torch
import torchvision

from config import get_parser
from model import generator


def gen_img(args, _gen, device):
    z = torch.randn((1, args.noise_features, 1, 1)).to(device)
    fake_img = _gen(z)
    return fake_img


if __name__ == '__main__':
    # load model
    try:
        args = get_parser()

        device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'
        G = generator.Generator(args.noise_features).to(device)

        ckpt_path = os.path.join('./trained_model', args.model)
        _model = torch.load(ckpt_path)
        G.load_state_dict(_model)
        print(f'Successfully load the model {args.model}')

        # generate
        generate_id = int(time.time())
        print(f'Generating ID: {generate_id}')

        os.makedirs('output', exist_ok=True)
        img = torch.cat([torch.cat([gen_img(args, G, device) for _ in range(4)], dim=-1) for _ in range(4)], dim=-2)
        torchvision.utils.save_image(img, f'output/{generate_id}.jpg')
        print(f'Successfully generate the image output/{generate_id}.jpg')
    except:
        print('Fail to load the model, please check the config.py')
