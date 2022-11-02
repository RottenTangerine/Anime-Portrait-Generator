import os
import time
import torch
import torchvision

from config import get_parser
from model import generator

args = get_parser()

device = 'cuda' if torch.cuda.is_available() and args.cuda else 'cpu'
G = generator.Generator(args.noise_features).to(device)

# load model
try:
    ckpt_path = os.path.join('./trained_model', args.model)
    _model = torch.load(ckpt_path)
    G.load_state_dict(_model)
    print(f'Successfully load the model {args.model}')

    # generate
    generate_id = int(time.time())
    print(f'Generating ID: {generate_id}')

    for i in range(args.gen_number):
        z = torch.randn((1, args.noise_features, 1, 1)).to(device)
        fake_img = G(z)
        os.makedirs('output', exist_ok=True)
        torchvision.utils.save_image(fake_img[0], f'output/{generate_id}_{i}.jpg')
        print(f'Successfully generate the image output/{generate_id}_{i}.jpg')
except:
    print('Fail to load the model, please check the config.py')
