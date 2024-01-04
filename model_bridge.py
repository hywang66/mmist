import os
import sys
import types
from contextlib import contextmanager

import numpy as np
import torch
import torch.nn as nn

adaattn_argv = [
    'test.py', 
    '--content_path', 'datasets/contents', 
    '--style_path', 'datasets/styles', 
    '--name', 'AdaAttN', 
    '--model', 'adaattn', 
    '--dataset_mode', 'unaligned', 
    '--load_size', '512', 
    '--crop_size', '512', 
    '--image_encoder_path', 'checkpoints/vgg_normalised.pth', 
    '--gpu_ids', '0', 
    '--skip_connection_3', 
    '--shallow_layer'
]


@contextmanager
def cwdargv(path, argv=None, block_stdout=True):
    oldpwd = os.getcwd()
    path0 = sys.path[0]
    oldargv = sys.argv
    if argv is None:
        argv = sys.argv

    os.chdir(path)
    sys.argv = argv
    sys.path.remove(path0)
    sys.path.insert(0, path)
    if block_stdout:
        old_stdout = sys.stdout
        f = open(os.devnull, 'w')
        sys.stdout = f
    try:
        yield
    finally:
        os.chdir(oldpwd)
        sys.argv = oldargv
        sys.path.insert(0, path0)
        sys.path.remove(path)
        if block_stdout:
            sys.stdout = old_stdout
            f.close()


def load_adaattn_model(adaattn_path, salient=True):
    with cwdargv(adaattn_path, adaattn_argv, block_stdout=salient):
        from models import create_model
        from options.test_options import TestOptions

        opt = TestOptions().parse()  # get test options
        # hard-code some parameters
        opt.num_threads = 0   # test code only supports num_threads = 0
        opt.batch_size = 1    # test code only supports batch_size = 1
        opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
        opt.no_flip = True    # no flip; comment this line if results on flipped images are needed.
        opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.

        model = create_model(opt)      # create a model given opt.model and other options
        model.setup(opt) 
        model.eval() 

    return model


def make_adaattn_input(content, style):
    if content.ndim == 3:
        content = content.unsqueeze(0)
    if isinstance(style, torch.Tensor):
        if style.ndim == 3:
            style = style.unsqueeze(0)
    else:
        assert isinstance(style, list) and isinstance(style[0], torch.Tensor)
        style = [s.unsqueeze(0) if s.ndim == 3 else s for s in style]

    data = {
        'c': content,
        's': style,
        'name': 'temp',
    }
    return data


def load_stylegan3_model_with_ws(stylegan3_path='./stylegan3', network_pkl='./stylegan3/models/wikiart-1024-stylegan3-t-17.2Mimg.pkl', seed=None):
    with cwdargv(stylegan3_path):
        import dnnlib
        import legacy

        device = torch.device('cuda')
        with dnnlib.util.open_url(network_pkl) as f:
            G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    G.output_pooling = nn.AdaptiveAvgPool2d(512)
    
    def forward_512(self, ws): # output 512x512, [0,1] image
        img = self.synthesis(ws, update_emas=False, noise_mode='const')
        img = self.output_pooling(img)
        img = (img * 0.5 + 0.5).clamp(0, 1)
        return img

    G.forward_512 = types.MethodType(forward_512, G)
    G.eval()

    for p in G.parameters():
        p.requires_grad = False

    if seed is None:
        seed = np.random.randint(0, 2**32)
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    ws = G.mapping(z, None, truncation_psi=1, truncation_cutoff=None, update_emas=False)
    ws = ws.detach().requires_grad_(True)

    return G, ws

