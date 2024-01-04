import argparse
import itertools
import os
import random

import torch
from torchvision.transforms import ToPILImage
from tqdm import tqdm

import utils
from model_bridge import load_adaattn_model, make_adaattn_input

# import pdb

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--style_reps_dir', type=str, required=True, help='path to source folder')
    parser.add_argument('--output_dir', type=str, required=True, help='path to target folder')
    parser.add_argument('--content_paths', nargs='+', type=str, default=[], help='paths to content file or dirs')
    parser.add_argument('--adaattn_path', type=str, default='./AdaAttN', help='path to AdaAttN model')
    parser.add_argument('--n_ens', type=int, default=None, help='number of ensemble')
    return parser.parse_args()

def build_sty_path_dict(result_dir):
    img_paths = os.listdir(result_dir)
    img_paths = [name for name in img_paths if f'-style.' in name]

    sty_path_dict = {}
    for img_path in img_paths:
        style = img_path.split('_')[0]
        if style not in sty_path_dict:
            sty_path_dict[style] = []
        sty_path_dict[style].append(os.path.join(result_dir, img_path))

    return sty_path_dict



def main(args):
    args.adaattn_path = os.path.abspath(args.adaattn_path)
    contents = {}
    for content_path in args.content_paths:
        assert os.path.exists(content_path), f'Content file {content_path} does not exist.' 
        if os.path.isfile(content_path):
            content_name = os.path.splitext(os.path.basename(content_path))[0]
            contents[content_name] = content_path
        elif os.path.isdir(content_path):
            # collect all files in the content folder, recursively
            for root, dirs, files in os.walk(content_path):
                for name in files:
                    if name.endswith('.png') or name.endswith('.jpg'):
                        content_name = os.path.splitext(name)[0]
                        contents[content_name] = os.path.join(root, name)
                    
    adaattn = load_adaattn_model(args.adaattn_path)

    @torch.no_grad()
    def multi_adaattn_ens(content_path, style_paths):
        content = utils.load_image_512(content_path)
        style_imgs = []
        for sp in style_paths:
            s = utils.load_image_512(sp)
            sstd = s.std()
            if sstd < 0.03:
                print(f'Warning: {sp} has std {sstd}, this style image will be ignored.')
            else:
                style_imgs.append(s)
                
        adaattn.set_input(make_adaattn_input(content, style_imgs))
        adaattn.forward()
        target = adaattn.cs
        return target

    sd = build_sty_path_dict(args.style_reps_dir)


    os.makedirs(args.output_dir, exist_ok=True)

    for s, c in tqdm(itertools.product(sd, contents), total=len(sd)*len(contents)):
        content_path = contents[c]

        if args.n_ens is None:
            result = multi_adaattn_ens(content_path, sd[s])
        else:
            result = multi_adaattn_ens(content_path, random.sample(sd[s], args.n_ens))

        result_path = os.path.join(args.output_dir, f'{s}_{c}.png')
        result_pil = ToPILImage()(result.squeeze().clamp(0,1))
        result_pil.save(result_path)
        
if __name__ == '__main__':
    main(get_args())

    