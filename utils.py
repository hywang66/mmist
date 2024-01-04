import contextlib
import warnings
from contextlib import contextmanager
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms
from tqdm import tqdm

from template import imagenet_templates

_VGG = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def img_denormalize(image):
    mean=torch.tensor([0.485, 0.456, 0.406]).to(_device)
    std=torch.tensor([0.229, 0.224, 0.225]).to(_device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)

    image = image*std +mean
    return image

def img_normalize(image):
    mean=torch.tensor([0.485, 0.456, 0.406]).to(_device)
    std=torch.tensor([0.229, 0.224, 0.225]).to(_device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)

    image = (image-mean)/std
    return image

def clip_normalize(image, device=_device):
    image = F.interpolate(image,size=224,mode='bicubic')
    mean=torch.tensor([0.48145466, 0.4578275, 0.40821073]).to(image.device)
    std=torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(image.device)
    mean = mean.view(1,-1,1,1)
    std = std.view(1,-1,1,1)
    image = (image-mean)/std
    return image


def load_image(img_path, img_size=None):
    
    image = Image.open(img_path)
    if img_size is not None:
        image = image.resize((img_size, img_size))  # change image size to (3, img_size, img_size)
    
    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),   # this is from ImageNet dataset
                        ])   
    image = transform(image)[:3, :, :].unsqueeze(0)

    return image
def load_image2(img_path, img_height=None,img_width =None):
    
    image = Image.open(img_path)
    if img_width is not None:
        image = image.resize((img_width, img_height))  # change image size to (3, img_size, img_size)
    
    transform = transforms.Compose([
                        transforms.ToTensor(),
                        ])   

    image = transform(image)[:3, :, :].unsqueeze(0)

    return image

def im_convert(tensor):

    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze(0)    # change size to (channel, height, width)

    image = image.transpose(1,2,0)
    image = image * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))   # change into unnormalized image
    image = image.clip(0, 1)    # in the previous steps, we change PIL image(0, 255) into tensor(0.0, 1.0), so convert it

    return image

def im_convert2(tensor):
    """ Display a tensor as an image. """

    image = tensor.to("cpu").clone().detach()
    image = image.numpy().squeeze(0)    # change size to (channel, height, width)
    image = image.transpose(1,2,0)
    image = image.clip(0, 1)    # in the previous steps, we change PIL image(0, 255) into tensor(0.0, 1.0), so convert it
    return image


def get_features(image, model, layers=None):
    if layers is None:
        layers = {'0': 'conv1_1',  
                  '5': 'conv2_1',  
                  '10': 'conv3_1', 
                  '19': 'conv4_1', 
                  '21': 'conv4_2', 
                  '28': 'conv5_1',
                  '31': 'conv5_2'
                 }  
    features = {}
    x = image
    for name, layer in model._modules.items():
        x = layer(x)   
        if name in layers:
            features[layers[name]] = x
    
    return features


def get_vgg_features(image):
    # image is supposed to be a tensor of size (bs, 3, img_size, img_size) in the range [0, 1]
    if _VGG is None:
        # load vgg
        VGG = models.vgg19(pretrained=True).features
        VGG.to(_device)
        for parameter in VGG.parameters():
            parameter.requires_grad_(False)

    return get_features(img_normalize(image), VGG)


def rand_bbox(size, res):
    W = size
    H = size
    cut_w = res
    cut_h = res
    tx = np.random.randint(0,W-cut_w)
    ty = np.random.randint(0,H-cut_h)
    bbx1 = tx
    bby1 = ty
    return bbx1, bby1


def rand_sampling(args,content_image):
    bbxl=[]
    bbyl=[]
    bbx1, bby1 = rand_bbox(args.img_size, args.crop_size)
    crop_img = content_image[:,:,bby1:bby1+args.crop_size,bbx1:bbx1+args.crop_size]
    return crop_img

def rand_sampling_all(args):
    bbxl=[]
    bbyl=[]
    out = []
    for cc in range(50):
        bbx1, bby1 = rand_bbox(args.img_size, args.crop_size)
        bbxl.append(bbx1)
        bbyl.append(bby1)
    return bbxl,bbyl


def load_image_512(img_path, img_height=None,img_width =None):
    image = Image.open(img_path)

    transform = transforms.Compose([
        transforms.Resize(size=(512,)),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
                        ])   
    image = transform(image)[:3, :, :].unsqueeze(0)
    return image


def get_state():
    return {
        'cpu': torch.get_rng_state().detach(),
        'cuda': torch.cuda.get_rng_state().detach(),
    }

def set_state(fixed_state):
    torch.set_rng_state(fixed_state['cpu'])
    torch.cuda.set_rng_state(fixed_state['cuda'])


@contextmanager
def temp_rng():
    state = get_state()
    try:
        yield
    finally:
        set_state(state)


def compose_text_with_templates(text: str, templates=imagenet_templates) -> list:
    return [template.format(text) for template in templates]



class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, cut_pow=1., min_size=64, max_size=256, aug=None):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.min_size = min_size
        self.max_size = max_size
        self.aug = aug

    def forward(self, input):
        sideY, sideX = input.shape[2:4]
        min_size = self.min_size
        max_size = min(sideX, sideY, self.max_size)
        cutouts = []
        for _ in range(self.cutn):
            size = int(torch.rand([])**self.cut_pow * (max_size - min_size) + min_size)
            offsetx = torch.randint(0, sideX - size + 1, ())
            offsety = torch.randint(0, sideY - size + 1, ())
            cutout = input[:, :, offsety:offsety + size, offsetx:offsetx + size]
            result = F.interpolate(cutout, size=self.cut_size,mode='bicubic')
            if self.aug is not None:
                result = self.aug(result)
            cutouts.append(result)

        return torch.cat(cutouts)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def make_patchs(img, num_crops, crop_size, augs=['perspective']):
    aug_dict = {
        'perspective': transforms.RandomPerspective(fill=0, p=1,distortion_scale=0.5),
        'rotate': transforms.RandomRotation(degrees=180),
        'hflip': transforms.RandomHorizontalFlip(),
        'vflip': transforms.RandomVerticalFlip(),
    }
    cropper = transforms.Compose([
        transforms.RandomCrop(crop_size)
    ])

    augment = transforms.Compose([aug_dict[x] for x in augs] + [transforms.Resize(224)])

    img_proc =[]
    for ttt in img:
        for n in range(num_crops):
            target_crop = cropper(ttt.unsqueeze(0))
            target_crop = augment(target_crop)
            img_proc.append(target_crop)
    if isinstance(img, torch.Tensor):
        img_proc = torch.cat(img_proc, dim=0)
    elif isinstance(img, np.ndarray):
        img_proc = np.concatenate(img_proc, axis=0)
    else:
        raise ValueError(f'img_proc must be torch.Tensor or np.ndarray, got {type(img_proc)}')
    return img_proc

@torch.no_grad()
def get_best_ws(stylegan, clip_loss_model, style_text, mode='global_directional', content_image=None, source_text=None, num=128):

    if 'directional' in mode:
        measure = clip_loss_model.directional_loss
    elif 'basic' in mode:
        measure = lambda target_image, content_image, style_text, source_text: clip_loss_model.basic_loss(target_image, style_text)
    else:
        raise ValueError('Either "directional" or "basic" must appear in mode')

    if 'global' in mode:
        preprocess = lambda x: x
    elif 'patch' in mode:
        preprocess = partial(make_patchs, crop_size=128, num_crops=64)
    else:
        raise ValueError('Either "global" or "patch" must appear in mode')

    print('Initializing ws...')
    ws_list = []
    imgs_list = []
    
    if num == 1:
        print('forwarding stylegan...')
        z = torch.from_numpy(np.random.randn(1, stylegan.z_dim)).to('cuda')
        ws = stylegan.mapping(z, None, truncation_psi=1, truncation_cutoff=None, update_emas=False)
        return ws.detach().requires_grad_(True)
    else:
        assert num >= 16

    n_iter = num // 16

    print('forwarding stylegan...')
    for i in tqdm(range(n_iter)):
        z = torch.from_numpy(np.random.randn(16, stylegan.z_dim)).to('cuda')
        ws = stylegan.mapping(z, None, truncation_psi=1, truncation_cutoff=None, update_emas=False)
        imgs = stylegan.forward_512(ws)
        ws_list.append(ws)
        imgs_list.append(imgs)
    ws = torch.cat(ws_list, dim=0)
    imgs = torch.cat(imgs_list, dim=0)

    ws_loss = []
    for i in tqdm(range(len(imgs))):
        target_image = preprocess(imgs[i].unsqueeze(0))
        l = measure(target_image, content_image, style_text, source_text).mean()
        assert not l.requires_grad
        ws_loss.append((ws[i], l))

    ws_loss = sorted(ws_loss, key=lambda x: x[1], reverse=False)
    loss_mean = sum([x[1] for x in ws_loss]) / len(ws_loss)
    loss_median = ws_loss[len(ws_loss) // 2][1]
    print(f'loss_median: {loss_median}, loss_mean: {loss_mean}')
    index = 0
    print(f'loss_selected: {ws_loss[index][1]}')
    ws_best = ws_loss[index][0].unsqueeze(0).detach().requires_grad_(True)
    return ws_best


@torch.no_grad()
def get_best_ws_multi(
        stylegan, 
        clip_loss_model, 
        style_text=None,
        isi_patches=None, 
        bidirectional_isi=False,
        pairwise_isi=False, 
        lambda_style_patch=[1.0],
        lambda_isi_patch=[1.0],
        mode='global_directional', 
        content_image=None, 
        source_text=None, 
        num=128
    ):
    assert isi_patches is not None or style_text

    if 'directional' in mode:
        measure = clip_loss_model.directional_loss
    elif 'basic' in mode:
        raise NotImplementedError
        measure = lambda target_image, content_image, style_text, source_text: clip_loss_model.basic_loss(target_image, style_text)
    else:
        raise ValueError('Either "directional" or "basic" must appear in mode')

    if 'global' in mode:
        preprocess = lambda x: x
    elif 'patch' in mode:
        preprocess = partial(make_patchs, crop_size=256, num_crops=64)
    else:
        raise ValueError('Either "global" or "patch" must appear in mode')

    print('Initializing ws...')
    ws_list = []
    imgs_list = []

    if num == 1:
        print('forwarding stylegan...')
        z = torch.randn(1, stylegan.z_dim).to('cuda')
        ws = stylegan.mapping(z, None, truncation_psi=1, truncation_cutoff=None, update_emas=False)
        return ws.detach().requires_grad_(True)
    else:
        assert num >= 16

    n_iter = num // 16

    print('forwarding stylegan...')
    for i in tqdm(range(n_iter)):
        z = torch.randn(16, stylegan.z_dim).to('cuda')
        ws = stylegan.mapping(z, None, truncation_psi=1, truncation_cutoff=None, update_emas=False)
        imgs = stylegan.forward_512(ws)
        ws_list.append(ws)
        imgs_list.append(imgs)
    ws = torch.cat(ws_list, dim=0)
    imgs = torch.cat(imgs_list, dim=0)


    ws_loss = []
    for i in tqdm(range(len(imgs))):
        target_image = preprocess(imgs[i].unsqueeze(0))

        l_text = torch.tensor(0.0).to('cuda')
        if style_text:
            for sty_weight, st in zip(lambda_style_patch, style_text):
                l_text = l_text + sty_weight * measure(target_image, content_image, st, source_text).mean() 


        l_img = torch.tensor(0.0).to('cuda')
        if isi_patches is not None:
            for isi_weight, isi_pch in zip(lambda_isi_patch, isi_patches):
                l_img = l_img + isi_weight * measure(
                    target_image, content_image, isi_pch, content_image, 
                    bidirectional_isi=bidirectional_isi,
                    pairwise_isi=pairwise_isi
                ).mean()
            

        l = l_text + l_img

        ws_loss.append((ws[i], l))

    ws_loss = sorted(ws_loss, key=lambda x: x[1], reverse=False)

    loss_mean = sum([x[1].item() for x in ws_loss]) / len(ws_loss)
    loss_median = ws_loss[len(ws_loss) // 2][1]
    print(f'loss_median: {loss_median}, loss_mean: {loss_mean}')
    index = 0
    print(f'loss_selected: {ws_loss[index][1]}')
    ws_best = ws_loss[index][0].unsqueeze(0).detach().requires_grad_(True)
    return ws_best



def to_terms_str(exp_dict):
    indent = '    '
    s = 'terms = [\n'
    s += indent + "'(.*)',\n"
    for k in exp_dict:
        s += indent + f"'{k}',\n"
    s += indent + "'(.*)'\n"
    s += ']'
    return s

def shorten(string, len_each=2):
    length = (string.count('_') + 1) * (len_each + 1) - 1
    if len(string) <= length:
        return string
    return '_'.join([x[:len_each] for x in string.split('_')])


#----------------------------------------------------------------------------
# Symbolic assert.

try:
    symbolic_assert = torch._assert # 1.8.0a0 # pylint: disable=protected-access
except AttributeError:
    symbolic_assert = torch.Assert # 1.7.0

#----------------------------------------------------------------------------
# Context manager to temporarily suppress known warnings in torch.jit.trace().
# Note: Cannot use catch_warnings because of https://bugs.python.org/issue29672

@contextlib.contextmanager
def suppress_tracer_warnings():
    flt = ('ignore', None, torch.jit.TracerWarning, None, 0)
    warnings.filters.insert(0, flt)
    yield
    warnings.filters.remove(flt)

#----------------------------------------------------------------------------
# Assert that the shape of a tensor matches the given list of integers.
# None indicates that the size of a dimension is allowed to vary.
# Performs symbolic assertion when used in torch.jit.trace().

def assert_shape(tensor, ref_shape):
    if tensor.ndim != len(ref_shape):
        raise AssertionError(f'Wrong number of dimensions: got {tensor.ndim}, expected {len(ref_shape)}')
    for idx, (size, ref_size) in enumerate(zip(tensor.shape, ref_shape)):
        if ref_size is None:
            pass
        elif isinstance(ref_size, torch.Tensor):
            with suppress_tracer_warnings(): # as_tensor results are registered as constants
                symbolic_assert(torch.equal(torch.as_tensor(size), ref_size), f'Wrong size for dimension {idx}')
        elif isinstance(size, torch.Tensor):
            with suppress_tracer_warnings(): # as_tensor results are registered as constants
                symbolic_assert(torch.equal(size, torch.as_tensor(ref_size)), f'Wrong size for dimension {idx}: expected {ref_size}')
        elif size != ref_size:
            raise AssertionError(f'Wrong size for dimension {idx}: got {size}, expected {ref_size}')
