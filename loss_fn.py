import hashlib
import pickle
from contextlib import nullcontext
from functools import lru_cache

import clip
import torch

import utils
from template import imagenet_templates


def hashobj(obj):
    return hashlib.sha256(pickle.dumps(obj)).hexdigest()


def get_image_prior_losses(inputs_jit):
    diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
    diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
    diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
    diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]

    loss_var_l2 = torch.norm(diff1) + torch.norm(diff2) + \
        torch.norm(diff3) + torch.norm(diff4)

    return loss_var_l2


def vgg_feat_loss(feat_1, feat_2):
    loss = 0

    loss += torch.mean((feat_1['conv4_2'] - feat_2['conv4_2']) ** 2)
    loss += torch.mean((feat_1['conv5_2'] - feat_2['conv5_2']) ** 2)

    return loss


def vgg_loss(img1, img2):
    feat_1 = utils.get_vgg_features(img1)
    feat_2 = utils.get_vgg_features(img2)
    return vgg_feat_loss(feat_1, feat_2)



class CLIPLossModel:
    def __init__(self, model='ViT-B/32', device=None, jit=False):
        super().__init__()
        if device is None:
            device = torch.device(
                'cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.device = device
        self.clip_model, self.preprocess = clip.load(model, device, jit=jit)
        for parameter in self.clip_model.parameters():
            parameter.requires_grad_(False)

    @staticmethod
    def pairwise_cosine_distance(input_a, input_b):
        normalized_input_a = torch.nn.functional.normalize(input_a)  
        normalized_input_b = torch.nn.functional.normalize(input_b)
        res = torch.mm(normalized_input_a, normalized_input_b.T)
        return res

    @lru_cache
    def get_text_features(self, text: str, aug=True):
        if aug:
            template_text = utils.compose_text_with_templates(
                text, imagenet_templates)
        else:
            template_text = [text]
        tokens = clip.tokenize(template_text).to(self.device)
        text_features = self.clip_model.encode_text(tokens).detach()
        text_features = text_features.mean(axis=0, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        return text_features

    def get_img_features(self, img):
        img_features = self.clip_model.encode_image(
            utils.clip_normalize(img, self.device))
        img_features /= (img_features.clone().norm(dim=-1, keepdim=True))
        return img_features

    @lru_cache(maxsize=32)
    def get_img_features_cached(self, img):
        return self.get_img_features(img)

    def get_img_features_auto(self, img):
        if img.requires_grad:
            return self.get_img_features(img)
        else:
            return self.get_img_features_cached(img).detach()

    @lru_cache
    # for directional clip loss
    def get_text_direction(self, target_text, source_text, aug=True, no_grad=True):
        grad_context = torch.no_grad() if no_grad else nullcontext()
        with grad_context:
            target_text_features = self.get_text_features(target_text, aug=aug)
            source_text_features = self.get_text_features(source_text, aug=aug)
            text_direction = (target_text_features - source_text_features)
            text_direction /= text_direction.norm(dim=-1, keepdim=True)
            return text_direction

    def get_features(self, obj):
        if isinstance(obj, str):
            return self.get_text_features(obj)
        else:
            assert isinstance(obj, torch.Tensor)
            return self.get_img_features_auto(obj)

    def get_direction(self, target, source):
        assert type(target) == type(source)
        if isinstance(target, str):
            return self.get_text_direction(target, source)
        else:
            assert isinstance(target, torch.Tensor)
            return self.get_image_direction(target, source)

    def get_image_direction(self, target_image, source_image):
        target_features = self.get_img_features_auto(target_image)
        source_features = self.get_img_features_auto(source_image)
        image_direction = (target_features - source_features)
        image_direction /= image_direction.clone().norm(dim=-1, keepdim=True)
        return image_direction

    def cosine_similarity_loss(self, embedding1, embedding2):
        return 1 - torch.cosine_similarity(embedding1, embedding2, dim=-1)

    def basic_loss(self, obj1, obj2, thresh=None):
        # General CLIP loss
        # works for both text and image
        feature1 = self.get_features(obj1)
        feature2 = self.get_features(obj2)
        # loss = (1 - torch.cosine_similarity(feature1, feature2, dim=-1))
        loss = self.cosine_similarity_loss(feature1, feature2)
        if thresh is not None:
            loss[loss < thresh] = 0
        return loss

    def directional_loss(
            self, 
            target1=None, 
            source1=None, 
            target2=None, 
            source2=None, 
            direction1=None, 
            direction2=None, 
            thresh=None,
            bidirectional_isi=False,
            pairwise_isi=False,
        ):
        # General CLIP direction loss
        # works for both text and image pairs
        direction1 = direction1 if direction1 is not None else self.get_direction(target1, source1)
        direction2 = direction2 if direction2 is not None else self.get_direction(target2, source2)

        if direction1.size(0) == 1 or direction2.size(0) == 1:

            if direction1.size(0) == 1 and direction2.size(0) > 1:
                direction1 = direction1.repeat(direction2.size(0), 1)
            elif direction2.size(0) == 1 and direction1.size(0) > 1:
                direction2 = direction2.repeat(direction1.size(0), 1)
            else:
                assert direction1.size(0) == 1 and direction2.size(0) == 1

            loss_dir = (
                1 - torch.cosine_similarity(direction1, direction2, dim=1))
            if thresh is not None:
                loss_dir[loss_dir < thresh] = 0

            loss_dir = loss_dir.mean()

        elif pairwise_isi: # pairwise isi, as implemented in the paper.
            assert not bidirectional_isi
            loss_dir = (
                1 - self.pairwise_cosine_distance(direction1, direction2))
            if thresh is not None:
                loss_dir[loss_dir < thresh] = 0

            loss_dir = loss_dir.mean()

        elif bidirectional_isi:  # multi to multi symetric bidirectional directional loss
            assert not pairwise_isi
            direction1_mean = torch.mean(direction1, dim=0, keepdim=True)
            direction2_mean = torch.mean(direction2, dim=0, keepdim=True)

            loss_dir = self.directional_loss(direction1=direction1_mean, direction2=direction2, thresh=thresh) + \
                self.directional_loss(direction1=direction1, direction2=direction2_mean, thresh=thresh)

            loss_dir = loss_dir / 2

        else: # multi to multi directional loss
            direction2_mean = torch.mean(direction2, dim=0, keepdim=True)
            loss_dir = self.directional_loss(direction1=direction1, direction2=direction2_mean, thresh=thresh)

        return loss_dir
