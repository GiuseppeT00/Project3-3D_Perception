from typing import List, Dict
from math import sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.image_utils import ChannelDimension
from transformers.models.segformer import SegformerImageProcessor, SegformerConfig, \
    SegformerModel, SegformerDecodeHead


class MMSegmentor(nn.Module):
    def __init__(self, modals: List[str], labels: List[str], ignore_label: int,
                 image_height: int, image_width: int, device: str, use_mha: bool):
        super().__init__()

        self.__use_mha = use_mha
        self.__device = device

        size = {
            'height': image_height,
            'width': image_width
        }

        self.__image_processor = SegformerImageProcessor(size=size)

        self.__feature_extractors = {
            modal: SegformerModel.from_pretrained(
                pretrained_model_name_or_path="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
                config=self.__get_config(version='b0', labels=labels, ignore_label=ignore_label),
                ignore_mismatched_sizes=True
            )
            for modal in modals
        }
        for modal in modals:
            self.__feature_extractors[modal].to(device)

        self.__batchNorm = [
            nn.BatchNorm2d(32, affine=True),
            nn.BatchNorm2d(64, affine=True),
            nn.BatchNorm2d(160, affine=True),
            nn.BatchNorm2d(256, affine=True)
        ]
        for idx in range(len(self.__batchNorm)):
            self.__batchNorm[idx].to(device)

        '''
        self.__multi_head_attention = [
            nn.MultiheadAttention(embed_dim=1024, num_heads=4, batch_first=True),
            nn.MultiheadAttention(embed_dim=1024, num_heads=4, batch_first=True),
            nn.MultiheadAttention(embed_dim=1024, num_heads=4, batch_first=True),
            nn.MultiheadAttention(embed_dim=1024, num_heads=4, batch_first=True)
        ]
        '''

        self.__conv2d = [
            nn.Conv2d(in_channels=32 * len(modals), out_channels=32,
                      kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.Conv2d(in_channels=64 * len(modals), out_channels=64,
                      kernel_size=(3, 3), stride=(1, 1), padding='same'),
            nn.Conv2d(in_channels=160 * len(modals), out_channels=160,
                      kernel_size=(1, 1), stride=(1, 1), padding='same'),
            nn.Conv2d(in_channels=256 * len(modals), out_channels=256,
                      kernel_size=(1, 1), stride=(1, 1), padding='same')
        ]
        for idx in range(len(self.__conv2d)):
            self.__conv2d[idx].to(device)

        self.__decode = SegformerDecodeHead.from_pretrained(
            pretrained_model_name_or_path="nvidia/segformer-b0-finetuned-cityscapes-1024-1024",
            config=self.__get_config(version='b0', labels=labels, ignore_label=ignore_label),
            ignore_mismatched_sizes=True
        )
        self.__decode.to(device)

    def forward(self, raw_images: Dict[str, torch.Tensor]):
        """
        1) Starting data
            {
                'img': Tensor(B, 3, 1024, 1024),
                'depth': Tensor(B, 3, 1024, 1024),
                'event': Tensor(B, 3, 1024, 1024),
                'lidar': Tensor(B, 3, 1024, 1024)
            }
        2) Image processor
            {
                'img': Tensor(B, 3, 1024, 1024),
                'depth': Tensor(B, 3, 1024, 1024),
                'event': Tensor(B, 3, 1024, 1024),
                'lidar': Tensor(B, 3, 1024, 1024)
            }
        3) Feature extractors
            {
                'img': [Tensor(B, 32, 256, 256), Tensor(B, 64, 128, 128), Tensor(B, 160, 64, 64), Tensor(B, 256, 32, 32)],
                'depth': [Tensor(B, 32, 256, 256), Tensor(B, 64, 128, 128), Tensor(B, 160, 64, 64), Tensor(B, 256, 32, 32)],
                'event': [Tensor(B, 32, 256, 256), Tensor(B, 64, 128, 128), Tensor(B, 160, 64, 64), Tensor(B, 256, 32, 32)],
                'lidar': [Tensor(B, 32, 256, 256), Tensor(B, 64, 128, 128), Tensor(B, 160, 64, 64), Tensor(B, 256, 32, 32)]
            }
        4) Batch Normalizations
            {
                'img': [Tensor(B, 32, 256, 256), Tensor(B, 64, 128, 128), Tensor(B, 160, 64, 64), Tensor(B, 256, 32, 32)],
                'depth': [Tensor(B, 32, 256, 256), Tensor(B, 64, 128, 128), Tensor(B, 160, 64, 64), Tensor(B, 256, 32, 32)],
                'event': [Tensor(B, 32, 256, 256), Tensor(B, 64, 128, 128), Tensor(B, 160, 64, 64), Tensor(B, 256, 32, 32)],
                'lidar': [Tensor(B, 32, 256, 256), Tensor(B, 64, 128, 128), Tensor(B, 160, 64, 64), Tensor(B, 256, 32, 32)]
            }
        5) Flatten on dimensions 2 and 3
            {
                'img': [Tensor(B, 32, 65536), Tensor(B, 64, 16384), Tensor(B, 160, 4096), Tensor(B, 256, 1024)],
                'depth': [Tensor(B, 32, 65536), Tensor(B, 64, 16384), Tensor(B, 160, 4096), Tensor(B, 256, 1024)],
                'event': [Tensor(B, 32, 65536), Tensor(B, 64, 16384), Tensor(B, 160, 4096), Tensor(B, 256, 1024)],
                'lidar': [Tensor(B, 32, 65536), Tensor(B, 64, 16384), Tensor(B, 160, 4096), Tensor(B, 256, 1024)]
            }
        6) Concat on dimension 1 of vectors at same hidden state level
            [Tensor(B, 128, 65536), Tensor(B, 256, 16384), Tensor(B, 640, 4096), Tensor(B, 1024, 1024)]
        7) Multi-Head Attention, or Cross Attention
            [Tensor(B, 128, 65536), Tensor(B, 256, 16384), Tensor(B, 640, 4096), Tensor(B, 1024, 1024)]
        8) Reshape of Tensor(B, C, H*W) to Tensor(B, C, H, W)
            [Tensor(B, 128, 256, 256), Tensor(B, 256, 128, 128), Tensor(B, 640, 64, 64), Tensor(B, 1024, 32, 32)]
        9) Different Conv2d on each tensor + unsqueeze(0) for compatibility
            [Tensor(B, 32, 256, 256), Tensor(B, 64, 128, 128), Tensor(B, 160, 64, 64), Tensor(B, 256, 32, 32)]
        10) Decode
            Tensor(B, 25, 256, 256)
        11) Interpolate
            Tensor(B, 25, 1024, 1024)
        """

        # ------------------------------- Step 2 -------------------------------
        raw_images = {
            modal: self.__image_processor(raw_image, return_tensors='pt')['pixel_values']
            for modal, raw_image in raw_images.items()
        }
        for modal in raw_images.keys():
            raw_images[modal] = raw_images[modal].to(self.__device)

        required_size = raw_images['img'].shape[-2:]

        # ------------------------------- Step 3 -------------------------------
        # [1] -> to take all hidden states -> tuple of shapes [torch.Size([2, 32, 256, 256]), torch.Size([2, 64, 128, 128]), torch.Size([2, 160, 64, 64]), torch.Size([2, 256, 32, 32])]
        features = {
            modal: self.__feature_extractors[modal](raw_image, output_hidden_states=True)[1]
            for modal, raw_image in raw_images.items()
        }
        # print(f'After extr shapes: {[f.shape for f in list(features["img"])]}')

        # ------------------------------- Step 4 -------------------------------
        features = {
            modal: [
                self.__batchNorm[hidden_state_idx](feature[hidden_state_idx])
                for hidden_state_idx in range(len(feature))
            ]
            for modal, feature in features.items()
        }
        # print(f'After batch norm shapes: {[f.shape for f in list(features["img"])]}')

        # ------------------------------- Step 5 -------------------------------
        '''
        features = {
            modal: [torch.flatten(hidden_state, start_dim=2) for hidden_state in feature]
            for modal, feature in features.items()
        }
        print(f'After flatten shapes: {[f.shape for f in list(features["img"])]}')
        '''

        # ------------------------------- Step 6 -------------------------------
        features = [
            torch.cat([
                features[modal][hidden_state_idx] for modal in features.keys()
            ], dim=1)
            for hidden_state_idx in range(len(features['img']))
        ]
        # print(f'After concat shape: {[f.shape for f in features]}')

        # ------------------------------- Step 7 -------------------------------
        # nn.MultiHeadAttention will output a tuple (attention_output, averaged_attention_weights)
        # features = list(list(tuple(ao, aaw))) of shape (num_modals, batch_size)
        '''
        features = [
            self.__multi_head_attention[i](features[i], features[i], features[i], need_weights=False)[0]
            for i in range(len(features))
        ]
        print(f'After mha shape: {[f.shape for f in features]}')
        '''

        # ------------------------------- Step 8 -------------------------------
        '''
        features = [
            torch.reshape(feature, (feature.shape[0], feature.shape[1], int(sqrt(feature.shape[2])), int(sqrt(feature.shape[2]))))
            for feature in features
        ]
        print(f'After reshape shape: {[f.shape for f in features]}')
        '''

        # ------------------------------- Step 9 -------------------------------
        features = tuple([self.__conv2d[idx](features[idx]) for idx in range(len(features))])

        # print(f'After conv2d shape: {[f.shape for f in features]}')

        # ------------------------------- Step 10 -------------------------------
        logits = self.__decode(features)
        # print(f'After decoder: {logits.shape}')

        # ------------------------------- Step 11 -------------------------------
        logits = F.interpolate(
            input=logits,
            size=required_size,
            mode='bilinear',
            align_corners=False
        )
        # print(f'Final out shape: {logits.shape}')
        return logits

    def __get_config(self, version: str, labels: List[str], ignore_label: int):
        config = SegformerConfig.from_pretrained(f"nvidia/segformer-{version}-finetuned-cityscapes-1024-1024")
        config.id2label = {
            str(label_number): label_name for label_number, label_name in enumerate(labels)
        }
        config.label2id = {
            label_name: label_number for label_number, label_name in enumerate(labels)
        }
        config.semantic_loss_ignore_index = ignore_label
        return config


def get_model(dataset_cfg, device):
    return MMSegmentor(dataset_cfg['DATASET']['MODALS'],
                       dataset_cfg['DATASET']['CLASSES'],
                       dataset_cfg['DATASET']['IGNORE_LABEL'],
                       1024, 1024,
                       use_mha=False,
                       device=device)


if __name__ == '__main__':
    import yaml

    with open('../configs/deliver_dataset.yml', 'r') as f:
        dataset_cfg = yaml.load(f, Loader=yaml.SafeLoader)

    model = MMSegmentor(dataset_cfg['DATASET']['MODALS'],
                        dataset_cfg['DATASET']['CLASSES'],
                        dataset_cfg['DATASET']['IGNORE_LABEL'],
                        1024, 1024, use_mha=False, device='cpu')
    imgs_orig = {
        modal: (torch.rand((2, 3, 1042, 1042)) * 255).type(torch.uint8)
        for modal in dataset_cfg['DATASET']['MODALS']
    }
    # print(imgs_orig)
    out = model(imgs_orig)
