import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import io
import torchvision.transforms.functional as TF
import glob
from typing import List, Tuple

from torchvision.io import ImageReadMode


class FreiburgForest(Dataset):
    def __init__(self, root: str, split: str, transform, modals: List[str],
                  classes: List[str]):
        super().__init__()

        assert len(classes) > 0
        assert split in ['train', 'test'], f'Invalid split selected: {split}'
        for modal in modals:
            assert modal in ['depth_color', 'evi_color', 'ndvi_color', 'nir_color', 'nrg', 'rgb'], \
                f'Invalid modal selected: {modal}'

        self._split = split
        self._transform = transform
        self._modals = modals
        self._classes = classes
        self._n_classes = len(classes)

        self._files = sorted(glob.glob(os.path.join(*[root, split, 'rgb', '*'])))

        debug_str = f"""
{'*' * 80}
Deliver dataset correctly initialized.
Selected split: {'Train' if self._split == 'train' else 'Val'}.
Selected modals: {self._modals}.
Number of classes: {self._n_classes}.
Number of images found (for a single modal): {len(self._files)}.
{'*' * 80}\n
        """
        print(debug_str, flush=True)

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, index: int) -> Tuple[list, torch.Tensor]:
        base_path = str(self._files[index])
        sample_paths = {
            key: base_path.replace('rgb', key).replace(
                'jpg', 'jpg' if key in ['rgb', 'nrg', 'ndvi_color']
                else 'png' if key in ['nir_color', 'evi_color', 'depth_color']
                else ''
            )
            for key in self._modals
        }
        label_path = base_path.replace('img', 'GT_color')
        sample = dict()
        for key, path in sample_paths.items():
            sample[key] = self._open_img(path)
        sample['mask'] = self._open_img(label_path)

        if self._transform:
            sample = self._transform(sample)
        return list(sample.values()), sample['mask']

    def _open_img(self, file):
        return io.read_image(file)


dataset = FreiburgForest('..\\freiburg_forest_annotated', 'train', None,
                         modals=['rgb', 'depth_color', 'evi_color', 'ndvi_color', 'nir_color', 'nrg'],
                         classes=['Void', 'Obstacle', 'Trail', 'Sky', 'Grass', 'Vegetation'])
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
'''
min_shape_imgs = [1000, 1000]
min_shape_lbl = [1000, 1000]
for img, label in dataloader:
    for i in img:
        shape = i.shape[-2:]
        if shape[0] < min_shape_imgs[0]:
            min_shape_imgs[0] = shape[0]
        if shape[1] < min_shape_imgs[1]:
            min_shape_imgs[1] = shape[1]
    shape = label.shape[-2:]
    if shape[0] < min_shape_lbl[0]:
        min_shape_lbl[0] = shape[0]
    if shape[1] < min_shape_imgs[1]:
        min_shape_lbl[1] = shape[1]
    #print(label.shape)
    #print(label)  # [122, 125, 130], [74, 62, 24]
    #break
    #uniques = torch.cat((uniques, label))
    #uniques = uniques.unique()
print(f'min shape imgs: {min_shape_imgs}')
print(f'min shape lbls: {min_shape_lbl}')
min shape imgs: [450, 729]
min shape lbls: [450, 1000]
'''

uniques = torch.Tensor()

mapping = {
    [170, 170, 170]: 2,
    [0, 120, 255]: 3,
    [0, 255, 0]: 4,
    [102, 102, 51]: 5
}

for img, label in dataloader:
    label = label.permute(0, 2, 3, 1)
    # print(label.shape)
    # print(label)
    uniques = torch.cat((uniques, label.unique())).unique()

print('\n\n', uniques)
