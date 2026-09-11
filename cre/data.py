"""Use supplied splits; never create a new random split of GPID-22."""
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode


def image_transform(training):
    spatial = ([transforms.RandomResizedCrop(256, scale=(.2, 1.),
                                            interpolation=InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip()] if training else
               [transforms.Resize(256, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(256)])
    return transforms.Compose(spatial + [transforms.ToTensor(),
                                         transforms.Normalize((.5,) * 3, (.5,) * 3)])


class TwoViews:
    def __init__(self):
        self.transform = image_transform(True)

    def __call__(self, image):
        return self.transform(image), self.transform(image)


class ManifestDataset(Dataset):
    """UTF-8 manifest: relative/path.jpg integer_label, one sample per line."""
    def __init__(self, root, manifest, transform):
        self.root = Path(root)
        self.transform = transform
        self.samples = []
        with open(manifest, encoding='utf-8') as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                path, label = line.strip().rsplit(maxsplit=1)
                label = int(label)
                if label < 0:
                    raise ValueError('Negative label at manifest line {}'.format(number))
                self.samples.append((path, label))
        if not self.samples:
            raise ValueError('Empty manifest: {}'.format(manifest))
        self.targets = [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(self.root / path) as image:
            return self.transform(image.convert('RGB')), label


def build_dataset(root, manifest=None, training=True, pretraining=False):
    transform = TwoViews() if pretraining else image_transform(training)
    if manifest:
        return ManifestDataset(root, manifest, transform)
    return datasets.ImageFolder(root, transform=transform)
