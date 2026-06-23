import copy
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os


class ImageDataset(Dataset):
    """カテゴリフォルダから画像を読み込む PyTorch Dataset．

    PIL Image をメモリに保持し，__getitem__ で transform を適用する．
    これにより毎エポック異なるランダム拡張が適用され，過学習が抑制される．
    """

    def __init__(self, categories, root_dir='uploads', transform=None):
        self.samples = []  # list of (PIL.Image, label_idx)
        self.transform = transform

        for idx, category in enumerate(categories):
            category_path = os.path.join(root_dir, category)
            if not os.path.exists(category_path):
                continue
            files = [f for f in os.listdir(category_path)
                     if not f.startswith('.') and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            for f in files:
                img_path = os.path.join(category_path, f)
                try:
                    image = Image.open(img_path).convert('RGB')
                    self.samples.append((image, idx))
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image, label = self.samples[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def create_model(num_classes):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def train_model(model, categories, epochs=10, fc_only=False, progress_callback=None):
    """モデルをファインチューニングして返す．

    fc_only=True: FC層のみ更新（線形プローブ）．Backbone 凍結で高速・過学習しにくい．
    fc_only=False: 全層更新．Backbone に小さい lr，FC に大きい lr を設定（差分学習率）．
    両モードともクラス不均衡補正・CosineAnnealing スケジューラ・データ拡張を適用する．
    """
    from utils import TRAIN_TRANSFORM, get_device

    device = get_device()
    model = model.to(device)

    dataset = ImageDataset(categories, transform=TRAIN_TRANSFORM)
    batch_size = min(32, len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # クラス不均衡補正：枚数が少ないクラスの損失を大きく重みづけする
    class_counts = [0] * len(categories)
    for _, label in dataset.samples:
        class_counts[label] += 1
    total = sum(class_counts)
    weights = torch.tensor(
        [total / (len(categories) * max(c, 1)) for c in class_counts],
        dtype=torch.float, device=device
    )
    criterion = nn.CrossEntropyLoss(weight=weights)

    if fc_only:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True
        optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
    else:
        # 差分学習率：Backbone は事前学習済みのため小さい lr，FC は大きい lr
        backbone_params = [p for name, p in model.named_parameters() if 'fc' not in name]
        optimizer = torch.optim.Adam([
            {'params': backbone_params,       'lr': 1e-4},
            {'params': model.fc.parameters(), 'lr': 1e-3},
        ])

    # CosineAnnealing: エポックが進むにつれて lr をなめらかに減衰させる
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        n_batches = 0
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        if progress_callback is not None:
            progress_callback(epoch + 1, epochs, epoch_loss / max(n_batches, 1))

    return model


def quantize_model(model):
    """CPU推論専用の動的量子化（INT8）モデルを返す．元モデルは変更しない．"""
    cpu_model = copy.deepcopy(model).cpu()
    cpu_model.eval()
    return torch.quantization.quantize_dynamic(
        cpu_model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
    )
