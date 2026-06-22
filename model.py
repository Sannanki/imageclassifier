import copy
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os



class ImageDataset(Dataset):
    """カテゴリフォルダから画像を読み込む PyTorch Dataset．

    各カテゴリのインデックス番号がそのままクラスラベルになる．
    全画像を初期化時にメモリに読み込んで前処理することで、
    毎エポックのディスクI/Oと前処理のオーバーヘッドを削減する。
    """

    def __init__(self, categories, root_dir='uploads'):
        self.categories = categories
        self.root_dir = root_dir
        self.samples = []

        from utils import get_transform
        self.transform = get_transform()

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
                    image_tensor = self.transform(image)
                    self.samples.append((image_tensor, idx))
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")

    def __len__(self):
        """データセット全体の画像枚数を返す．"""
        return len(self.samples)

    def __getitem__(self, idx):
        """指定インデックスの画像テンソルとラベルを返す．"""
        return self.samples[idx]



def create_model(num_classes):
    """ImageNet 事前学習済み ResNet-18 の最終層をクラス数に合わせて差し替える．

    weights=DEFAULT を指定すると，初回呼び出し時に PyTorch Hub から
    事前学習済み重み（約44MB）を自動ダウンロードし，2回目以降はキャッシュを使用する．
    FC 層だけ差し替えることで，Backbone の特徴抽出能力をそのまま引き継ぐ．
    """
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    num_features = model.fc.in_features  # ResNet-18 は 512
    # 元の FC 層（1000クラス分類）を指定クラス数の線形層で上書き
    model.fc = nn.Linear(num_features, num_classes)
    return model


def train_model(model, categories, epochs=10, fc_only=False):
    """モデルをファインチューニングして返す．

    fc_only=True の場合は最終FC層のみ更新し，Backbone を凍結する（線形プローブ）．
    fc_only=False（デフォルト）は全層を更新する全層ファインチューニング．
    デバイス選択は CUDA → MPS（Apple Silicon）→ CPU の優先順で自動選択する．
    """
    if fc_only:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.fc.parameters():
            param.requires_grad = True

    # デバイス判定を集約関数から取得
    from utils import get_device
    device = get_device()
    model = model.to(device)

    dataset = ImageDataset(categories)
    # 総画像枚数が batch_size より少ない場合のエラーを防ぐため上限を設定（4から32へ最適化）
    batch_size = min(32, len(dataset))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    criterion = nn.CrossEntropyLoss()
    # fc_only=True の場合，requires_grad=True のパラメータ（FC層のみ）だけ最適化対象にする
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=0.001
    )

    model.train()
    for epoch in range(epochs):
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    return model


def quantize_model(model):
    """CPU推論専用の動的量子化（INT8）モデルを返す．元モデルは変更しない．

    deepcopy してから CPU に移すことで，GPU学習済みモデルのデバイスを保持したまま量子化できる．
    量子化モデルは autograd 非対応のため Saliency Map には使用不可．
    """
    cpu_model = copy.deepcopy(model).cpu()
    cpu_model.eval()
    return torch.quantization.quantize_dynamic(
        cpu_model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
    )
