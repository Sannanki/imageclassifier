import os
from PIL import Image

_IMG_TRANSFORM = None


def get_transform():
    """前処理用トランスフォーム（transforms.Compose）を遅延ロードし、キャッシュして返す．"""
    global _IMG_TRANSFORM
    if _IMG_TRANSFORM is None:
        import torchvision.transforms as transforms
        _IMG_TRANSFORM = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # ImageNet 統計値で正規化：事前学習済み重みと入力分布を合わせるために必須
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    return _IMG_TRANSFORM


def get_device():
    """利用可能な最適なデバイス（CUDA, MPS, CPU）を返す．"""
    import torch
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')




def create_directories(categories):
    """各カテゴリの画像保存先ディレクトリを作成する．

    exist_ok=True により，既存フォルダへの再呼び出しはエラーにならない．
    ページ遷移のたびに main() から呼ばれるが，何度実行しても同じ結果になる．
    """
    os.makedirs('uploads', exist_ok=True)
    for category in categories:
        os.makedirs(os.path.join('uploads', category), exist_ok=True)


def delete_category_files(category):
    """指定カテゴリの全画像ファイルを削除する．フォルダ自体は残す．"""
    category_path = os.path.join('uploads', category)
    if not os.path.exists(category_path):
        return 0
    deleted = 0
    for f in os.listdir(category_path):
        if not f.startswith('.') and f.lower().endswith(('.png', '.jpg', '.jpeg')):
            os.remove(os.path.join(category_path, f))
            deleted += 1
    return deleted


def list_images(category):
    """カテゴリフォルダ内の画像ファイル名リストを返す．

    隠しファイルと非画像ファイルを除外する．フォルダが存在しない場合は空リストを返す．
    呼び出し側で st.cache_data と組み合わせてネットワークドライブの遅延を回避する．
    """
    category_path = os.path.join('uploads', category)
    if not os.path.exists(category_path):
        return []
    return [f for f in os.listdir(category_path)
            if not f.startswith('.') and f.lower().endswith(('.png', '.jpg', '.jpeg'))]


def save_uploaded_file(uploaded_file, category):
    """Streamlit の UploadedFile をカテゴリフォルダに保存する．

    getbuffer() はゼロコピーで bytes-like object を返すため，
    read() より効率的にバイナリ書き込みができる．
    """
    save_path = os.path.join('uploads', category, uploaded_file.name)
    with open(save_path, 'wb') as f:
        f.write(uploaded_file.getbuffer())


def preprocess_image(image):
    """PIL Image をモデル入力用テンソルに変換する．

    - Resize(224, 224): ResNet-18 の標準入力サイズ
    - Normalize: ImageNet の平均・分散で正規化（事前学習済み重みと合わせるため必須）
    - unsqueeze(0): バッチ次元を追加して (1, 3, 224, 224) に整形
    """
    return get_transform()(image).unsqueeze(0)

