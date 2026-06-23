import base64
import copy
import io
import os
import threading
from contextlib import asynccontextmanager
from typing import List

import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel

from model import create_model, quantize_model, train_model
from utils import (
    apply_hot_colormap,
    create_directories,
    delete_category_files,
    get_device,
    list_images,
    preprocess_image,
)


# ── Application state ──────────────────────────────────────────────────────────

class AppState:
    def __init__(self):
        self.categories: List[str] = ['class_1', 'class_2']
        self.trained_model = None
        self.quantized_model = None
        self.training_complete: bool = False
        self.training_status: str = None   # None | 'running' | 'complete' | 'error'
        self.training_epoch: int = 0
        self.training_total: int = 10
        self.training_msg: str = ''
        self.training_error: str = ''
        self.predictions: List[dict] = []  # {'filename', 'tensor', 'image'}
        self._lock = threading.Lock()


_s = AppState()


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    create_directories(_s.categories)
    yield


app = FastAPI(title='Image Classifier', lifespan=lifespan)


# ── Static ────────────────────────────────────────────────────────────────────

@app.get('/', include_in_schema=False)
def root():
    return FileResponse('static/index.html')


# ── State ──────────────────────────────────────────────────────────────────────

@app.get('/api/state')
def get_state():
    return {
        'categories': _s.categories,
        'training_complete': _s.training_complete,
        'training_status': _s.training_status,
        'training_epoch': _s.training_epoch,
        'training_total': _s.training_total,
        'training_msg': _s.training_msg,
        'training_error': _s.training_error,
    }


# ── Categories ────────────────────────────────────────────────────────────────

class CategoriesRequest(BaseModel):
    categories: List[str]


@app.post('/api/categories')
def set_categories(req: CategoriesRequest):
    cats = list(dict.fromkeys(c.strip() for c in req.categories if c.strip()))
    if len(cats) < 2:
        raise HTTPException(400, '2クラス以上必要です')
    changed = cats != _s.categories
    _s.categories = cats
    create_directories(cats)
    if changed:
        _s.trained_model = None
        _s.quantized_model = None
        _s.training_complete = False
        _s.predictions = []
    return {'categories': cats, 'reset': changed}


# ── Images ────────────────────────────────────────────────────────────────────

@app.get('/api/images/{category}')
def get_images(category: str):
    if category not in _s.categories:
        raise HTTPException(404, 'カテゴリが見つかりません')
    return {'category': category, 'files': list_images(category)}


@app.post('/api/upload/{category}')
async def upload_images(category: str, files: List[UploadFile] = File(...)):
    if category not in _s.categories:
        raise HTTPException(404, 'カテゴリが見つかりません')
    saved = []
    for f in files:
        if not f.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue
        content = await f.read()
        path = os.path.join('uploads', category, f.filename)
        with open(path, 'wb') as fp:
            fp.write(content)
        saved.append(f.filename)
    return {'saved': saved, 'count': len(saved)}


@app.delete('/api/images/{category}')
def delete_images(category: str):
    if category not in _s.categories:
        raise HTTPException(404, 'カテゴリが見つかりません')
    deleted = delete_category_files(category)
    _s.trained_model = None
    _s.quantized_model = None
    _s.training_complete = False
    _s.predictions = []
    return {'deleted': deleted}


@app.delete('/api/images')
def delete_all_images():
    total = sum(delete_category_files(c) for c in _s.categories)
    _s.trained_model = None
    _s.quantized_model = None
    _s.training_complete = False
    _s.predictions = []
    return {'deleted': total}


# ── Training ──────────────────────────────────────────────────────────────────

class TrainRequest(BaseModel):
    fc_only: bool = False
    epochs: int = 10


def _run_training(categories: List[str], fc_only: bool, epochs: int):
    try:
        def on_progress(epoch, total, loss):
            _s.training_epoch = epoch
            _s.training_total = total
            _s.training_msg = f'Epoch {epoch}/{total}  loss: {loss:.4f}'

        _s.training_msg = 'モデルを構築中...'
        model = create_model(num_classes=len(categories))

        _s.training_msg = '学習中...'
        trained = train_model(
            model, categories, epochs=epochs,
            fc_only=fc_only, progress_callback=on_progress,
        )

        _s.training_msg = '量子化中...'
        q_model = quantize_model(trained)

        with _s._lock:
            _s.trained_model = trained
            _s.quantized_model = q_model
            _s.training_complete = True
            _s.training_status = 'complete'
            _s.training_msg = '学習完了'

    except Exception as e:
        with _s._lock:
            _s.training_status = 'error'
            _s.training_error = str(e)
            _s.training_msg = f'エラー: {e}'


@app.post('/api/train')
def start_training(req: TrainRequest):
    if _s.training_status == 'running':
        raise HTTPException(409, '既に学習中です')
    for cat in _s.categories:
        if len(list_images(cat)) < 1:
            raise HTTPException(400, f'{cat}: 画像が 1 枚以上必要です')
    _s.training_status = 'running'
    _s.training_epoch = 0
    _s.training_total = req.epochs
    _s.training_msg = '初期化中...'
    _s.training_error = ''
    _s.training_complete = False
    threading.Thread(
        target=_run_training,
        args=(_s.categories, req.fc_only, req.epochs),
        daemon=True,
    ).start()
    return {'status': 'started'}


@app.get('/api/train/status')
def train_status():
    return {
        'status': _s.training_status,
        'epoch': _s.training_epoch,
        'total': _s.training_total,
        'msg': _s.training_msg,
        'error': _s.training_error,
    }


# ── Predict ───────────────────────────────────────────────────────────────────

@app.post('/api/predict')
async def predict(files: List[UploadFile] = File(...)):
    if not _s.training_complete:
        raise HTTPException(400, '先にモデルを学習してください')

    base_model = _s.trained_model
    device = next(base_model.parameters()).device
    infer_model = (
        _s.quantized_model
        if device.type == 'cpu' and _s.quantized_model is not None
        else base_model
    )
    if infer_model is not base_model:
        pass
    else:
        infer_model.eval()

    results = []
    predictions = []

    for f in files:
        content = await f.read()
        image = Image.open(io.BytesIO(content)).convert('RGB')

        thumb = image.copy()
        thumb.thumbnail((300, 300), Image.LANCZOS)
        buf = io.BytesIO()
        thumb.save(buf, format='PNG')
        thumb_b64 = base64.b64encode(buf.getvalue()).decode()

        tensor = preprocess_image(image)
        with torch.inference_mode():
            out = infer_model(tensor.to(device))
            probs = torch.nn.functional.softmax(out[0], dim=0).cpu().tolist()

        pred_id = len(predictions)
        results.append({
            'id': pred_id,
            'filename': f.filename,
            'thumbnail': thumb_b64,
            'probabilities': {cat: p for cat, p in zip(_s.categories, probs)},
        })
        predictions.append({
            'filename': f.filename,
            'tensor': tensor.cpu(),
            'image': image,
        })

    _s.predictions = predictions
    return {'results': results, 'categories': _s.categories}


# ── Saliency ──────────────────────────────────────────────────────────────────

@app.get('/api/saliency/{pred_id}')
def get_saliency(pred_id: int):
    if not _s.training_complete:
        raise HTTPException(400, '先にモデルを学習してください')
    if pred_id < 0 or pred_id >= len(_s.predictions):
        raise HTTPException(404, '予測データが見つかりません')

    pred = _s.predictions[pred_id]
    model = _s.trained_model
    model.eval()
    device = next(model.parameters()).device

    inp = pred['tensor'].detach().clone().to(device)
    inp.requires_grad_(True)
    out = model(inp)
    pred_class = out.argmax(dim=1).item()

    try:
        (grad,) = torch.autograd.grad(out[0, pred_class], inp)
    except Exception as e:
        raise HTTPException(500, f'勾配計算に失敗しました: {e}')

    grads = grad.abs().squeeze().cpu().numpy()
    sal = np.max(grads, axis=0)
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-8)
    sal = np.clip(sal / np.percentile(sal, 99), 0, 1)

    orig = pred['image'].resize((224, 224)).convert('RGB')
    heat = Image.fromarray(apply_hot_colormap(sal))
    blended = Image.blend(orig, heat, alpha=0.5)

    bar = np.tile(np.linspace(0, 1, 224), (16, 1))
    bar_img = Image.fromarray(apply_hot_colormap(bar))

    composite = Image.new('RGB', (224, 244), (30, 30, 30))
    composite.paste(blended, (0, 0))
    composite.paste(bar_img, (0, 226))

    buf = io.BytesIO()
    composite.save(buf, format='PNG')
    img_b64 = base64.b64encode(buf.getvalue()).decode()

    return {
        'image_b64': img_b64,
        'predicted_class': _s.categories[pred_class],
    }


# ── Model save / load ─────────────────────────────────────────────────────────

@app.get('/api/model/download')
def download_model():
    if not _s.training_complete:
        raise HTTPException(400, '先にモデルを学習してください')
    buf = io.BytesIO()
    torch.save({
        'model_state_dict': _s.trained_model.state_dict(),
        'categories': _s.categories,
    }, buf)
    buf.seek(0)
    return StreamingResponse(
        buf,
        media_type='application/octet-stream',
        headers={'Content-Disposition': 'attachment; filename="image_classifier_model.pth"'},
    )


@app.post('/api/model/load')
async def load_model(file: UploadFile = File(...)):
    content = await file.read()
    try:
        checkpoint = torch.load(io.BytesIO(content), map_location='cpu', weights_only=False)
    except Exception as e:
        raise HTTPException(400, f'ファイルの読み込みに失敗しました: {e}')

    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        categories = checkpoint.get('categories', _s.categories)
        state_dict = checkpoint['model_state_dict']
    else:
        categories = _s.categories
        state_dict = checkpoint

    try:
        model = create_model(num_classes=len(categories))
        model.load_state_dict(state_dict)
        model.eval()
        q_model = quantize_model(model)
    except Exception as e:
        raise HTTPException(400, f'モデルの復元に失敗しました: {e}')

    _s.trained_model = model
    _s.quantized_model = q_model
    _s.training_complete = True
    _s.categories = categories
    _s.predictions = []
    return {'categories': categories}


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
