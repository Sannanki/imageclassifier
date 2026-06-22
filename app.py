import streamlit as st
import os
from PIL import Image
import numpy as np
import io
from utils import create_directories, save_uploaded_file, delete_category_files, list_images, preprocess_image, get_device, get_transform


@st.cache_data(ttl=2)
def _cached_list_images(category):
    """os.listdir をキャッシュして，ページ再実行ごとのネットワークI/Oを削減する．
    ttl=2 秒で自動失効し，アップロード・削除の直後は最新状態に更新される．
    """
    return list_images(category)

st.set_page_config(
    page_title="Image Classifier",
    layout="wide",
    menu_items={
        'Get Help': None,
        'Report a bug': None,
        'About': None,
    }
)

# --- セッションステート初期化 ---
# Streamlit はページ遷移・操作ごとにスクリプト全体を再実行するため，
# 状態保持が必要な変数はすべて st.session_state で管理する．
if 'trained_model' not in st.session_state:
    st.session_state.trained_model = None
if 'training_complete' not in st.session_state:
    st.session_state.training_complete = False
if 'last_prediction_image' not in st.session_state:
    st.session_state.last_prediction_image = None
if 'last_prediction_tensor' not in st.session_state:
    st.session_state.last_prediction_tensor = None
if 'last_prediction_results' not in st.session_state:
    st.session_state.last_prediction_results = None
if 'quantized_model' not in st.session_state:
    st.session_state.quantized_model = None
# Predict ページの推論結果キャッシュ（ページ遷移後も表示を維持するため）
if 'prediction_cache' not in st.session_state:
    st.session_state.prediction_cache = []
# サイドバーのページ選択（key で session_state と連動させることでプログラムから遷移できる）
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Upload Training Images"
# ユーザーが設定した分類クラス名リスト（デフォルト2クラス）
if 'categories' not in st.session_state:
    st.session_state.categories = ['class_1', 'class_2']
# Upload ページで表示する入力欄の数（確定前の編集中クラス数）
if 'num_edit_classes' not in st.session_state:
    st.session_state.num_edit_classes = len(st.session_state.categories)

MIN_IMAGES_PER_CLASS = 1


def apply_hot_colormap(saliency):
    """正規化済み [0, 1] の 2D 配列を hot カラーマップの RGB 画像配列に変換する．

    matplotlib の 'hot' カラーマップを numpy のみで再現し，matplotlib への依存を排除．
    - 0.00〜0.33: 黒 → 赤（R のみ上昇）
    - 0.33〜0.67: 赤 → 黄（G が上昇）
    - 0.67〜1.00: 黄 → 白（B が上昇）
    """
    r = np.clip(saliency * 3,     0, 1)
    g = np.clip(saliency * 3 - 1, 0, 1)
    b = np.clip(saliency * 3 - 2, 0, 1)
    return (np.stack([r, g, b], axis=2) * 255).astype(np.uint8)


def _reset_model():
    """学習済みモデルと予測結果をすべてリセットする．"""
    st.session_state.trained_model = None
    st.session_state.quantized_model = None
    st.session_state.training_complete = False
    st.session_state.last_prediction_image = None
    st.session_state.last_prediction_tensor = None
    st.session_state.last_prediction_results = None
    st.session_state.prediction_cache = []


def main():
    """アプリのエントリーポイント．ページ選択に応じて各関数を呼び出す．"""
    st.title("Image Classification System")

    # 起動時の初期化（PyTorchなどの重いライブラリを初回だけプリロードしてキャッシュ）
    if 'preloaded' not in st.session_state:
        with st.spinner("システムの初期化中（PyTorchなどのライブラリを読み込んでキャッシュしています）..."):
            import torch
            from utils import get_device, get_transform
            # 内部でのインポートとキャッシュの構築を実行
            get_device()
            get_transform()
            # 他の主要な処理モジュールもプリロードしておく
            from model import create_model, train_model, quantize_model
        st.session_state.preloaded = True
        st.rerun()

    # カテゴリが変わったときだけディレクトリを作成する
    # Box等のネットワークドライブでは os.makedirs がレイテンシを持つため毎回呼ぶと遅延が生じる
    if st.session_state.get('_dirs_created_for') != st.session_state.categories:
        create_directories(st.session_state.categories)
        st.session_state._dirs_created_for = list(st.session_state.categories)

    # 起動直後に1回だけトーストを表示して「サーバーに繋がった」ことをユーザーに伝える
    if 'boot_notified' not in st.session_state:
        st.session_state.boot_notified = True
        st.toast("アプリが起動しました", icon="✅")

    # サイドバーにシステム状態を常時表示
    with st.sidebar:
        st.divider()
        st.caption("システム状態")
        classes_str = ', '.join(st.session_state.categories)
        st.caption(f"クラス: {classes_str}")
        if st.session_state.training_complete:
            st.success("モデル: 学習済み", icon="🟢")
        else:
            st.warning("モデル: 未学習", icon="🟡")

    page = st.sidebar.selectbox(
        "Choose a page",
        ["Upload Training Images", "Train Model", "Predict", "Saliency Map Analysis",
         "Save Model", "Load Model"],
        key="current_page"
    )

    if page == "Upload Training Images":
        upload_training_images()
    elif page == "Train Model":
        train_page()
    elif page == "Predict":
        predict_page()
    elif page == "Saliency Map Analysis":
        shap_analysis_page()
    elif page == "Save Model":
        save_model_page()
    else:
        load_model_page()


def upload_training_images():
    """クラス名の設定と訓練画像のアップロードを行うページ．

    クラス名編集 UI と画像アップロード UI の2段構成になっている．
    クラス名を確定するまでアップロードセクションは既存クラスで表示される．
    """
    st.header("Upload Training Images")

    # --- クラス名管理 UI ---
    st.subheader("クラス名の設定")
    st.write("分類するクラス名を入力してください（最低2クラス必要です）．")

    if st.session_state.num_edit_classes < 2:
        st.session_state.num_edit_classes = 2

    for i in range(st.session_state.num_edit_classes):
        key = f"edit_cat_{i}"
        # st.text_input にキーを渡す前にセッションステートへ初期値を設定する．
        # value= 引数でなくセッションステートを使うことで，ページ再実行時に
        # ユーザーの入力内容が上書きされるのを防ぐ．
        if key not in st.session_state:
            if i < len(st.session_state.categories):
                st.session_state[key] = st.session_state.categories[i]
            else:
                st.session_state[key] = f"class_{i + 1}"
        st.text_input(f"クラス {i + 1}", key=key)

    col_add, col_confirm = st.columns([1, 2])
    with col_add:
        if st.button("＋ クラスを追加"):
            st.session_state.num_edit_classes += 1
            # st.rerun() で即座に再描画して新しい入力欄を表示する
            st.rerun()
    with col_confirm:
        if st.button("クラス名を確定する", type="primary"):
            new_cats = []
            seen = set()
            for i in range(st.session_state.num_edit_classes):
                name = st.session_state.get(f"edit_cat_{i}", f"class_{i + 1}").strip()
                # 空文字と重複クラス名を除外する
                if name and name not in seen:
                    new_cats.append(name)
                    seen.add(name)
            if len(new_cats) < 2:
                st.error("2クラス以上を入力してください．")
            else:
                # クラス構成が変わった場合は学習済みモデルをリセット
                # （旧モデルの出力次元と新クラス数が一致しなくなるため）
                if new_cats != st.session_state.categories:
                    _reset_model()
                st.session_state.categories = new_cats
                st.session_state.num_edit_classes = len(new_cats)
                create_directories(new_cats)
                st.success(f"クラスを確定しました: {', '.join(new_cats)}")

    st.info(f"現在のクラス: **{', '.join(st.session_state.categories)}**")

    st.divider()

    # --- 画像アップロード UI ---
    st.subheader("訓練画像のアップロード")
    st.write("各クラスに1枚以上の画像をアップロードしてください（クラスごとに異なる枚数でも可）．")

    for category in st.session_state.categories:
        existing = _cached_list_images(category)

        label = f"「{category}」の画像"
        if existing:
            label += f"（保存済み: {len(existing)} 枚）"

        col_head, col_del = st.columns([4, 1])
        col_head.subheader(label)
        if existing and col_del.button("🗑️ 削除", key=f"del_{category}", help=f"「{category}」の画像をすべて削除"):
            deleted = delete_category_files(category)
            _cached_list_images.clear()
            _reset_model()
            st.success(f"「{category}」の画像 {deleted} 枚を削除しました（モデルをリセットしました）")
            st.rerun()

        uploaded_files = st.file_uploader(
            f"画像を選択してください（{category}）",
            type=['png', 'jpg', 'jpeg'],
            accept_multiple_files=True,
            key=f"upload_{category}"
        )
        if uploaded_files:
            for file in uploaded_files:
                save_uploaded_file(file, category)
            _cached_list_images.clear()
            st.success(f"{category}: {len(uploaded_files)} 枚追加しました（合計 {len(existing) + len(uploaded_files)} 枚）")

    # 全クラス一括削除
    st.divider()
    total = sum(len(_cached_list_images(c)) for c in st.session_state.categories)
    if total > 0 and st.button("🗑️ 全クラスの画像をまとめて削除", type="secondary"):
        deleted_total = sum(delete_category_files(c) for c in st.session_state.categories)
        _cached_list_images.clear()
        _reset_model()
        st.success(f"全クラスの画像 {deleted_total} 枚を削除しました（モデルをリセットしました）")
        st.rerun()


def train_page():
    """訓練画像の枚数確認とモデル学習を行うページ．"""
    st.header("Train Model")

    categories = st.session_state.categories

    # 全クラスの画像枚数を確認し，不足があれば学習ボタンを表示しない
    all_ready = True
    counts = {}
    for category in categories:
        files = _cached_list_images(category)
        counts[category] = len(files)
        if not os.path.exists(os.path.join('uploads', category)):
            all_ready = False
            st.error(f"{category} のフォルダが見つかりません．先に「Upload Training Images」でクラスを確定してください．")
        elif len(files) < MIN_IMAGES_PER_CLASS:
            all_ready = False
            st.error(f"{category}: 画像が {MIN_IMAGES_PER_CLASS} 枚以上必要です（現在: {len(files)} 枚）")

    if counts:
        summary = "　".join([f"{cat}: {n}枚" for cat, n in counts.items()])
        st.info(f"現在のアップロード枚数 — {summary}")

    if all_ready:
        st.write("**学習方式を選んでください**")
        st.caption("全層: 精度が出やすいが時間がかかる　／　最終層のみ: 高速・過学習しにくいが精度は劣る場合あり")
        col_full, col_fc = st.columns(2)
        with col_full:
            full_clicked = st.button("全層トレーニング", use_container_width=True)
        with col_fc:
            fc_clicked = st.button("最終層のみトレーニング", use_container_width=True)

        if full_clicked or fc_clicked:
            fc_only = fc_clicked
            label = "最終層のみ" if fc_only else "全層"
            with st.status(f"学習中（{label}）...", expanded=True) as status:
                st.write("① ライブラリを読み込み中...（初回のみ時間がかかります）")
                from model import create_model, train_model, quantize_model
                st.write("② モデルを学習中...（初回実行時は事前学習済みモデルのダウンロードが発生します・約44MB）")
                model = create_model(num_classes=len(categories))
                st.session_state.trained_model = train_model(model, categories, fc_only=fc_only)
                st.session_state.training_complete = True

                st.write("② CPU推論用に量子化中...")
                st.session_state.quantized_model = quantize_model(st.session_state.trained_model)

                status.update(label=f"学習完了！（{label}ファインチューニング）", state="complete", expanded=False)


def predict_page():
    """画像をアップロードしてクラス分類を実行するページ．

    ① アップロード時に全ファイルをまとめて処理し prediction_cache に保存する．
    ② 結果はキャッシュから表示するため，Saliency Map ページに遷移して戻っても消えない．
    ③ 新しいファイルをアップロードするたびにキャッシュを更新する．
    """
    st.header("Predict Image Class")

    categories = st.session_state.categories

    if not st.session_state.training_complete:
        st.warning("先にモデルの学習を完了してください！")
        return

    uploaded_files = st.file_uploader(
        "予測する画像を選択（複数可）",
        type=['png', 'jpg', 'jpeg'],
        accept_multiple_files=True
    )

    # ファイルがアップロードされた場合のみ再推論してキャッシュを更新する
    if uploaded_files:
        import torch

        base_model = st.session_state.trained_model
        device = next(base_model.parameters()).device
        if device.type == 'cpu' and st.session_state.quantized_model is not None:
            infer_model = st.session_state.quantized_model
            device_label = "cpu（量子化INT8）"
        else:
            infer_model = base_model
            infer_model.eval()
            device_label = str(device)

        new_cache = []
        # フェーズ1: 全枚数分のレイアウトとサムネイルを先に表示する
        items = []
        for uploaded_file in uploaded_files:
            st.divider()
            col_img, col_result = st.columns([1, 2])
            image = Image.open(uploaded_file).convert('RGB')
            thumb = image.copy()
            thumb.thumbnail((300, 300), Image.LANCZOS)
            col_img.image(thumb, caption=uploaded_file.name, width=200)
            placeholder = col_result.empty()
            placeholder.caption("推論中...")
            items.append({
                'image': image,
                'thumb': thumb,
                'filename': uploaded_file.name,
                'placeholder': placeholder,
            })

        # フェーズ2: 全画像の推論を st.status 内でまとめて実行し，結果をリストに収集する
        # プレースホルダの更新はここでは行わない（status 内での逐次更新が部分表示の原因）
        with st.status(f"{len(uploaded_files)} 枚を推論中...", expanded=True) as status:
            for i, item in enumerate(items):
                st.write(f"推論中: {item['filename']}（{i + 1}/{len(uploaded_files)}）")
                input_tensor = preprocess_image(item['image']).to(device)
                with torch.inference_mode():
                    outputs = infer_model(input_tensor)
                    item['probabilities'] = torch.nn.functional.softmax(outputs[0], dim=0).cpu()
                item['tensor'] = input_tensor.cpu()
            status.update(
                label=f"{len(uploaded_files)} 枚の処理完了（{device_label}）",
                state="complete", expanded=False
            )

        # フェーズ3: status 完了後に各プレースホルダを結果で一括更新する
        # status の外で実行することで，1画像分の全カテゴリが揃ってから表示される
        for item in items:
            with item['placeholder'].container():
                for prob, cat in zip(item['probabilities'], categories):
                    prob_pct = float(prob) * 100
                    st.write(f"{cat}: {prob_pct:.2f}%")
                    st.progress(prob_pct / 100)

        new_cache = [
            {
                'filename': item['filename'],
                'image': item['thumb'],
                'tensor': item['tensor'],
                'probabilities': item['probabilities'],
            }
            for item in items
        ]
        st.session_state.prediction_cache = new_cache

    # キャッシュから結果を表示（ページ遷移後に戻っても消えない）
    cache = st.session_state.prediction_cache
    if not cache:
        return

    if not uploaded_files:
        st.info("前回の推論結果を表示しています．新しい画像をアップロードすると更新されます．")
        for i, result in enumerate(cache):
            st.divider()
            col_img, col_result = st.columns([1, 2])
            col_img.image(result['image'], caption=result['filename'], width=200)
            with col_result:
                for prob, cat in zip(result['probabilities'], categories):
                    prob_pct = float(prob) * 100
                    st.write(f"{cat}: {prob_pct:.2f}%")
                    st.progress(prob_pct / 100)

    # Saliency Map 送信ボタンはキャッシュ表示の下に一覧で並べる
    # （推論ループ中にボタンを作ると再実行時にキーが衝突するため分離する）
    if cache:
        st.divider()
        st.caption("Saliency Mapで分析する画像を選択してください")
        cols = st.columns(len(cache))
        for i, (result, col) in enumerate(zip(cache, cols)):
            if col.button(result['filename'], key=f"saliency_btn_{i}", use_container_width=True):
                st.session_state.last_prediction_image = result['image']
                st.session_state.last_prediction_tensor = result['tensor']
                st.session_state.last_prediction_results = result['probabilities']
                # current_page を更新してから rerun することでサイドバーも連動して遷移する
                st.session_state.current_page = "Saliency Map Analysis"
                st.rerun()


def shap_analysis_page():
    """Gradient-based Saliency Map を計算・表示するページ．

    厳密な SHAP 値ではなく，入力テンソルに対する出力勾配の絶対値を
    重要度マップとして使用する（Simonyan et al., 2013）．
    matplotlib を使わず PIL + numpy のみで描画する．
    """
    st.header("Saliency Map Analysis")
    st.write("この機能は予測結果に対して、どの部分が予測に影響を与えたかを可視化します。")

    categories = st.session_state.categories

    if not st.session_state.training_complete:
        st.warning("先にモデルの学習を完了してください！")
        return

    if st.session_state.last_prediction_image is None:
        st.warning("先に Predict ページで予測を実行してください！")
        return

    import torch

    st.subheader("前回の予測結果")
    col1, col2 = st.columns(2)

    with col1:
        st.image(st.session_state.last_prediction_image, caption="Analyzed Image", width=300)

        st.write("Prediction probabilities:")
        for prob, category in zip(st.session_state.last_prediction_results, categories):
            prob_percentage = float(prob) * 100
            st.write(f"{category}: {prob_percentage:.2f}%")

    with col2:
        st.subheader("Gradient-based Saliency Map")

        with st.status("特徴重要度を計算中...", expanded=True) as status:
            st.write("① モデル準備中...")
            model = st.session_state.trained_model
            model.eval()
            device = next(model.parameters()).device
            # session_state には CPU テンソルで保存されているため，モデルのデバイスに移す
            input_tensor = st.session_state.last_prediction_tensor.detach().clone().to(device)
            input_tensor.requires_grad_(True)

            st.write("② 順伝播（Forward pass）...")
            output = model(input_tensor)
            predicted_class = output.argmax(dim=1).item()

            st.write("③ 勾配計算（Backward pass）...")
            # autograd.grad で入力テンソルの勾配だけを直接計算する．
            # .backward() と違いモデルパラメータの勾配を計算しないため高速．
            try:
                (saliency_grad,) = torch.autograd.grad(
                    output[0, predicted_class], input_tensor
                )
            except Exception as e:
                status.update(label="エラー", state="error")
                st.error(f"勾配の計算に失敗しました．再度 Predict ページで予測を実行してください．（{e}）")
                return

            gradients = saliency_grad.abs().squeeze().cpu().numpy()

            st.write("④ ヒートマップ生成...")
            # RGB 3チャネルの勾配から最大値を取り，2D の重要度マップに集約する
            saliency_map = np.max(gradients, axis=0)
            saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-8)
            # 外れ値（ノイズ的な高輝点）の影響を抑えるため 99 パーセンタイルで上限クリップ
            percentile_99 = np.percentile(saliency_map, 99)
            saliency_map = np.clip(saliency_map / percentile_99, 0, 1)

            # --- PIL による重ね合わせ描画（matplotlib 不使用）---
            orig_resized = st.session_state.last_prediction_image.resize((224, 224)).convert('RGB')
            heat_img = Image.fromarray(apply_hot_colormap(saliency_map))
            blended = Image.blend(orig_resized, heat_img, alpha=0.5)

            bar_arr = np.tile(np.linspace(0, 1, 224), (16, 1))
            bar_img = Image.fromarray(apply_hot_colormap(bar_arr))

            composite = Image.new('RGB', (224, 244), (30, 30, 30))
            composite.paste(blended, (0, 0))
            composite.paste(bar_img, (0, 226))

        # 画像レンダリング完了後に「計算完了」へ更新する
        # status.update を st.image より前に呼ぶと，画像転送前に完了表示が出てしまう
        st.image(composite,
                 caption=f"Feature Importance: '{categories[predicted_class]}'",
                 width=400)
        st.caption("← 影響小　　　　　　　影響大 →")
        status.update(label="計算完了", state="complete", expanded=False)

        st.info("赤/黄/白の領域は予測に最も影響を与えた部分を示しています。黒い領域は影響が少ない部分です。")


def save_model_page():
    """学習済みモデルをクラス情報とともにダウンロードするページ．

    state_dict だけでなく categories リストも一緒に保存することで，
    Load Model 時にクラス名・クラス数を自動復元できる．
    """
    st.header("Save Model")
    st.write("トレーニング済みモデルをローカルPCに保存します。")

    if not st.session_state.training_complete:
        st.warning("先にモデルの学習を完了してください！")
        return

    import torch

    buffer = io.BytesIO()
    # クラス名リストを一緒に保存することで，ロード時にクラス数・名称を自動復元する
    torch.save({
        'model_state_dict': st.session_state.trained_model.state_dict(),
        'categories': st.session_state.categories
    }, buffer)
    buffer.seek(0)

    st.download_button(
        label="Download Model",
        data=buffer,
        file_name="image_classifier_model.pth",
        mime="application/octet-stream"
    )

    st.success(f"モデルファイルをダウンロードできます。クラス: {', '.join(st.session_state.categories)}")


def load_model_page():
    """保存済みモデルファイルを読み込むページ．

    新フォーマット（dict 形式・カテゴリ情報付き）と
    旧フォーマット（state_dict のみ）の両方に対応する．
    """
    st.header("Load Model")
    st.write("保存されたモデルファイルをアップロードして、モデルを再現します。")

    uploaded_file = st.file_uploader("モデルファイル (.pth) を選択", type=['pth'])

    if uploaded_file:
        try:
            with st.status("モデルを読み込み中...", expanded=True) as status:
                st.write("① ライブラリを読み込み中...（初回のみ時間がかかります）")
                import torch
                from model import create_model, quantize_model

                st.write("② チェックポイントファイルを解析中...")
                buffer = io.BytesIO(uploaded_file.read())
                checkpoint = torch.load(buffer, map_location=torch.device('cpu'), weights_only=False)

                # 新フォーマット（categories 付き dict）と旧フォーマット（state_dict のみ）を判別
                if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                    categories = checkpoint.get('categories', st.session_state.categories)
                    state_dict = checkpoint['model_state_dict']
                else:
                    categories = st.session_state.categories
                    state_dict = checkpoint

                st.write(f"③ モデルを構築中...（クラス: {', '.join(categories)}）")
                model = create_model(num_classes=len(categories))
                model.load_state_dict(state_dict)
                model.eval()

                st.write("④ CPU推論用に量子化中...")
                quantized = quantize_model(model)

                st.session_state.trained_model = model
                st.session_state.quantized_model = quantized
                st.session_state.training_complete = True
                st.session_state.categories = categories
                st.session_state.num_edit_classes = len(categories)
                st.session_state.last_prediction_image = None
                st.session_state.last_prediction_tensor = None
                st.session_state.last_prediction_results = None
                st.session_state.prediction_cache = []

                status.update(
                    label=f"読み込み完了（クラス: {', '.join(categories)}）",
                    state="complete", expanded=False
                )

            st.info("'Predict' または 'Saliency Map Analysis' ページに移動してください。")

        except Exception as e:
            st.error(f"モデルの読み込みに失敗しました: {str(e)}")
            st.info("正しいモデルファイル (.pth) をアップロードしてください。")


if __name__ == "__main__":
    main()
