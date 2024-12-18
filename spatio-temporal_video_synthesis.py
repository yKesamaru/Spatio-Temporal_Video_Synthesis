"""spatio-temporal_video_synthesis.py.

Summary:
    このスクリプトは、入力動画内の検出された人物を基に時間を圧縮し、
    複数の時間帯に存在した人物を同時に可視化する動画を生成するためのコードです。
    DAMO-YOLOを使って人物を検出し、DeepSORTアルゴリズムで追跡を行います。

    主な特徴:
    - 時間圧縮: 異なる時間に検出された人物を1つのフレームに統合。
    - DAMO-YOLO: リアルタイム物体検出モデル。
    - DeepSORT: 人物の動き追尾する。
    - 動画の時間を指定した長さに調整可能。
    - 背景画像を設定し、検出された人物を重ね合わせた新しい動画を生成。

Example:
    1. 動画や背景画像のパスを指定します（`video_path`と`background_image_path`）。
    2. スクリプトを実行すると、`output_video_path`で指定された場所に結果の動画が保存されます。
    3. 入力フレーム数に応じて出力動画の時間を調整するには、`target_duration_seconds`を設定してください。

Parameters:
    - `video_path` (str): 入力動画のパス。
    - `output_video_path` (str): 出力動画のパス。
    - `background_image_path` (str): 背景画像のパス。
    - `frame_extension_factor` (int): フレーム拡張倍率（1フレームを何倍に引き伸ばすか）。
    - `target_duration_seconds` (int): 出力動画の目標時間（秒単位）。
    - DAMO-YOLO関連:
        - `config_path` (str): DAMO-YOLOの設定ファイルパス。
        - `ckpt_path` (str): DAMO-YOLOの学習済みモデルのパス。
    - DeepSORT関連:
        - `max_age` (int): オブジェクトが検出されなくなってからも追跡を続ける最大フレーム数。
        - `n_init` (int): 追跡を確定するために必要な連続検出回数。

License:
    This script is licensed under the terms provided by the author, including modifications for your specific use case.
"""

from collections import defaultdict

import cv2
from deep_sort_realtime.deepsort_tracker import DeepSort
from tqdm import tqdm  # type: ignore

from damo_yolo.base import COCO_CLASSES, Infer
from damo_yolo.damo_internal.config.base import parse_config

# =========================================================
# 設定値
# =========================================================
frame_extension_factor = 1  # 1フレームをn倍に拡張
target_duration_seconds = 5  # 出力動画の長さをx秒に。

# =========================================================
# 入力動画・背景画像パス、モデルコンフィグ等
# =========================================================
video_path = "assets/ライブカメラ1.mp4"   # 入力動画ファイル
output_video_path = "output_video.mp4"      # 出力動画ファイル
background_image_path = "assets/background1.png"  # 背景画像

config_path = "damo_yolo/configs/damoyolo_tinynasL20_T.py"  # 設定ファイルパス
ckpt_path = "damo_yolo/pretrained_models/damoyolo_tinynasL20_T_420.pth"  # チェックポイントファイルパス

# =========================================================
# モデル初期化
# =========================================================
config = parse_config(config_path)  # 設定のパース
infer = Infer(config=config, ckpt_path=ckpt_path)  # 推論インスタンス作成
tracker = DeepSort(
    max_age=60,  # 検出されなくなった後も追跡を続けるフレーム数
    n_init=3     # トラックが「確定」するまでに必要な連続検出回数
)

# =========================================================
# 動画キャプチャの初期化
# =========================================================
cap = cv2.VideoCapture(video_path)  # 入力動画読み込み
frame_rate = cap.get(cv2.CAP_PROP_FPS)  # フレームレート取得
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 幅取得
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 高さ取得

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 出力コーデック
out = cv2.VideoWriter(output_video_path, fourcc, frame_rate, (width, height))  # 出力動画作成

background = cv2.imread(background_image_path)  # 背景画像読み込み
if background is None:
    raise FileNotFoundError("背景画像が読み込めません")

# =========================================================
# データ格納用
# =========================================================
trajectories = defaultdict(list)  # {track_id: [(f_idx, x, y, w, h), ...]}
person_images = {}  # {(track_id, f_idx): image}

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 総フレーム数取得

# =========================================================
# フレーム処理ループ
# =========================================================
frame_idx = 0
with tqdm(total=total_frames, desc="Processing video frames") as pbar:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        # 640x640にリサイズ
        resized_frame = cv2.resize(frame, (640, 640))

        # DAMO-YOLOで人物検出
        bboxes, scores, cls_inds = infer.forward(resized_frame)

        # リサイズ前のスケールに戻す
        scale_x = frame.shape[1] / 640
        scale_y = frame.shape[0] / 640
        bboxes = [
            [int(x1 * scale_x), int(y1 * scale_y), int(x2 * scale_x), int(y2 * scale_y)]
            for x1, y1, x2, y2 in bboxes
        ]

        # 人物のみ抽出
        detections = []
        for bbox, score, cls_ind in zip(bboxes, scores, cls_inds):
            if score >= 0.5 and COCO_CLASSES[int(cls_ind)] == "person":
                x1, y1, x2, y2 = bbox
                # 座標クリッピング
                x1 = max(0, min(x1, frame.shape[1] - 1))
                y1 = max(0, min(y1, frame.shape[0] - 1))
                x2 = max(0, min(x2, frame.shape[1] - 1))
                y2 = max(0, min(y2, frame.shape[0] - 1))

                w = x2 - x1
                h = y2 - y1
                detections.append(([x1, y1, w, h], score, "person"))

        # トラッカー更新
        tracks = tracker.update_tracks(detections, frame=frame)

        # 検出人物を保存
        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            x, y, w, h = track.to_tlwh()
            x, y, w, h = int(x), int(y), int(w), int(h)

            if w > 0 and h > 0:
                person_crop = frame[y:y + h, x:x + w].copy()
                person_images[(track_id, frame_idx)] = person_crop
                trajectories[track_id].append((frame_idx, x, y, w, h))

        frame_idx += 1
        pbar.update(1)

cap.release()

# =========================================================
# 時間リスケーリング修正（全員0秒から登場）
# =========================================================
target_duration = int(target_duration_seconds * frame_rate)

rescaled_trajectories = defaultdict(list)  # {new_f_idx: [(track_id, orig_f_idx, x, y, w, h), ...]} # orig_f_idxも格納

for track_id, frames in trajectories.items():
    frames_sorted = sorted(frames, key=lambda x: x[0])

    if len(frames_sorted) < 1:
        continue

    earliest_orig_fidx = frames_sorted[0][0]

    for i in range(len(frames_sorted) - 1):
        orig_f_idx, x, y, w, h = frames_sorted[i]
        orig_f_idx_next, x_next, y_next, w_next, h_next = frames_sorted[i + 1]

        shifted_f_idx = orig_f_idx - earliest_orig_fidx
        shifted_f_idx_next = orig_f_idx_next - earliest_orig_fidx

        start_new_f_idx = shifted_f_idx * frame_extension_factor
        end_new_f_idx = shifted_f_idx_next * frame_extension_factor
        total_steps = end_new_f_idx - start_new_f_idx
        if total_steps <= 0:
            continue

        for step in range(total_steps):
            alpha = step / float(total_steps)
            inter_x = int(x + (x_next - x) * alpha)
            inter_y = int(y + (y_next - y) * alpha)
            inter_w = int(w + (w_next - w) * alpha)
            inter_h = int(h + (h_next - h) * alpha)

            current_f_idx = start_new_f_idx + step
            if 0 <= current_f_idx < target_duration:
                # 補間した時点のorig_f_idxも近似
                # ここでは単純にfloorで計算（より正確には丸めてよい）
                inter_orig_fidx = int(orig_f_idx + (orig_f_idx_next - orig_f_idx) * alpha)  # 補間したorig_f_idxの近似
                rescaled_trajectories[current_f_idx].append((track_id, inter_orig_fidx, inter_x, inter_y, inter_w, inter_h))  # orig_f_idxを格納

    # 最後のフレーム処理
    if len(frames_sorted) == 1:
        orig_f_idx, x, y, w, h = frames_sorted[0]
        shifted_f_idx = orig_f_idx - earliest_orig_fidx
        for ext_i in range(frame_extension_factor):
            new_f_idx = shifted_f_idx * frame_extension_factor + ext_i
            if 0 <= new_f_idx < target_duration:
                rescaled_trajectories[new_f_idx].append((track_id, orig_f_idx, x, y, w, h))  # orig_f_idxをそのまま格納
    else:
        orig_f_idx_last, x_last, y_last, w_last, h_last = frames_sorted[-1]
        shifted_f_idx_last = orig_f_idx_last - earliest_orig_fidx
        for ext_i in range(frame_extension_factor):
            new_f_idx = shifted_f_idx_last * frame_extension_factor + ext_i
            if 0 <= new_f_idx < target_duration:
                rescaled_trajectories[new_f_idx].append((track_id, orig_f_idx_last, x_last, y_last, w_last, h_last))  # orig_f_idx_lastを格納

# =========================================================
# 動画生成ループ
# =========================================================
with tqdm(total=target_duration, desc="Generating output video") as pbar:
    for f_idx in range(target_duration):
        output_frame = background.copy()

        if f_idx in rescaled_trajectories:
            # (track_id, inter_orig_fidx, x, y, w, h) で展開
            for track_id, inter_orig_fidx, x, y, w, h in rescaled_trajectories[f_idx]:  # inter_orig_fidxを受け取り
                # inter_orig_fidxを用いてperson_imagesを取得
                if (track_id, inter_orig_fidx) in person_images:  # 補間したorig_fidxに最も近い実画像を使用
                    person_crop = person_images[(track_id, inter_orig_fidx)]
                    x1 = max(0, x)
                    y1 = max(0, y)
                    x2 = min(width, x + w)
                    y2 = min(height, y + h)

                    if person_crop is not None:
                        crop_h, crop_w = person_crop.shape[:2]
                        paste_w = x2 - x1
                        paste_h = y2 - y1
                        if paste_w <= 0 or paste_h <= 0 or crop_w <= 0 or crop_h <= 0:
                            # サイズが0以下の場合はスキップ
                            continue
                        if paste_w > 0 and paste_h > 0:
                            if paste_w != crop_w or paste_h != crop_h:
                                person_crop = cv2.resize(person_crop, (paste_w, paste_h))  # サイズを合わせる
                            output_frame[y1:y2, x1:x2] = person_crop  # 貼り付け

        resized_frame = cv2.resize(output_frame, (width, height))
        out.write(resized_frame)
        pbar.update(1)

out.release()
print("出力動画が完成しました:", output_video_path)
