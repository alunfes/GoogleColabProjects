import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np
import cv2
import pyautogui
import os
import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # First convolutional layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)
        # Dropout layer to prevent overfitting
        self.dropout = nn.Dropout(0.25)
        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Apply first convolutional layer, followed by activation and pooling
        x = self.pool(torch.relu(self.conv1(x)))
        # Apply second convolutional layer, followed by activation and pooling
        x = self.pool(torch.relu(self.conv2(x)))
        # Flatten the output from convolutional layers
        x = torch.flatten(x, 1)
        # Apply dropout
        x = self.dropout(x)
        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

model = CNN()

# モデルの読み込み
model_path = './digits_cnn.pth'
model.load_state_dict(torch.load(model_path))
model.eval()

# キャンバスの幅と高さ
canvas_width = 392
canvas_height = 392

image1 = Image.new("L", (canvas_width, canvas_height), (255))  # "L"はグレースケールを表します
draw = ImageDraw.Draw(image1)

image_counter = 0

# マウスのドラッグ操作のコールバック関数
def paint(event):
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    canvas.create_oval(x1, y1, x2, y2, fill="black", width=1)
    draw.line([x1, y1, x2, y2], fill="black", width=1)

def clear_canvas():
    canvas.delete("all")  # キャンバス上のすべての描画を削除

def capture_canvas_image():
    # キャンバスの位置とサイズを取得
    canvas_x = canvas.winfo_rootx()
    canvas_y = canvas.winfo_rooty()
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    # キャンバスの範囲を指定してスクリーンショットを撮影
    screenshot = pyautogui.screenshot(region=(canvas_x, canvas_y, canvas_width, canvas_height))
    # スクリーンショットをPIL Imageに変換して保存
    screenshot.save("canvas_screenshot.png")

def save_canvas_image():
    global image_counter
    # キャンバス上の画像を保存するディレクトリ
    save_dir = "canvas_images"
    os.makedirs(save_dir, exist_ok=True)
    image_path = os.path.join(save_dir, f"canvas_image_{image_counter}.png")
    # キャンバスの位置とサイズを取得
    canvas_x = canvas.winfo_rootx()
    canvas_y = canvas.winfo_rooty()
    canvas_width = canvas.winfo_width()
    canvas_height = canvas.winfo_height()
    # キャンバスの範囲を指定してスクリーンショットを撮影
    screenshot = pyautogui.screenshot(region=(canvas_x, canvas_y, canvas_width, canvas_height))
    # スクリーンショットをPIL Imageに変換して保存
    screenshot.save(image_path)
    image_counter += 1

# 数字の予測を行う関数
def predict_digit():
    # キャンバス上のピクセル値を取得
    canvas.update()
    capture_canvas_image()
    # OpenCVで画像を読み込み、処理
    img = cv2.imread("canvas_screenshot.png", cv2.IMREAD_GRAYSCALE)  # グレースケールで読み込む
    img = cv2.resize(img, (28, 28))  # 学習時の入力サイズに合わせる
    img = cv2.bitwise_not(img)  # 白黒を反転させる
    img = img.astype(np.float32) / 255.0  # 0から1の範囲に正規化
    img = np.expand_dims(img, axis=0)  # バッチの次元を追加
    img = np.expand_dims(img, axis=0)  # カラーチャネルの次元を追加
    # モデルに画像を渡して予測
    with torch.no_grad():
        predictions = model(torch.tensor(img))
    # 予測結果を取得
    _, predicted_digit = torch.max(predictions, 1)
    # 予測結果を表示
    result_label.config(text=f"Predicted Digit: {predicted_digit.item()}")
    # バーチャートを更新
    update_bar_chart(predictions[0])

# GUIのセットアップ
root = tk.Tk()
root.title("Handwritten Digit Recognition")

# キャンバスの設定
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg='white')
canvas.pack(side=tk.LEFT, expand=tk.YES, fill=tk.BOTH)
canvas.bind("<B1-Motion>", paint)

# 出力の表示用フレーム
output_frame = tk.Frame(root)
output_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

# Clearボタンの作成と配置
clear_button = tk.Button(root, text="Clear", command=clear_canvas)
clear_button.pack(side=tk.BOTTOM)

# Saveボタンの作成と配置
save_button = tk.Button(root, text="Save", command=save_canvas_image)
save_button.pack(side=tk.BOTTOM)

# 予測結果の表示ラベル
result_label = tk.Label(root, text="Predicted Digit: None", font=("Helvetica", 16))
result_label.pack(side=tk.BOTTOM)

# 予測ボタン
predict_button = tk.Button(root, text="Predict", command=predict_digit)
predict_button.pack(side=tk.BOTTOM)

# バーチャートの設定
fig, ax = plt.subplots(figsize=(4, 4))
canvas_bar = FigureCanvasTkAgg(fig, master=output_frame)
canvas_bar.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

def update_bar_chart(predictions):
    ax.clear()
    ax.bar(range(10), predictions.detach().numpy())
    ax.set_xticks(range(10))
    ax.set_title('Prediction Probabilities')
    canvas_bar.draw()

root.mainloop()