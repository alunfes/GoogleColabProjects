import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import tkinter as tk
from PIL import Image, ImageDraw
import numpy as np



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(64, 128)
        self.linear2 = nn.Linear(128, 32)  # 入力サイズを32に修正
        #self.linear3 = nn.Linear(128, 64)  # 入力サイズを32に修正
        self.linear4 = nn.Linear(32, 10)  # 入力サイズを32に修正

    def forward(self, x):
        x = F.relu(self.linear1(x))  # 線形変換と活性化関数を同時に適用
        x = F.relu(self.linear2(x))  # 線形変換と活性化関数を同時に適用
        #x = F.relu(self.linear3(x))  # 線形変換と活性化関数を同時に適用
        x = self.linear4(x)
        return x

model = Net()

# モデルの読み込み
model_path = './num.pth'
model.load_state_dict(torch.load(model_path))
model.eval()

# キャンバスの幅と高さ
canvas_width = 200
canvas_height = 200

image1 = Image.new("RGB", (canvas_width, canvas_height), (255, 255, 255))
draw = ImageDraw.Draw(image1)

# マウスのドラッグ操作のコールバック関数
def paint(event):
    x1, y1 = (event.x - 10), (event.y - 10)
    x2, y2 = (event.x + 10), (event.y + 10)
    canvas.create_oval(x1, y1, x2, y2, fill="black", width=5)
    draw.line([x1, y1, x2, y2], fill="black", width=5)

# 数字の予測を行う関数
def predict_digit():
    # キャンバス上のピクセル値を取得
    img = Image.new("L", (canvas_width, canvas_height), 255)  
    draw = ImageDraw.Draw(img)
    canvas.update()
    canvas.postscript(file="temp.ps", colormode="gray")
    img = Image.open("temp.ps")
    img = img.resize((8, 8))  # 学習時の入力サイズに合わせる
    img = img.point(lambda x: 255-x)
    img = transforms.ToTensor()(img)
    img = img.view(-1, 64)  # モデルの入力サイズに変換
    with torch.no_grad():
        prediction = model(img)
    predicted_digit = torch.argmax(prediction).item()
    result_label.config(text=f"Predicted Digit: {predicted_digit}")

# GUIのセットアップ
root = tk.Tk()
root.title("Handwritten Digit Recognition")

# キャンバスの設定
canvas = tk.Canvas(root, width=canvas_width, height=canvas_height, bg='white')
canvas.pack(expand=tk.YES, fill=tk.BOTH)
canvas.bind("<B1-Motion>", paint)

# 予測結果の表示ラベル
result_label = tk.Label(root, text="Predicted Digit: None", font=("Helvetica", 16))
result_label.pack()

# 予測ボタン
predict_button = tk.Button(root, text="Predict", command=predict_digit)
predict_button.pack()

root.mainloop()
