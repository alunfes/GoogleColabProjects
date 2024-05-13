import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader

# 保存した画像データのフォルダ
data_dir = "canvas_images"

# 画像データの前処理
transform = transforms.Compose([
    transforms.Grayscale(),  # グレースケールに変換
    transforms.Resize((28, 28)),  # サイズを28x28に変換
    transforms.ToTensor(),  # テンソルに変換
])

# データセットの読み込み
dataset = ImageFolder(data_dir, transform=transform)

# データローダーの作成
batch_size = 64
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# 学習用のモデルや損失関数、最適化手法などを定義し、学習を行う
# 以下は例として、ランダムなネットワークの定義と学習の一部を示したものです
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(64 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

model = Net()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 学習ループの実行
for epoch in range(num_epochs):
    for batch in dataloader:
        images, labels = batch
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
