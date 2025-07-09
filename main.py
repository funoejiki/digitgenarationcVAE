import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import io

# ----- モデル定義 -----
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, 1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 13 * 13, 128),
            nn.ReLU(),
            nn.Linear(128, 10)  # MNISTなどの10クラス分類を想定
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ----- デバイス設定 -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ----- モデルロード -----
model = SimpleCNN().to(device)

# 例: モデル全体（構造 + 重み）を保存した .pth の場合（weights_only=False）
model.load_state_dict(torch.load("model.pth", map_location=device))  # weights_only=False
model.eval()

# ----- 入力変換 -----
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ----- Streamlit アプリ -----
st.title("手書き数字分類器")

uploaded_file = st.file_uploader("画像をアップロードしてください（28x28の手書き数字画像）", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="アップロードされた画像", use_column_width=True)

    # 前処理
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()

    st.write(f"予測された数字: **{predicted_class}**")
