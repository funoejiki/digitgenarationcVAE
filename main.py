import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

# --- カスタムCSSでFrutiger Aero風に ---
def frutiger_aero_style():
    st.markdown("""
        <style>
            html, body, [class*="css"] {
                font-family: 'Segoe UI', sans-serif;
                background: linear-gradient(to bottom right, #d2f0f7, #f2fcff);
                color: #003344;
            }
            .stButton>button {
                background-color: #a3e3ff;
                color: #003344;
                border: none;
                border-radius: 12px;
                padding: 0.5em 1.2em;
                box-shadow: 0 4px 10px rgba(0, 150, 200, 0.2);
                font-weight: bold;
            }
            .stButton>button:hover {
                background-color: #c6f1ff;
                color: #002233;
            }
            .stSelectbox label, .stSlider label {
                font-weight: bold;
                font-size: 1.1em;
            }
        </style>
    """, unsafe_allow_html=True)

# --- モデル定義 ---
class Encoder(nn.Module):
    def __init__(self, latent_dim=3, num_classes=10):
        super().__init__()
        self.label_embedding = nn.Linear(num_classes, 16)
        self.fc_hidden = nn.Linear(28 * 28 + 16, 128)
        self.fc_mu = nn.Linear(128, latent_dim)
        self.fc_logvar = nn.Linear(128, latent_dim)

    def forward(self, image, label):
        flattened_image = image.view(image.size(0), -1)
        label_one_hot = F.one_hot(label, num_classes=10).float()
        label_embedding = F.relu(self.label_embedding(label_one_hot))
        concatenated_input = torch.cat([flattened_image, label_embedding], dim=1)
        hidden = F.relu(self.fc_hidden(concatenated_input))
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=3, num_classes=10):
        super().__init__()
        self.label_embedding = nn.Linear(num_classes, 16)
        self.fc_hidden = nn.Linear(latent_dim + 16, 128)
        self.fc_out = nn.Linear(128, 28 * 28)

    def forward(self, latent_vector, label):
        label_one_hot = F.one_hot(label, num_classes=10).float()
        label_embedding = F.relu(self.label_embedding(label_one_hot))
        concatenated_latent = torch.cat([latent_vector, label_embedding], dim=1)
        hidden = F.relu(self.fc_hidden(concatenated_latent))
        output = torch.sigmoid(self.fc_out(hidden))
        return output.view(-1, 1, 28, 28)

class CVAE(nn.Module):
    def __init__(self, latent_dim=3, num_classes=10):
        super().__init__()
        self.encoder = Encoder(latent_dim, num_classes)
        self.decoder = Decoder(latent_dim, num_classes)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def forward(self, image, label):
        mu, logvar = self.encoder(image, label)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, label), mu, logvar

# --- UI ---
frutiger_aero_style()

st.title("🧊 CVAE Digit Generator")
st.markdown("Frutiger Aero 風インターフェースで数字画像を生成します")

digit = st.selectbox("🪧 生成したい数字を選んでください (0〜9)", list(range(10)))
num_images = st.slider("🖼️ 生成する画像の枚数", 1, 20, 6)

# モデルロード
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
latent_dim = 3
model = CVAE(latent_dim=latent_dim).to(device)
model.load_state_dict(torch.load("cvae.pth", map_location=device))
model.eval()

# 生成
if st.button("✨ 画像を生成"):
    z = torch.randn(num_images, latent_dim).to(device)
    labels = torch.full((num_images,), digit, dtype=torch.long, device=device)
    with torch.no_grad():
        generated = model.decoder(z, labels)

    nrow = int(np.ceil(np.sqrt(num_images)))
    ncol = int(np.ceil(num_images / nrow))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 2, nrow * 2))
    axes = np.array(axes).reshape(nrow, ncol)

    for i in range(nrow * ncol):
        ax = axes[i // ncol, i % ncol]
        if i < num_images:
            ax.imshow(generated[i].squeeze().cpu().numpy(), cmap="gray")
        ax.axis("off")

    st.pyplot(fig)
