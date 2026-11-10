# Малый свёрточный автоэнкодер без skip connections и предобучения.
from torch import nn


class Autoencoder(nn.Module):
    # 3*128*128 → 32*8*8 → 3*128*128; 23 603 параметра.
    def __init__(self):
        super().__init__()
        # Encoder уменьшает пространственный размер и выделяет признаки текстуры.
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(24, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        )
        # Decoder увеличивает разрешение без транспонированных свёрток.
        layers = []
        for incoming, outgoing in [(32, 24), (24, 16), (16, 8), (8, 3)]:
            layers.extend([
                nn.Upsample(scale_factor=2, mode='nearest'),
                nn.Conv2d(incoming, outgoing, kernel_size=3, padding=1),
                nn.Sigmoid() if outgoing == 3 else nn.ReLU(),
            ])
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        return self.decoder(self.encoder(x))
