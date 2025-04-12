import torch
import torch.nn as nn
import torch.nn.functional as F
class CNN(nn.Module):
    def __init__(self, action_dim, enable_dueling_dqn=True):
        super(CNN, self).__init__()
        self.enable_dueling_dqn = enable_dueling_dqn

        # CNN สำหรับรับ input ภาพ (shape: 1x84x84)
        # self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)  # Output: (32, 20, 20)
        # self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2) # Output: (64, 9, 9)
        # self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1) # Output: (128, 7, 7)

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),  # Output: (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), # Output: (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1), # Output: (64, 7, 7)
            nn.ReLU()
        )


        self.flattened_size = 128 * 7 * 7  # ขึ้นกับ input image size

        self.fc1 = nn.Linear(self.flattened_size, 512)

        if self.enable_dueling_dqn:
            self.fc_value = nn.Linear(512, 256)
            self.value = nn.Linear(256, 1)

            self.fc_advantages = nn.Linear(512, 256)
            self.advantages = nn.Linear(256, action_dim)
        else:
            self.output = nn.Linear(512, action_dim)

    def forward(self, x):  # x shape: (batch_size, 1, 84, 84)
        x = self.cnn(x)
        # x = F.relu(self.conv1(x))
        # self._plot_feature(x[0], title="Conv1", save_prefix="conv1")

        # # Conv Layer 2
        # x = F.relu(self.conv2(x))
        # self._plot_feature(x[0], title="Conv2", save_prefix="conv2")

        # # Conv Layer 3
        # x = F.relu(self.conv3(x))
        # self._plot_feature(x[0], title="Conv3", save_prefix="conv3")
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))

        if self.enable_dueling_dqn:
            v = F.relu(self.fc_value(x))
            V = self.value(v)

            a = F.relu(self.fc_advantages(x))
            A = self.advantages(a)

            Q = V + A - torch.mean(A, dim=1, keepdim=True)
        else:
            Q = self.output(x)
        return Q
    def _plot_feature(self, feature_maps, title="Feature Maps", num_channels=6, save_prefix="featuremap"):
        import matplotlib.pyplot as plt
        import os

        os.makedirs("feature_maps", exist_ok=True)

        feature_maps = feature_maps[:num_channels]
        num_cols = num_channels
        fig, axes = plt.subplots(1, num_cols, figsize=(3*num_cols, 3))
        fig.suptitle(title, fontsize=16)

        for i, ax in enumerate(axes):
            ax.imshow(feature_maps[i].detach().cpu().numpy(), cmap='gray')
            ax.set_title(f"Ch {i}")
            ax.axis('off')

        plt.tight_layout()
        filename = f"feature_maps/{save_prefix}_{title.replace(' ', '_')}.png"
        plt.savefig(filename)
        plt.close()