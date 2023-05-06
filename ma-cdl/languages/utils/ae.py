import io
import os
import wandb
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch.optim import Adam
from languages.utils.cdl import CDL
from torch.utils.data import Dataset, DataLoader
from languages.utils.networks import Autoencoder

IMAGE_SIZE = (84, 84)

class AE:
    def __init__(self, output_dims, rng):
        self.model = Autoencoder(output_dims)
        try:
            state_dict = torch.load('ma-cdl/languages/history/ae.pth')
            self.model.load_state_dict(state_dict)
        except:
            self._init_hyperparams()
            self._init_wandb()
            self.loss = torch.nn.MSELoss()
            self.dataset = ImageDataset(rng, 300)
            self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
            self._train()
            self._save_model()
            
    def _init_hyperparams(self):
        self.batch_size = 64
        self.learning_rate = 5e-4
        self.num_train_epochs = 1000
            
    def _init_wandb(self):
        wandb.init(project='autoencoder', entity='ethanmclark1')
        config = wandb.config
        config.learning_rate = self.learning_rate
        config.num_train_epochs = self.num_train_epochs
            
    def _save_model(self):
        directory = 'ma-cdl/languages/history'
        filename = 'ae.pth'
        file_path = os.path.join(directory, filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        torch.save(self.model.state_dict(), file_path)
        
    # Debugging function
    @staticmethod
    def _display(regions):
        _, ax = plt.subplots()
        for idx, region in enumerate(regions):
            ax.fill(*region.exterior.xy)
            ax.text(region.centroid.x, region.centroid.y, idx, ha='center', va='center')
        plt.show()
        
    @staticmethod
    def pixelate(regions):
        _, ax = plt.subplots()
        # Remove borders
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        # Remove tick labels
        ax.set_xticks([])
        ax.set_yticks([])
        for region in regions:
            ax.plot(*region.exterior.xy)
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
        buf.seek(0)
        im = Image.open(buf).convert('L')
        pixel_array = np.array(im)
        pixel_array = np.array(Image.fromarray(pixel_array).resize(IMAGE_SIZE))
        plt.clf()
        plt.close()
        
        # Autoencoder expects a 4D tensor (batch_size, channels, height, width)
        pixel_tensor = torch.from_numpy(pixel_array).float().unsqueeze(0)
        return pixel_tensor
        
    def get_state(self, regions):
        pixel_tensor = self.pixelate(regions)
        state = self.model.get_encoded(pixel_tensor)
        return state
    
    def _train(self):
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        
        for epoch in range(self.num_train_epochs):
            for img in dataloader:
                output = self.model(img)
                loss = self.loss(img, output)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            print(f'Epoch [{epoch + 1}/{self.num_train_epochs}], Loss: {loss.item():.4f}')
                
class ImageDataset(Dataset):
    def __init__(self, rng, num_episodes):
        self.rng = rng
        self.num_episodes = num_episodes
        self.images = self.generate_images()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]
    
    def generate_images(self):
        images = []
        
        for _ in range(self.num_episodes):
            prev_num_lines = 4
            valid_lines  = set()
            done = False
            while not done: 
                action = self.rng.uniform(-1, 1, 3)
                line = CDL.get_lines_from_coeffs(action)
                valid = CDL.get_valid_lines(line)
                valid_lines.update(valid)
                regions = CDL.create_regions(list(valid_lines))
                pixel_tensor = AE.pixelate(regions)
                images.append(pixel_tensor)
                if len(valid_lines) == prev_num_lines:
                    done = True
                    continue
                prev_num_lines = len(valid_lines)
                
        return images
