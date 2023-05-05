import io
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch.optim import Adam
from itertools import product
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
            self.loss = torch.nn.MSELoss()
            self.dataset = ImageDataset(rng, 10000)
            self.optimizer = Adam(self.model.parameters(), lr=0.001)
            self._train()
            self._save_model()
            
    def _save_model(self):
        directory = 'ma-cdl/languages/history'
        filename = 'ae.pth'
        file_path = os.path.join(directory, filename)
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        torch.save(self.model.state_dict(), file_path)
        
    @staticmethod
    def pixelate(regions):
        _, ax = plt.subplots()
        for region in regions:
            ax.plot(*region.exterior.xy)
            
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
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
    
    def _train(self, batch_size=64, num_epochs=1000):
        dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=True)
        
        for epoch in range(num_epochs):
            for img in dataloader:
                output = self.model(img)
                loss = self.loss(img, output)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            print(f'Epoch [{epoch + 1}/{200}], Loss: {loss.item():.4f}')
                
class ImageDataset(Dataset):
    def __init__(self, rng, num_images):
        self.rng = rng
        self.num_images = num_images
        self.images = self.generate_images()

    def generate_images(self):
        images = []
        for num_lines, _ in product(15, range(self.num_images)):
            action = self.rng.uniform(-1, 1, num_lines*3)
            lines = CDL.get_lines_from_coeffs(action)
            regions = CDL.create_regions(lines)
            pixel_tensor = AE.pixelate(regions)
            images.append(pixel_tensor)
        return images

    def __len__(self):
        return self.num_images

    def __getitem__(self, idx):
        return self.images[idx]
