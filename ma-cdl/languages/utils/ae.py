import io
import os
import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from languages.utils.cdl import CDL
from languages.utils.networks import Autoencoder
from torch.utils.data import Dataset, DataLoader, random_split

IMAGE_SIZE = (64, 64)

class AE:
    def __init__(self, output_dims, rng, max_lines):
        self.model = Autoencoder(output_dims)
        try:
            state_dict = torch.load('ma-cdl/languages/history/AE.pth')
            self.model.load_state_dict(state_dict)
        except:
            self._init_hyperparams()
            self._init_wandb()
            self.loss = torch.nn.MSELoss()
            self.dataset = ImageDataset(rng, self.num_train_epochs, max_lines)
            self.optimizer = Adam(self.model.parameters(), lr=self.learning_rate)
            self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10)
            self._train()
            self._save_model()
            
    def _init_hyperparams(self):
        self.batch_size = 16
        self.learning_rate = 1e-4
        self.num_train_epochs = 1000
    
    def _init_wandb(self):
        wandb.init(project='ma-cdl', entity='ethanmclark1', name='autoencoder')
        config = wandb.config
        config.learning_rate = self.learning_rate
        config.num_train_epochs = self.num_train_epochs
            
    def _save_model(self):
        directory = 'ma-cdl/languages/history'
        filename = f'{self.__class__.__name__}.pth'
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
            ax.plot(*region.exterior.xy, linewidth=2, color='black')
        
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
        pixel_tensor = pixel_tensor.unsqueeze(0)
        state = self.model.get_encoded(pixel_tensor)
        return state
    
    def _train(self):
        # Split the dataset into training and validation sets
        train_set, val_set = random_split(self.dataset, [int(0.8 * len(self.dataset)), len(self.dataset) - int(0.8 * len(self.dataset))])
        train_loader = DataLoader(train_set, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=self.batch_size, shuffle=False)

        best_val_loss = float('inf')
        patience_counter = 0
        patience_limit = 10

        for epoch in range(self.num_train_epochs):
            # Training loop
            for img in train_loader:
                reconstructed = self.model(img)
                loss = self.loss(img, reconstructed)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Validation loop
            val_loss = 0
            with torch.no_grad():
                for img in val_loader:
                    reconstructed = self.model(img)
                    loss = self.loss(img, reconstructed)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            wandb.log({'loss': loss.item(), 'val_loss': val_loss})
            print(f'Epoch [{epoch + 1}/{self.num_train_epochs}], Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}')
            
            self.scheduler.step(val_loss)

            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience_limit:
                    print(f'Early stopping triggered at epoch {epoch + 1}')
                    break
                        
        # Choose a batch of images from the validation set
        images = next(iter(val_loader))

        # Pass the images through the autoencoder
        with torch.no_grad():
            reconstructed_images = self.model(images)

        # Plot the original images and reconstructed images side by side
        n = 5  # Number of images to display
        plt.figure(figsize=(20, 4))
        for i in range(n):
            # Display original image
            ax = plt.subplot(2, n, i + 1)
            plt.imshow(images[i].squeeze().numpy(), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            # Display reconstructed image
            ax = plt.subplot(2, n, i + 1 + n)
            plt.imshow(reconstructed_images[i].squeeze().numpy(), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        plt.show()

class ImageDataset(Dataset):
    def __init__(self, rng, num_episodes, max_lines):
        self.rng = rng
        self.num_episodes = num_episodes
        self.max_lines = max_lines
        try:
            self.images = self.load_images()
        except:
            self.images = self.generate_images()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]
    
    def load_images(self):
        images = []
        image_folder = 'ma-cdl/languages/history/images'
        image_files = sorted(os.listdir(image_folder))

        for image_file in image_files:
            img_path = os.path.join(image_folder, image_file)
            img = Image.open(img_path).convert('L')
            pixel_array = np.array(img)
            pixel_array = np.array(Image.fromarray(pixel_array).resize(IMAGE_SIZE)) / 255
            pixel_tensor = torch.from_numpy(pixel_array).float().unsqueeze(0)
            images.append(pixel_tensor)

        return images
    
    def save_image(self, img, idx):
        save_path = 'ma-cdl/languages/history/images'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        img_path = os.path.join(save_path, f'image_{idx}.png')
        img = Image.fromarray(img.squeeze().numpy().astype(np.uint8))
        img.save(img_path)

    def generate_images(self):
        images = []
        image_idx = len(images)
        for _ in range(self.num_episodes):
            done = False
            num_action = 1
            prev_num_lines = 4
            valid_lines  = set()
            while not done: 
                action = self.rng.uniform(-1, 1, 3)
                line = CDL.get_lines_from_coeffs(action)
                valid = CDL.get_valid_lines(line)
                valid_lines.update(valid)
                regions = CDL.create_regions(list(valid_lines))
                pixel_tensor = AE.pixelate(regions)
                
                images.append(pixel_tensor / 255)
                self.save_image(pixel_tensor, image_idx)
                image_idx += 1
                
                if len(valid_lines) == prev_num_lines or num_action == self.max_lines:
                    done = True
                    continue
                
                prev_num_lines = len(valid_lines)
                num_action += 1
                
        return images