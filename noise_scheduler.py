import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

import matplotlib.pyplot as plt
from torchvision import transforms

class NoiseScheduler():
    def __init__(self, beta_start, beta_end, timesteps, batch_size, img_channels=1):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps
        self.batch_size = batch_size
        self.img_channels = img_channels
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def beta_scheduler(self):
        self.betas = torch.linspace(self.beta_start, self.beta_end, self.timesteps)
        self.alphas = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alphas_cumprod[:-1]])
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.betas = self.betas.to(self.device)
        self.alphas_cumprod = self.alphas_cumprod.to(self.device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(self.device)

        self.posterior_variance = self.betas*(1-self.alphas_cumprod_prev)/(1.0 - self.alphas_cumprod)
    
    def get_index_at_t(self, t, vals, x_shape):
        batch_size = t.shape[0]

        out = vals.cpu().gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,)*(len(x_shape) -1))).to(t.device)
        
    def forward_diffusion(self, x_0, t):

        x_0 = x_0.to(self.device)
        noise = torch.randn_like(x_0).to(self.device)
        self.sqrt_alphas_cumprod_t = self.get_index_at_t(t, self.sqrt_alphas_cumprod, x_0.shape).to(self.device)
        self.sqrt_one_minus_alphas_cumprod_t = self.get_index_at_t(t, self.sqrt_one_minus_alphas_cumprod, x_0.shape).to(self.device)

        return self.sqrt_alphas_cumprod_t*x_0 + self.sqrt_one_minus_alphas_cumprod_t*noise, noise
    
    def show(self, image):
        reverse_transform = transforms.Compose([
            transforms.Lambda(lambda x: (x+1)/2),
            transforms.Lambda(lambda x: x.permute(1,2,0)),
            transforms.Lambda(lambda x: x*255.),
            transforms.Lambda(lambda x: x.numpy().astype(np.uint8)),
            transforms.ToPILImage()
        ])

        if(len(image.shape) == 4):
            image = image[0,:,:,:]

        plt.imshow(reverse_transform(image))

    @torch.no_grad()
    def sample_timestep(self, x, t, model):
        self.betas_t = self.get_index_at_t(t, self.betas, x.shape)
        self.sqrt_one_minus_alphas_cumprod_t = self.get_index_at_t(t, self.sqrt_one_minus_alphas_cumprod, x.shape)
        self.sqrt_recip_alphas_t = self.get_index_at_t(t, self.sqrt_recip_alphas, x.shape)

        model_mean = self.sqrt_recip_alphas_t * (x - self.betas_t*model(x, t) / self.sqrt_one_minus_alphas_cumprod_t)

        self.posterior_variance_t = self.get_index_at_t(t, self.posterior_variance, x.shape)

        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(self.posterior_variance_t)*noise
        
    @torch.no_grad()
    def sample_plot_image(self, IMG_SIZE, model):
        img_size = IMG_SIZE
        img = torch.randn((1,self.img_channels,img_size, img_size), device = self.device)
        plt.figure(figsize=(15,15))
        num_images = 10
        stepsize = int(self.timesteps/num_images)

        for i in range(0, self.timesteps)[::-1]:
            t = torch.full((1,), i , device = self.device, dtype=torch.long)
            img = self.sample_timestep(img, t, model)
            if i%stepsize == 0:
                plt.subplot(1, num_images, int((i/stepsize)+1))
                self.show(img.detach().cpu())
                plt.axis("off")

        plt.show()


