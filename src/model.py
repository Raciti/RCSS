import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from basic_resnet import generate_model_resnet, get_mobilenet, get_shufflenet

class SpatialMRIClassifier(nn.Module):
    def __init__(self, num_classes=6):
        super(SpatialMRIClassifier, self).__init__()
        self.model = generate_model_resnet(
            model_depth=18, 
            n_input_channels=1,
            n_classes=num_classes,
            no_max_pool=True
            )

    def forward(self, x):
        return self.model(x)

class KSpaceMRIClassifier(nn.Module):
    def __init__(self, num_classes=6):
        """
        K-Print classifier for the frequency domain (K-space).
        Takes the logarithmic Power Spectral Density (PSD) tensor as input.
        """
        super(KSpaceMRIClassifier, self).__init__()
        self.model = generate_model_resnet(
            model_depth=18,
            n_input_channels=1,     
            n_classes=num_classes,
            no_max_pool=True
        )

    def forward(self, x):
        return self.model(x)
    
    
class MobileNetMRIClassifier(nn.Module):
    def __init__(self, num_classes=6, sample_size=(182, 218, 182)):
        """
        Wrapper for 3D MobileNetV2.
        Configured for single-channel input (MRI).
        """
        super(MobileNetMRIClassifier, self).__init__()
        self.model = get_mobilenet(
            num_classes=num_classes,
            sample_size=sample_size,
            in_channels=1  
        )

    def forward(self, x):
        return self.model(x)

class ShuffleNetMRIClassifier(nn.Module):
    def __init__(self, num_classes=6, sample_size=(182, 218, 182)):
        """
        Wrapper for 3D ShuffleNetV2.
        Optimized for memory efficiency.
        """
        super(ShuffleNetMRIClassifier, self).__init__()
        self.model = get_shufflenet(
            num_classes=num_classes,
            sample_size=sample_size,
            in_channels=1  
        )

    def forward(self, x):
        return self.model(x)
    
class ConstrainedConv2d(nn.Module):
    def __init__(self, in_channels=1, out_channels=3, kernel_size=5, stride=1, padding=2):
        super(ConstrainedConv2d, self).__init__()
        self.kernel_size = kernel_size
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        nn.init.xavier_normal_(self.conv.weight)

    def forward(self, x):
        return self.conv(x)

    def apply_clipping_correction(self):
        """
        Algorithm 1: Prevents Weight Divergence by forcing the predictive filter.
        """
        with torch.no_grad():
            w = self.conv.weight
            center = self.kernel_size // 2
            
            w[:, :, center, center] = 0.0
            
            sum_surround = w.sum(dim=(2, 3), keepdim=True) + 1e-7
            w.div_(sum_surround)
            
            w[:, :, center, center] = -1.0
            
            w.clamp_(-1.0, 1.0)

class MISLnet(nn.Module):
    def __init__(self, num_classes=6):
        super(MISLnet, self).__init__()
        
        self.constrained_layer = ConstrainedConv2d(in_channels=1, out_channels=3, kernel_size=5, padding=2)
        
        self.features = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(96),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(96, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            nn.Conv2d(64, 128, kernel_size=1, stride=1),
            nn.BatchNorm2d(128),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(128, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, num_classes)
        )

    def forward(self, x):
        x = self.constrained_layer(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        out = self.classifier(x)
        return out

# MLP Mixer
class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MixerBlock(nn.Module):
    def __init__(self, num_patches, dim, tokens_mlp_dim, channels_mlp_dim, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mix = FeedForward(num_patches, tokens_mlp_dim, dropout)
        
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mix = FeedForward(dim, channels_mlp_dim, dropout)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = x.transpose(1, 2)
        x = self.token_mix(x)
        x = x.transpose(1, 2)
        x = x + residual
        
        residual = x
        x = self.norm2(x)
        x = self.channel_mix(x)
        x = x + residual
        
        return x

class KSpaceMLPMixer3D(nn.Module):
    def __init__(
        self, 
        image_size=(182, 218, 182), 
        patch_size=16,            
        in_channels=1,
        num_classes=6,
        dim=256,
        depth=8,
        tokens_mlp_dim=128,
        channels_mlp_dim=1024,
        dropout=0.1
    ):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        
        self.padded_d = math.ceil(image_size[0] / patch_size) * patch_size 
        self.padded_h = math.ceil(image_size[1] / patch_size) * patch_size 
        self.padded_w = math.ceil(image_size[2] / patch_size) * patch_size 
        
        diff_d = self.padded_d - image_size[0]
        diff_h = self.padded_h - image_size[1]
        diff_w = self.padded_w - image_size[2]
        
        self.pad_dims = (
            diff_w // 2, diff_w - diff_w // 2, 
            diff_h // 2, diff_h - diff_h // 2,  
            diff_d // 2, diff_d - diff_d // 2   
        )
        
        num_patches = (self.padded_d // patch_size) * (self.padded_h // patch_size) * (self.padded_w // patch_size)
        print(f"Inizializzazione Mixer: Volume espanso a ({self.padded_d}, {self.padded_h}, {self.padded_w}). Totale Patch: {num_patches}")
        
        self.patch_embed = nn.Conv3d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        
        self.blocks = nn.ModuleList([
            MixerBlock(num_patches, dim, tokens_mlp_dim, channels_mlp_dim, dropout) 
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        
        x = F.pad(x, self.pad_dims, mode='constant', value=0) 
        
        x = self.patch_embed(x)          
        x = x.flatten(2)                 
        x = x.transpose(1, 2)            
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        x = x.mean(dim=1)                
        x = self.head(x)                 
        return x

class ArcFaceLayer(nn.Module):
    """ Academic implementation of ArcFace for Fine-Grained Classification """
    def __init__(self, in_features, out_features, s=30.0, m=0.50):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s  
        self.m = m  
        
        self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label=None):
        cosine = F.linear(F.normalize(input, dim=1), F.normalize(self.weight, dim=1))
        
        if label is None:
            return cosine * self.s
            
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(1e-7, 1.0))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        
        one_hot = torch.zeros(cosine.size(), device=input.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s  
        
        return output
    
class KSpaceMLPMixer3D_ArcFace(nn.Module):
    """
    Identical to the baseline, but equipped with an ArcFace head for Metric Learning.
    """
    def __init__(
        self, 
        image_size=(182, 218, 182), 
        patch_size=16,        
        in_channels=1,
        num_classes=6,
        dim=256,
        depth=8,
        tokens_mlp_dim=128,
        channels_mlp_dim=1024,
        dropout=0.0,
        s=30.0,
        m=0.50
    ):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        
        self.m = m
        self.s = s
        
        self.padded_d = math.ceil(image_size[0] / patch_size) * patch_size  
        self.padded_h = math.ceil(image_size[1] / patch_size) * patch_size  
        self.padded_w = math.ceil(image_size[2] / patch_size) * patch_size  
        
        diff_d = self.padded_d - image_size[0]
        diff_h = self.padded_h - image_size[1]
        diff_w = self.padded_w - image_size[2]
        
        self.pad_dims = (
            diff_w // 2, diff_w - diff_w // 2, 
            diff_h // 2, diff_h - diff_h // 2,  
            diff_d // 2, diff_d - diff_d // 2   
        )
        
        num_patches = (self.padded_d // patch_size) * (self.padded_h // patch_size) * (self.padded_w // patch_size)
        print(f"Mixer initialization: Volume expanded to ({self.padded_d}, {self.padded_h}, {self.padded_w}). Total Patches: {num_patches}")
        
        self.patch_embed = nn.Conv3d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        
        self.blocks = nn.ModuleList([
            MixerBlock(num_patches, dim, tokens_mlp_dim, channels_mlp_dim, dropout) 
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim)
        self.head = ArcFaceLayer(in_features=dim, out_features=num_classes, s=self.s, m=self.m)

    def forward(self, x, labels=None):
        x = F.pad(x, self.pad_dims, mode='constant', value=0) 
        x = self.patch_embed(x)          
        x = x.flatten(2)                 
        x = x.transpose(1, 2)            
        
        for block in self.blocks:
            x = block(x)
            
        x = self.norm(x)
        x = x.mean(dim=1)
        
        if isinstance(self.head, nn.Identity):
            return self.head(x)
            
        return self.head(x, labels)

class MixerConditioner3D(nn.Module):
    def __init__(self, image_size=(182, 218, 182), patch_size=16, dim=256, depth=8, tokens_mlp_dim=128, channels_mlp_dim=1024, dropout=0.1):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim
        
        self.padded_d = math.ceil(image_size[0] / patch_size) * patch_size 
        self.padded_h = math.ceil(image_size[1] / patch_size) * patch_size 
        self.padded_w = math.ceil(image_size[2] / patch_size) * patch_size 
        
        diff_d = self.padded_d - image_size[0]
        diff_h = self.padded_h - image_size[1]
        diff_w = self.padded_w - image_size[2]
        
        self.pad_dims = (
            diff_w // 2, diff_w - diff_w // 2,
            diff_h // 2, diff_h - diff_h // 2,
            diff_d // 2, diff_d - diff_d // 2 
        )
        
        self.d_patches = self.padded_d // patch_size
        self.h_patches = self.padded_h // patch_size
        self.w_patches = self.padded_w // patch_size
        
        num_patches = self.d_patches * self.h_patches * self.w_patches
        
        self.patch_embed = nn.Conv3d(1, dim, kernel_size=patch_size, stride=patch_size)
        self.blocks = nn.ModuleList([
            MixerBlock(num_patches, dim, tokens_mlp_dim, channels_mlp_dim, dropout) 
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        
        self.unpatchify = nn.ConvTranspose3d(dim, 2, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = F.pad(x, self.pad_dims, mode='constant', value=0) 

        x = self.patch_embed(x)          
        x = x.flatten(2).transpose(1, 2)            
        
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        
        x = x.transpose(1, 2).view(-1, self.dim, self.d_patches, self.h_patches, self.w_patches)
        
        st = self.unpatchify(x)
        
        st = st[:, :,
                self.pad_dims[4] : self.padded_d - self.pad_dims[5],
                self.pad_dims[2] : self.padded_h - self.pad_dims[3],
                self.pad_dims[0] : self.padded_w - self.pad_dims[1]]
                
        s, t = st.chunk(2, dim=1)
        
        s = torch.tanh(s)
        return s, t
    
class CouplingLayer3D(nn.Module):
    def __init__(self, conditioner, image_size=(182, 218, 182), mask_type="even"):
        super().__init__()
        self.conditioner = conditioner
        self.mask_type = mask_type

        D, H, W = image_size
        d_grid, h_grid, w_grid = torch.meshgrid(
            torch.arange(D),
            torch.arange(H),
            torch.arange(W),
            indexing='ij'
        )
        
        mask = ((d_grid + h_grid + w_grid) % 2 == 0).float()
        mask = mask.view(1, 1, D, H, W)
        
        if self.mask_type == "odd":
            mask = 1.0 - mask
        
        self.register_buffer("mask", mask)

    def forward(self, x):
        
        mask = self.mask
        x_masked = x * self.mask
        
        s, t = self.conditioner(x_masked)
        
        s = s * (1.0 - self.mask)
        t = t * (1.0 - self.mask)
        
        z = x_masked + (1.0 - self.mask) * (x * torch.exp(s) + t)
        
        log_det_J = torch.sum(s, dim=(1, 2, 3, 4))
        return z, log_det_J
    
class KSpaceFlowDetector(nn.Module):
    def __init__(self, image_size=(182, 218, 182),num_layers=4,lambda_penalty=5.0, **mixer_kwargs):
        super().__init__()
        self.lambda_penalty = lambda_penalty
        self.layers = nn.ModuleList()

        for i in range(num_layers):
            mask_type = "even" if i % 2 == 0 else "odd"
            conditioner = MixerConditioner3D(image_size=image_size, **mixer_kwargs)
            self.layers.append(CouplingLayer3D(conditioner, image_size=image_size, mask_type=mask_type))
            
        D, H, W = image_size
        center_d, center_h, center_w = D // 2, H // 2, W // 2
        
        d_grid, h_grid, w_grid = torch.meshgrid(
            torch.arange(D),
            torch.arange(H),
            torch.arange(W),
            indexing='ij'
        )
        
        dist = torch.sqrt(
            ((d_grid - center_d) / (D/2))**2 + 
            ((h_grid - center_h) / (H/2))**2 + 
            ((w_grid - center_w) / (W/2))**2
        )
        
        freq_mask = 1.0 + lambda_penalty * dist
        self.register_buffer("freq_mask", freq_mask.view(1, 1, D, H, W))

    def forward(self, x):
        """
    x: [B, 1, 182, 218, 182], magnitude (MRI - GLOBAL)
    Returns the latent space Z and the calculated Negative Log-Likelihood.
    """
        z = x
        total_log_det_J = 0
        
        for layer in self.layers:
            z, log_det_J = layer(z)
            total_log_det_J += log_det_J
            
        
        prior_loss = torch.sum(self.freq_mask * (z ** 2) / 2.0, dim=(1, 2, 3, 4))
        
        nll_loss = prior_loss - total_log_det_J
        
        return z, nll_loss