import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import lightning as L
import torch.nn.functional as F
from torchmetrics.classification import MultilabelAUROC, MultilabelAveragePrecision
from sklearn.metrics import classification_report
import sys,json,h5py
from functools import partial
from scipy.signal import stft, butter, filtfilt, istft,get_window
from sklearn.metrics import r2_score
import math
import time

from .ViT_MAE_24ch import Model as autoencoder
from .ViT_MAE_24ch import unpatchify
from preprocessing.spectrogram_preprocessing import spectrogram_preprocessing_24ch
from postprocessing.spectrogram_postprocessing import spectrogram_postprocessing_24ch


# Load saved indices for reconstruct ECG
with open("path/of/mean_std", "r") as f:
    split_info_spec = json.load(f)
mean = split_info_spec["mean"]
std = split_info_spec["std"]
data_path_spec = 'path/of/spectrograms'
with h5py.File(data_path_spec, "r") as f1:
    original_spec_size = f1.attrs['spectrogram_shape'] 

def torch_istft_custom(Zxx: torch.Tensor,fs=500,nperseg=None,noverlap=None,nfft=None,input_onesided=True,
                       window='hann',freq_axis=-2,time_axis=-1,scaling='spectrum',boundary=True):
    Zxx = Zxx.type(torch.cfloat)      
    freq_axis = int(freq_axis)
    time_axis = int(time_axis)
    nseg = Zxx.shape[time_axis]
    
    if input_onesided:
        n_default = 2 * (Zxx.shape[freq_axis] - 1)
    else:
        n_default = Zxx.shape[freq_axis]
    
    if nperseg is None:
        nperseg = n_default
    else:
        nperseg = int(nperseg)
        if nperseg < 1:
            raise ValueError('nperseg must be a positive integer')

    if nfft is None:
        if (input_onesided) and (nperseg == n_default + 1):
            # Odd nperseg, no FFT padding
            nfft = nperseg
        else:
            nfft = n_default
    elif nfft < nperseg:
        raise ValueError('nfft must be greater than or equal to nperseg.')
    else:
        nfft = int(nfft)

    if noverlap is None:
        noverlap = nperseg//2
    else:
        noverlap = int(noverlap)
    if noverlap >= nperseg:
        raise ValueError('noverlap must be less than nperseg.')
    nstep = nperseg - noverlap

    if time_axis != Zxx.ndim-1 or freq_axis != Zxx.ndim-2:
        # Turn negative indices to positive for the call to transpose
        if freq_axis < 0:
            freq_axis = Zxx.ndim + freq_axis
        if time_axis < 0:
            time_axis = Zxx.ndim + time_axis
        zouter = list(range(Zxx.ndim))
        for ax in sorted([time_axis, freq_axis], reverse=True):
            zouter.pop(ax)
        permute_order = zouter+[freq_axis, time_axis]
        Zxx = Zxx.permute(*permute_order)
    
    if isinstance(window, str) or isinstance(window, tuple):
        win_np = get_window(window, nperseg)
        win = torch.tensor(win_np, dtype=torch.float32, device=Zxx.device)
    else:
        win = torch.tensor(window, dtype=torch.float32, device=Zxx.device)
    
    if input_onesided:
        xsubs = torch.fft.irfft(Zxx, n=nfft, dim=-2)[..., :nperseg, :]
    else:
        xsubs = torch.fft.ifft(Zxx, n=nfft, dim=-2)[..., :nperseg, :].real
    
    outputlength = nperseg + (nseg - 1) * nstep
    
    output_shape = list(Zxx.shape[:-2]) + [outputlength]
    x = torch.zeros(output_shape, dtype=xsubs.dtype, device=Zxx.device)
    norm = torch.zeros(outputlength, dtype=xsubs.dtype, device=Zxx.device)
    
    if scaling == 'spectrum':
        xsubs = xsubs * win.sum()
    elif scaling == 'psd':
        xsubs = xsubs * torch.sqrt(torch.tensor(fs * (win ** 2).sum(), dtype=xsubs.dtype, device=Zxx.device))
    else:
        raise ValueError(f"Parameter scaling={scaling} not supported.")
    
    for ii in range(nseg):
        start = ii * nstep
        end = start + nperseg
        x[..., start:end] += xsubs[..., ii] * win
        norm[start:end] += win ** 2

    if boundary:
        half = nperseg // 2
        x = x[..., half:-half]
        norm = norm[half:-half]
    
    norm_mask = norm > 1e-10
    if (~norm_mask).any():
        warnings.warn("NOLA condition failed, ISTFT may not be perfectly invertible.")
    norm = torch.where(norm_mask, norm, torch.ones_like(norm))
    x /= norm

    if input_onesided:
        x = x.real

    if x.ndim > 1 and time_axis != Zxx.ndim - 1:
        if freq_axis < time_axis:
            time_axis -= 1
        x = x.moveaxis(-1, time_axis)
    
    time = torch.arange(x.shape[0], device=x.device) / float(fs)

    return time,x

def torch_istft_fast(Zxx: torch.Tensor, fs=500, nperseg=None, noverlap=None, nfft=None,
                     input_onesided=True, window='hann', freq_axis=-2, time_axis=-1,
                     scaling='spectrum', boundary=True):
    import warnings
    Zxx = Zxx.type(torch.cfloat)
    freq_axis = int(freq_axis)
    time_axis = int(time_axis)
    nseg = Zxx.shape[time_axis]
    
    if input_onesided:
        n_default = 2 * (Zxx.shape[freq_axis] - 1)
    else:
        n_default = Zxx.shape[freq_axis]
    
    nperseg = nperseg or n_default
    nfft = nfft or n_default
    noverlap = noverlap if noverlap is not None else nperseg // 2
    nstep = nperseg - noverlap
    outputlength = nperseg + (nseg - 1) * nstep

    if freq_axis < 0: freq_axis += Zxx.ndim
    if time_axis < 0: time_axis += Zxx.ndim
    zouter = list(range(Zxx.ndim))
    for ax in sorted([time_axis, freq_axis], reverse=True): zouter.pop(ax)
    Zxx = Zxx.permute(*zouter, freq_axis, time_axis)

    # window
    if isinstance(window, str):
        from scipy.signal import get_window
        window = torch.tensor(get_window(window, nperseg), dtype=torch.float32, device=Zxx.device)
    else:
        window = torch.tensor(window, dtype=torch.float32, device=Zxx.device)
    
    # ISTFT
    if input_onesided:
        xsubs = torch.fft.irfft(Zxx, n=nfft, dim=-2)[..., :nperseg, :]
    else:
        xsubs = torch.fft.ifft(Zxx, n=nfft, dim=-2)[..., :nperseg, :].real

    if scaling == 'spectrum':
        xsubs *= window.sum()
    elif scaling == 'psd':
        xsubs *= torch.sqrt(fs * (window**2).sum())
    else:
        raise ValueError(f"scaling={scaling} not supported.")

    # window reshape for broadcasting
    window = window.view(*((1,) * (xsubs.ndim - 2)), -1, 1)
    xsubs = xsubs * window

    idx = torch.arange(nseg, device=Zxx.device) * nstep
    frame_idx = torch.arange(nperseg, device=Zxx.device)
    full_idx = idx.view(-1, 1) + frame_idx.view(1, -1)  # [nseg, nperseg]

    batch_shape = xsubs.shape[:-2]
    flat_batch = int(torch.prod(torch.tensor(batch_shape)).item())
    xsubs_flat = xsubs.reshape(flat_batch, nperseg, nseg)  # [B, nperseg, nseg]
    full_idx = full_idx.T.expand(flat_batch, -1, -1)  # [B, nperseg, nseg]

    x_out = torch.zeros(flat_batch, outputlength, device=Zxx.device)
    norm_out = torch.zeros(outputlength, device=Zxx.device)
    
    x_out.scatter_add_(1, full_idx.reshape(flat_batch, -1), xsubs_flat.reshape(flat_batch, -1))
    norm_out.scatter_add_(0, full_idx[0].reshape(-1), (window.squeeze()**2).repeat(nseg))

    x_out = x_out.view(*batch_shape, outputlength)

    # boundary crop
    if boundary:
        half = nperseg // 2
        x_out = x_out[..., half:-half]
        norm_out = norm_out[half:-half]

    norm_mask = norm_out > 1e-10
    if (~norm_mask).any():
        warnings.warn("NOLA condition failed, ISTFT may not be perfectly invertible.")
    norm_out = torch.where(norm_mask, norm_out, torch.ones_like(norm_out))
    x_out /= norm_out

    if input_onesided:
        x_out = x_out.real

    if x_out.ndim > 1 and time_axis != Zxx.ndim - 1:
        if freq_axis < time_axis:
            time_axis -= 1
        x_out = x_out.moveaxis(-1, time_axis)

    time = torch.arange(x_out.shape[0], device=x_out.device) / float(fs)
    return time, x_out
def _padding(downsample, kernel_size):
    """
    Calculates padding needed for the convolutional layers based on the downsample factor and kernel size.

    Args:
        downsample (int): The downsampling factor (stride of the convolution).
        kernel_size (int): The kernel size used in the convolution.

    Returns:
        int: Padding value calculated for the convolution.
    """
    return max(0, int(np.floor((kernel_size - downsample + 1) / 2)))


def _downsample(n_samples_in, n_samples_out):
    """
    Calculates the downsampling factor for a given input and output size.

    Args:
        n_samples_in (int): Number of input samples.
        n_samples_out (int): Number of output samples.

    Returns:
        int: Downsample factor.

    Raises:
        ValueError: If the number of samples does not decrease or downsampling is not by an integer factor.
    """
    downsample = int(n_samples_in // n_samples_out)
    if downsample < 1:
        raise ValueError("Number of samples should always decrease")
    if n_samples_in % n_samples_out != 0:
        raise ValueError("Downsampling must be by integer factor")
    return downsample


class ResBlock1d(nn.Module):
    """
    A residual block for 1D convolutional networks.

    This block contains two convolutional layers with skip connections, batch normalization, ReLU activations, 
    and dropout.
    """
    def __init__(self, n_filters_in, n_filters_out, downsample, kernel_size, dropout_rate):
        """
        Initializes the residual block.

        Args:
            n_filters_in (int): Number of input filters.
            n_filters_out (int): Number of output filters.
            downsample (int): Downsampling factor (stride).
            kernel_size (int): Size of the convolution kernel.
            dropout_rate (float): Dropout rate to apply after convolution layers.
        """
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("Kernel size must be odd")

        padding1 = _padding(1, kernel_size)
        self.conv1 = nn.Conv1d(n_filters_in, n_filters_out, kernel_size, padding=padding1, bias=True)
        self.bn1 = nn.BatchNorm1d(n_filters_out)
        self.relu = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)

        padding2 = _padding(downsample, kernel_size)
        self.conv2 = nn.Conv1d(n_filters_out, n_filters_out, kernel_size, stride=downsample,
                               padding=padding2, bias=True)
        self.bn2 = nn.BatchNorm1d(n_filters_out)
        self.dropout2 = nn.Dropout(dropout_rate)

        skip_layers = []
        if downsample > 1:
            skip_layers.append(nn.MaxPool1d(downsample, stride=downsample))
        if n_filters_in != n_filters_out:
            skip_layers.append(nn.Conv1d(n_filters_in, n_filters_out, 1, bias=True))
        self.skip_connection = nn.Sequential(*skip_layers) if skip_layers else None

    def forward(self, x, y):
        """
        Forward pass through the residual block.

        Args:
            x (torch.Tensor): Input tensor to the block.
            y (torch.Tensor): Skip connection input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through convolution and skip connections.
            torch.Tensor: The updated skip connection tensor.
        """
        y = self.skip_connection(y) if self.skip_connection else y
        x = self.dropout1(self.relu(self.bn1(self.conv1(x))))
        x = self.conv2(x)
        x += y
        y = x
        x = self.dropout2(self.relu(self.bn2(x)))
        return x, y


class ResNet1d(nn.Module):
    """
    A 1D ResNet model with multiple residual blocks for feature extraction.

    This model uses convolutional layers followed by residual blocks and ends with a fully connected layer for classification.
    """
    def __init__(self, input_dim, blocks_dim, n_classes, kernel_size=17, dropout_rate=0.8):
        """
        Initializes the 1D ResNet model.

        Args:
            input_dim (tuple): Tuple containing the number of input filters and input samples.
            blocks_dim (list): List of tuples specifying the number of filters and output samples for each block.
            n_classes (int): Number of output classes for classification.
            kernel_size (int): Kernel size for convolution layers.
            dropout_rate (float): Dropout rate to be used in convolutional and fully connected layers.
        """
        super().__init__()
        self.n_classes = n_classes
        self.blocks_dim = blocks_dim

        in_filters, in_samples = input_dim
        out_filters, out_samples = blocks_dim[0]
        downsample = _downsample(in_samples, out_samples)
        padding = _padding(downsample, kernel_size)
        self.conv1 = nn.Conv1d(in_filters, out_filters, kernel_size, stride=downsample, padding=padding, bias=True)
        self.bn1 = nn.BatchNorm1d(out_filters)
        self.relu = nn.ReLU()

        self.res_blocks = nn.ModuleList()
        prev_filters, prev_samples = out_filters, out_samples
        for i, (filters, samples) in enumerate(blocks_dim):
            if i == 0:
                downsample = _downsample(input_dim[1], samples)
            else:
                prev_samples = blocks_dim[i - 1][1]
                downsample = _downsample(prev_samples, samples)

            if i == len(blocks_dim) - 1:
                downsample = 1

            block = ResBlock1d(prev_filters, filters, downsample, kernel_size, dropout_rate)
            self.res_blocks.append(block)
            prev_filters, prev_samples = filters, samples
        
        final_filters, final_samples = blocks_dim[-1]
        self.last_layer_dim = final_filters * final_samples
        self.fc = nn.Linear(self.last_layer_dim, n_classes)


    def get_embedding(self, x):
        x = self.relu(self.bn1(self.conv1(x)))

        y = x
        for i, block in enumerate(self.res_blocks):
            x, y = block(x, y)

        x = x.view(x.size(0), -1)

        return x
    

    def forward(self, x):
        """
        Forward pass through the ResNet model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """
        x = self.relu(self.bn1(self.conv1(x)))

        y = x
        for i, block in enumerate(self.res_blocks):
            x, y = block(x, y)

        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x


class Model(L.LightningModule):
    """
    PyTorch Lightning model for training a 1D ResNet-based model for multi-label classification tasks.
    """
    def __init__(self, input_shape, n_labels, lr=1e-3, min_lr=1e-7, labels=None, adv_train=False, perturb_level=None, perturb_type=None, **kwargs):
        """
        Initializes the model with a ResNet backbone for feature extraction.

        Args:
            input_shape (tuple): Tuple representing the shape of the input (filters, samples).
            n_labels (int): Number of labels for classification.
            lr (float): Learning rate for training.
            min_lr (float): Minimum learning rate for the scheduler.
            labels (list): List of class labels for evaluation.
            **kwargs: Additional arguments for flexibility.
        """
        super().__init__()
        self.lr = lr
        self.min_lr = min_lr
        self.labels = labels
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.adv_train = adv_train

        input_dim = (input_shape[0], input_shape[1])
        blocks_dim = [(64, 512), (128, 128), (192, 32), (320, 8)]
        self.model = ResNet1d(input_dim, blocks_dim, n_classes=n_labels)

        
        '''Freeze autoencoder '''
        self.autoencoder = autoencoder.load_from_checkpoint("path/of/autoencoder/checkpoint")
        for param in self.autoencoder.parameters():  
            param.requires_grad = False
        sizes = [5, 7, 11, 15, 19]
        sigmas = [1.0, 3.0, 5.0, 7.0, 10.0]
        crafting_sizes = []
        crafting_weights = []
        self.perturb_level = perturb_level # add perturbation on input or embedding space
        self.perturb_type = perturb_type
        self.attack_label = range(n_labels)  # perturb which task/label
        self.attack_loss_label = range(n_labels) # calculate loss of which task/label
        num_heads = 12
        for size in sizes:
            for sigma in sigmas:
                crafting_sizes.append(size)
                weight = np.arange(size) - size//2
                weight = np.exp(-weight**2.0/2.0/(sigma**2))/np.sum(np.exp(-weight**2.0/2.0/(sigma**2)))
                weight = torch.from_numpy(weight).unsqueeze(0).unsqueeze(0).type(torch.FloatTensor)
                # when ECG has more than 1 head, copy #heads times for weight, which is used for conv
                if self.attack_level == 'input' and n_labels >1: 
                    weight = weight.repeat(1,num_heads,1)
                crafting_weights.append(weight)
        self.sizes = sizes
        self.weights = crafting_weights
        self.eps = 0.5 # default 0.5
        self.step_alpha = 0.001 # default 0.001
        self.num_steps = 20 # default 20

    def predict(self, inputs):
        ''' 
        Predict based on inputs.
        
        Inputs: 
            inputs: (embeddings from vitmae, ids_score) / ECGs
        Return:
            model output
        '''
        if self.perturb_level == 'embedding':
            embeddings, ids_restore = inputs
            # Decode embeddings
            decoder_outputs = self.autoencoder.model.decoder(embeddings, ids_restore)
            reconstructed_patches = decoder_outputs.logits
            decoded_specs = unpatchify(reconstructed_patches, self.autoencoder.patch_size, self.autoencoder.num_channels, self.autoencoder.img_h, self.autoencoder.img_w)
            # convert spec into ECG
            ecgs = self.spec_to_ECG(decoded_specs) 
            return self(ecgs)
        elif self.perturb_level == 'input':
            return self(inputs)

    def forward(self, x):
        """
        Forward pass through the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after passing through the model.
        """
        return self.model(x)

    def pgd_conv(self, attack_label, input_tensors, lengths, targets, criterion, eps = None, step_alpha = None, num_steps = None, sizes = None, weights = None):
        '''
        Add adversarial perturbation
        
        inputs: 
            attack_label: perturb which task/label
            input_tensors: where to add perturbations (encoder output from vitmae / ECGs)
            lengths: length of inputs
            targets: ground truth label of inputs
            
        '''
        if self.perturb_level == 'embedding':
            ids_restore = input_tensors.ids_restore
            inputs = input_tensors.last_hidden_state
            crafting_input = inputs.clone().detach().requires_grad_(True).cuda()
            crafting_target = targets.clone().detach().cuda()
            bce_loss = nn.BCEWithLogitsLoss()
            for i in range(num_steps):
                output = self.predict([crafting_input, ids_restore])
                if len(targets[0])>1:
                    loss = bce_loss(output[:, attack_label], crafting_target[:, attack_label])
                else:
                    loss = criterion(output, crafting_target)
                if crafting_input.grad is not None:
                    crafting_input.grad = None
                loss.backward()
                added = torch.sign(crafting_input.grad)
                step_output = crafting_input + step_alpha * added
                total_adv = step_output - inputs
                total_adv = torch.clamp(total_adv, -eps, eps)
                crafting_output = inputs + total_adv
                # crafting_input = torch.autograd.Variable(crafting_output.detach().clone(), requires_grad=True)
                crafting_input = (crafting_output).detach().requires_grad_()
            added = (crafting_output - inputs).detach().requires_grad_()
            for i in range(num_steps*2):
                new_input = inputs + added
                output = self.predict([new_input, ids_restore])
                loss_fn = nn.BCEWithLogitsLoss()
                loss_pred = loss_fn(output[:, attack_label], targets[:, attack_label].cuda())
                # different reg
                loss_reg = torch.nn.functional.cosine_similarity(inputs, new_input, dim=-1).mean()
                # loss_mse = ((new_input - inputs) ** 2).mean()
                # loss_cos = (1 - F.cosine_similarity(inputs, new_input, dim=-1)).mean()
                # loss_reg = loss_mse + loss_cos
                weight = 10 # two losses are in different scale
                loss = loss_pred + weight * loss_reg
                loss.backward()
                added = added + step_alpha * torch.sign(added.grad.data)
                added = torch.clamp(added, -eps, eps)
                # added = torch.autograd.Variable(added.detach().clone(), requires_grad=True)
                added = added.detach().requires_grad_()
            crafting_output = inputs + added.detach()
            crafting_output_clamp = crafting_output.clone()
            sys.stdout.flush()
        elif self.perturb_level == 'input':
            inputs = input_tensors
            crafting_input = inputs.clone().detach().requires_grad_(True).cuda()
            crafting_target = targets.clone().detach().cuda()
            bce_loss = nn.BCEWithLogitsLoss()
            for i in range(num_steps):
                output = self.predict(crafting_input)
                if len(targets[0])>1:
                    loss = bce_loss(output[:, attack_label], crafting_target[:, attack_label])
                else:
                    loss = criterion(output, crafting_target)
                if crafting_input.grad is not None:
                    crafting_input.grad = None
                loss.backward()
                added = torch.sign(crafting_input.grad)
                step_output = crafting_input + step_alpha * added
                total_adv = step_output - inputs
                total_adv = torch.clamp(total_adv, -eps, eps)
                crafting_output = inputs + total_adv
                # crafting_input = torch.autograd.Variable(crafting_output.detach().clone(), requires_grad=True)
                crafting_input = (crafting_output).detach().requires_grad_()
            added = (crafting_output - inputs).detach().requires_grad_()
            for i in range(num_steps*2):
                new_input = inputs + added
                output = self.predict(new_input)
                loss_fn = nn.BCEWithLogitsLoss()
                loss_pred = loss_fn(output[:, attack_label], targets[:, attack_label].cuda())
                # different reg
                loss_reg = torch.nn.functional.cosine_similarity(inputs, new_input, dim=-1).mean()
                # loss_mse = ((new_input - inputs) ** 2).mean()
                # loss_cos = (1 - F.cosine_similarity(inputs, new_input, dim=-1)).mean()
                # loss_reg = loss_mse + loss_cos

                weight = 10 # two losses are in different scale
                loss = loss_pred + weight * loss_reg
                loss.backward()
                added = added + step_alpha * torch.sign(added.grad.data)
                added = torch.clamp(added, -eps, eps)
                # added = torch.autograd.Variable(added.detach().clone(), requires_grad=True)
                added = added.detach().requires_grad_()
            crafting_output = inputs + added.detach()
            crafting_output_clamp = crafting_output.clone()
            sys.stdout.flush()
        return  crafting_output_clamp  

    
    def ECG_to_Spec(self, ecg_batch, fs=250, nperseg=128, noverlap=96, clip_freq=None, preprocess=False):
        '''
        Convert a batch ECG signal into a batch spectrogram ( for ViTMAE input)
        
        Args:
            ecg_batch: ecg siganls in a batch (batch_size, 12, 2048)
        Returns:
            spec_batch: spec in a batch (batch_size, 24, 224, 224)
            
        '''
        if preprocess:
            ecg_batch = np.array([bandpass_filter(torch.tensor(signal), fs, lowcut, highcut).numpy() for signal in ecg_batch])

        # scipy
        window = 'hann'
        # sfft
        # f, t, Sxx_batch = stft(ecg_batch.cpu(), fs=fs, nperseg=nperseg, noverlap=noverlap, axis=-1) # Sxx_batch: (batch_size, 12, 65, 65)
        f, t, Sxx_batch = stft(ecg_batch.cpu(), fs=fs, window=window, nperseg=nperseg, noverlap=noverlap, boundary='zeros', padded=True)
        # Keep only frequencies ≤ 110 Hz
        if clip_freq:
            freq_mask = f <= clip_freq
            Sxx_batch = Sxx_batch[:, :, freq_mask, :]  
            f = f[freq_mask]
        # Expand Sxx_batch to have 24 channels (alternating real and imaginary parts)
        num_samples, num_leads, num_freqs, num_times = Sxx_batch.shape
        Sxx_batch_expanded = np.empty((num_samples, 24, num_freqs, num_times), dtype=np.float32)
        # Assign real and imaginary parts to alternating channels
        Sxx_batch_expanded[:, 0::2, :, :] = np.real(Sxx_batch)  # Even indices → Real part
        Sxx_batch_expanded[:, 1::2, :, :] = np.imag(Sxx_batch)  # Odd indices → Imaginary part
        # preprocess for spec
        transform_spec = partial(spectrogram_preprocessing_24ch, mean=mean, std=std, resize_size=(224, 224), clip_value=3)
        spec_batch = transform_spec(torch.tensor(Sxx_batch_expanded).cuda()) # [batch_size, 24, 224, 224]

        # torch
        # batch_size, num_leads, sig_len = ecg_batch.shape
        # hop_length = nperseg - noverlap
        # window = torch.hann_window(nperseg, device=ecg_batch.device)
        # Sxx_list = []
        # for lead in range(num_leads):
        #     stft_lead = torch.stft(ecg_batch[:, lead, :], n_fft=nperseg,  hop_length=hop_length,  win_length=nperseg, window=window,return_complex=True,center=True, pad_mode='constant')  # → [batch, freq, time]
        #     Sxx_list.append(stft_lead)
        # Sxx_batch = torch.stack(Sxx_list, dim=1)
        # if clip_freq:
        #     nyquist = fs / 2
        #     freq_bins = Sxx_batch.shape[2]
        #     freqs = torch.linspace(0, nyquist, freq_bins, device=device)
        #     freq_mask = freqs <= clip_freq
        #     Sxx_batch = Sxx_batch[:, :, freq_mask, :]
        # real = Sxx_batch.real
        # imag = Sxx_batch.imag
        # num_samples, num_leads, num_freqs, num_times = real.shape
        # Sxx_expanded = torch.zeros((num_samples, 2 * num_leads, num_freqs, num_times), dtype=torch.float32, device=ecg_batch.device)
        # Sxx_expanded[:, 0::2, :, :] = real
        # Sxx_expanded[:, 1::2, :, :] = imag
        # transform_spec = partial(spectrogram_preprocessing_24ch, mean=mean, std=std, resize_size=(224, 224), clip_value=3)
        # spec_batch = transform_spec(Sxx_expanded)

        return spec_batch


    def spec_to_ECG(self, spec_batch, fs=250, nperseg=128, noverlap=96):
        ''' 
        Converts a batch spectrogram into a batch ECG signal (able to feed into resnet)
        
        Inputs:
            spec_batch: [batch_size, 24, 224, 224]
        Return:
            [batch_size, 12, 2048]

        '''

        # Post-processes model outputs to reconstruct original spectrograms.
        resized_specs = spectrogram_postprocessing_24ch(spec_batch, mean, std, original_size=(original_spec_size[1], original_spec_size[2])) # [64, 24, 65, 65]
        
        '''scipy'''
        # resized_specs = resized_specs.cpu().numpy()
        # batch_size, leads_2, F, T = resized_specs.shape
        # ecgs_list = []
        # for b in range(batch_size):
        #     lead_signals = []
        #     for l in range(leads_2//2):
        #         real_part = resized_specs[b, 2 * l, :, :]  
        #         imag_part = resized_specs[b, 2 * l + 1, :, :]  
        #         Zxx = real_part + 1j * imag_part
        #         _, x_rec = istft(Zxx, fs=fs, nperseg=nperseg, noverlap=noverlap)
        #         lead_signals.append(x_rec)
        #     lead_signals = np.stack(lead_signals, axis=0)  # (leads, signal_length)
        #     ecgs_list.append(lead_signals)
        # ecgs_list = np.stack(ecgs_list, axis=0)
        # ecgs_batch = torch.tensor(ecgs_list, device=spec_batch.device, dtype=torch.float32)
        # return ecgs_batch
        
        '''torch'''
        # batch_size, leads_2, F, T = resized_specs.shape
        # ecgs_list = []
        # for b in range(batch_size):
        #     lead_signals = []
        #     for l in range(leads_2//2):
        #         real_part = resized_specs[b, 2 * l, :, :]  
        #         imag_part = resized_specs[b, 2 * l + 1, :, :]  
        #         Zxx = torch.complex(real_part, imag_part)
        #         # _, x_rec = istft(Zxx, fs=fs, nperseg=nperseg, noverlap=noverlap)
        #         _, x_rec = torch_istft_fast(Zxx, fs=fs, nperseg=nperseg, noverlap=noverlap)
        #         lead_signals.append(x_rec)
        #     lead_signals = torch.stack(lead_signals, dim=0)  # (leads, signal_length)
        #     ecgs_list.append(lead_signals)
        # ecg_batch = torch.stack(ecgs_list, dim=0)

        '''torch speed up'''
        if  resized_specs.ndim == 3:
            resized_specs = resized_specs.unsqueeze(0)
        batch_size, leads_2, F, T = resized_specs.shape
        leads = leads_2 // 2
        # reshape to (batch * leads, 2, F, T)
        specs = resized_specs.view(batch_size, leads, 2, F, T)
        real = specs[:, :, 0, :, :]  # (B, L, F, T)
        imag = specs[:, :, 1, :, :]  # (B, L, F, T)
        Zxx = torch.complex(real, imag)  # (B, L, F, T)
        # Flatten to (B * L, F, T)
        Zxx = Zxx.view(-1, F, T)
        _, ecgs = torch_istft_fast(Zxx, fs=fs, nperseg=nperseg, noverlap=noverlap)
        # Reshape back to (B, L, signal_len)
        ecg_batch = ecgs.view(batch_size, leads, -1)


        return ecg_batch


    def on_mainfold_adv_gen(self, attack_label, inputs, lengths, targets):
        
        '''
        Perturb on latent space (embedding space learned by auto-encoder), then convert into ECG
        
        Inputs:
            inputs: ECG batch
            targets: true labels
        Return: 
            ECG_adv: adversarial ECGs
        '''
        # 1. Convert input ECG into spec (for ViTMAE input)
        spec = self.ECG_to_Spec(ecg_batch=inputs)

        # 2. Get embedding from VitMAE
        encoder_output = self.autoencoder.model.vit(spec)

        # 3. Perturb on embedding
        emb_adv = self.pgd_conv(self.attack_label, encoder_output, lengths, targets, F.cross_entropy, self.eps, self.step_alpha, self.num_steps, self.sizes, self.weights)  # [64, 50, 768]
        # 4. Decode perturbed embedding to new spec
        decoder_outputs_adv = self.autoencoder.model.decoder(emb_adv, encoder_output.ids_restore)
        reconstructed_patches_adv = decoder_outputs_adv.logits
        decoded_specs_adv = unpatchify(reconstructed_patches_adv, self.autoencoder.patch_size, self.autoencoder.num_channels, self.autoencoder.img_h, self.autoencoder.img_w) # [64, 24, 224, 224]
        # 5. Transform new spec to new ECG
        ecg_adv = self.spec_to_ECG(decoded_specs_adv) # (64, 12, 2048)
        return ecg_adv
    
    def input_adv_gen(self, attack_label, inputs, lengths, targets):
        
        '''
        Perturb on input space (ECG)
        
        Inputs:
            inputs: ECG batch
            targets: true labels
        Return: 
            ECG_adv: adversarial ECGs
        '''

        ecg_adv = self.pgd_conv(self.attack_label, inputs, lengths, targets, F.cross_entropy, self.eps, self.step_alpha, self.num_steps, self.sizes, self.weights)  # [64, 50, 768]
        
        return ecg_adv

    def band_limited_noise(self, batch_size, num_leads, signal_length, fs=250, low_freq=3, high_freq=12, device='cpu'):
        """
        Generate gaussian noise for specefic frequency range
        """
        noise = torch.randn(batch_size, num_leads, signal_length, device=device)
        noise_fft = torch.fft.rfft(noise, dim=-1)
        freqs = torch.fft.rfftfreq(signal_length, d=1/fs).to(device)  # [0, fs/2]
        band_mask = (freqs >= low_freq) & (freqs <= high_freq)
        band_mask = band_mask.view(1, 1, -1)  # shape: [1, 1, freq_bins]
        filtered_noise_fft = noise_fft * band_mask
        filtered_noise = torch.fft.irfft(filtered_noise_fft, n=signal_length, dim=-1)

        return filtered_noise

    def gaussian_noise_gen(self, inputs):
        '''
        Augmenting inputs with Gaussian noises filtered into four frequency bands corresponding to four real-world ECG noise:
            3~12 Hz: motion artifacts due to tremors,
            12~50 Hz: low-frequency muscle activation artifact,
            50~100 Hz: electrode motion noise,
            100-150 Hz: higher-frequency muscle activation artifact

        inputs:
            ECG batch, (batch_size, num_leads, signal_length)
        '''
        batch_size, num_leads, signal_length = inputs.shape
        noise_strengths = {'tremor': 0.1,'low_freq_muscle': 0.1, 'electrode_motion': 0.1, 'high_freq_muscle': 0.1}
        noise = torch.zeros_like(inputs)
        noise += noise_strengths['tremor'] * self.band_limited_noise(batch_size=batch_size, num_leads=num_leads, signal_length=signal_length, low_freq=3, high_freq=12, device=inputs.device)
        noise += noise_strengths['low_freq_muscle'] * self.band_limited_noise(batch_size=batch_size, num_leads=num_leads, signal_length=signal_length, low_freq=12, high_freq=50, device=inputs.device)
        noise += noise_strengths['electrode_motion'] * self.band_limited_noise(batch_size=batch_size, num_leads=num_leads, signal_length=signal_length, low_freq=50, high_freq=100, device=inputs.device)
        noise += noise_strengths['high_freq_muscle'] * self.band_limited_noise(batch_size=batch_size, num_leads=num_leads, signal_length=signal_length, low_freq=100, high_freq=150, device=inputs.device)
        noise_ECG = inputs + noise

        return noise_ECG


    def training_step(self, batch, batch_idx):
        """Training step for one batch."""
        if self.adv_train:
            # Get loss of original samples
            x, y, = batch
            y_hat = self(x)
            loss_metric = nn.BCEWithLogitsLoss(reduction='none')
            loss_org = loss_metric(y_hat[:, self.attack_loss_label], y[:, self.attack_loss_label]) 

            # Generate perturbed samples
            if self.perturb_level == 'input' and self.perturb_type == 'gaussian': # add gaussian noise for all samples
                x_adv = self.gaussian_noise_gen(x) 
            elif self.perturb_type == 'adversarial': # add adversarial perturbations for uncertain samples
                # Select uncertain samples
                probs = torch.sigmoid(y_hat.detach())
                entropy = - (probs * torch.log(probs + 1e-8) + (1 - probs) * torch.log(1 - probs + 1e-8))
                uncertainty = entropy.mean(dim=1)
                topk=int(x.shape[0]*0.3)
                topk_values, topk_indices = torch.topk(uncertainty, k=topk)
                mask = torch.zeros(uncertainty.shape, dtype=torch.bool, device=x.device)
                mask[topk_indices] = 1 
                # Generate perturbations for uncertain samples 
                lengths = torch.tensor([x.shape[-1]]*x.shape[0])
                if self.perturb_level == 'embedding':
                    x_adv = self.on_mainfold_adv_gen(self.attack_label, x[mask], lengths, y[mask])               
                elif self.perturb_level == 'input':
                    x_adv = self.input_adv_gen(self.attack_label, x[mask], lengths, y[mask])   
            # Get perturbation loss
            y_adv = self(x_adv) 
            loss_adv = loss_metric(y_adv, y[mask])
            # Final loss
            alpha=0.1
            loss = (loss_org.mean() + alpha*loss_adv.mean()) / 2.0
            self.log("train/loss", loss, prog_bar=True)
            self.log("train/loss_org", loss_org.mean(), prog_bar=True)
            self.log("train/loss_adv", loss_adv.mean(), prog_bar=True)          
        else:
            x, y = batch
            y_hat = self(x)
            loss = self.loss_fn(y_hat, y.float())
            self.log("train/loss", loss, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step for one batch."""
        x, y = batch
        y_hat = self(x)
        return self.calc_metrics(y, y_hat, "val")

    def test_step(self, batch, batch_idx):
        """Test step for one batch."""
        x, y = batch
        y_hat = self(x)
        return self.calc_metrics(y, y_hat, "test")

    def calc_metrics(self, y, y_hat, fold):
        """
        Calculates and logs the evaluation metrics.

        Args:
            y (torch.Tensor): Ground truth labels.
            y_hat (torch.Tensor): Predicted labels.
            fold (str): Name of the fold (train, val, test).

        Returns:
            None
        """
        loss = self.loss_fn(y_hat, y)
        self.log(f"{fold}/loss", loss, prog_bar=True, logger=True,on_epoch=True,sync_dist=True)

        auroc, auprc = {}, {}
        for ave in ["macro", "micro"]:
            auroc[ave] = MultilabelAUROC(num_labels=y.shape[1], average=ave)(y_hat, y.int())
            auprc[ave] = MultilabelAveragePrecision(num_labels=y.shape[1], average=ave)(y_hat, y.int())
            self.log(f"{fold}/auroc/{ave}", auroc[ave], prog_bar=True, logger=True,on_epoch=True,sync_dist=True)
            self.log(f"{fold}/auprc/{ave}", auprc[ave], prog_bar=True, logger=True,on_epoch=True,sync_dist=True)

        aurocs = MultilabelAUROC(num_labels=y.shape[1], average=None)(y_hat, y.int())
        auprcs = MultilabelAveragePrecision(num_labels=y.shape[1], average=None)(y_hat, y.int())
        labels = self.labels if self.labels else [str(i) for i in range(len(aurocs))]

        for i, auroc_val in zip(labels, aurocs):
            self.log(f"{fold}/auroc/{i}", auroc_val, prog_bar=False, logger=True,on_epoch=True,sync_dist=True)
        for i, auprc_val in zip(labels, auprcs):
            self.log(f"{fold}/auprc/{i}", auprc_val, prog_bar=False, logger=True,on_epoch=True,sync_dist=True)

        y_prob = torch.sigmoid(y_hat).detach().cpu().numpy()
        y_pred = (y_prob > 0.5).astype(int)
        y_true = y.detach().cpu().numpy()

        report = classification_report(
            y_true,
            y_pred,
            target_names=labels,
            output_dict=True,
            zero_division=0,
        )

        for class_name in labels:
            self.log(f"{fold}/{class_name}/precision", report[class_name]["precision"])
            self.log(f"{fold}/{class_name}/recall", report[class_name]["recall"])
            self.log(f"{fold}/{class_name}/f1", report[class_name]["f1-score"])

        if "RVEDV" in labels:
            idx = labels.index("RVEDV")
            tp = np.logical_and(y_true[:, idx] == 1, y_pred[:, idx] == 1).sum()
            fp = np.logical_and(y_true[:, idx] == 0, y_pred[:, idx] == 1).sum()
            fn = np.logical_and(y_true[:, idx] == 1, y_pred[:, idx] == 0).sum()
            tn = np.logical_and(y_true[:, idx] == 0, y_pred[:, idx] == 0).sum()

            self.log(f"{fold}/RVEDV/TP", tp)
            self.log(f"{fold}/RVEDV/FP", fp)
            self.log(f"{fold}/RVEDV/FN", fn)
            self.log(f"{fold}/RVEDV/TN", tn)


    def configure_optimizers(self):
        # Create the AdamW optimizer
        optimizer = optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)

        # Set up the learning rate scheduler (StepLR)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "frequency": 1,
            },
        }
