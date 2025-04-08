import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SimpleAmbienceExtractor(nn.Module):
    def __init__(
            self,
            n_fft=4096,
            hop_length=1024,
            context_frames=2
    ):
        super(SimpleAmbienceExtractor, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.context_frames = context_frames
        self.window = torch.hann_window(n_fft)

        freq_bins = n_fft // 2 + 1
        flattened_features = freq_bins * (2 * context_frames + 1) * 4

        # MLP Network from Ibrahim 2018
        self.mlp = nn.Sequential(
            nn.Linear(flattened_features, 15),
            nn.ReLU(),
            nn.Linear(15, 10),
            nn.ReLU(),
            nn.Linear(10, 1),
            nn.Sigmoid()
        )

        self.criterion = nn.MSELoss()

    def preprocess(self, x):
        """Preprocess raw audio x into STFT-based feature vectors."""
        stft_left = torch.stft(
            x[:, 0], n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window.to(x.device), return_complex=True
        )
        stft_right = torch.stft(
            x[:, 1], n_fft=self.n_fft, hop_length=self.hop_length,
            window=self.window.to(x.device), return_complex=True
        )

        # Split real and imaginary parts
        real_left, imag_left = stft_left.real, stft_left.imag
        real_right, imag_right = stft_right.real, stft_right.imag

        # Combine features for both channels in the correct order
        features = torch.stack([real_left, real_right, imag_left, imag_right], dim=-1)  # Shape: [Batch, Freq, Time, 4]

        # Add temporal context (preceding and succeeding frames)
        batch_size, freq_bins, time_frames, _ = features.shape

        context_features = []

        for t in range(time_frames):
            # Extract the current frame
            context_slice = features[:, :, t:t + 1, :]  # Shape: [Batch, Freq, 1, Features]

            # Pad the beginning
            if t < self.context_frames:
                pad_size = self.context_frames - t
                pad_shape = (0, 0, pad_size, 0)
                preceding_frames = F.pad(features[:, :, :t, :], pad=pad_shape)
            else:
                preceding_frames = features[:, :, t - self.context_frames:t, :]

            # Pad the end
            if t > time_frames - self.context_frames - 1:
                pad_size = t - (time_frames - self.context_frames - 1)
                pad_shape = (0, 0, 0, pad_size)
                succeeding_frames = F.pad(features[:, :, t + 1:, :], pad=pad_shape)
            else:
                succeeding_frames = features[:, :, t + 1:t + self.context_frames + 1, :]

            # Concatenate the context: [Preceding, Current, Succeeding]
            # Shape: [Batch, Freq, Context, Features]
            context_slice = torch.cat([preceding_frames, context_slice, succeeding_frames], dim=2)
            context_features.append(context_slice)

        # Stack the context features into a tensor Shape: [Batch, Freq, Time, Context, Features]
        context_features = torch.stack(context_features, dim=2)

        # Permute and reshape for downstream processing
        context_features = context_features.permute(0, 2, 1, 3, 4)  # [Batch, Time, Freq, Context, Features]
        context_features = context_features.reshape(batch_size, time_frames, -1)  # [Batch, Time, Flattened Features]

        return context_features, stft_left, stft_right

    def forward(self, x, labels=None):
        """Forward pass: preprocess the x, compute y, and optionally calculate loss."""
        input_features, stft_left, stft_right = self.preprocess(x)
        y = self.mlp(input_features)  # Shape: [Batch, Time, 1]

        if self.training and labels is not None:
            labels_expanded = labels.view(-1, 1, 1).expand(-1, y.shape[1], -1)
            loss = self.criterion(y, labels_expanded)
            return loss

        elif labels is not None:
            labels_expanded = labels.view(-1, 1, 1).expand(-1, y.shape[1], -1)
            loss = self.criterion(y, labels_expanded)
            return y, loss
        else:
            y_mask = rearrange(y, "b f 1 -> b 1 f")

            # Apply the mask to both left and right STFT
            masked_stft_left = stft_left * y_mask
            masked_stft_right = stft_right * y_mask

            # Convert masked STFTs back to time-domain signals
            reconstructed_left = torch.istft(masked_stft_left, n_fft=self.n_fft, hop_length=self.hop_length,
                                             window=self.window.to(x.device), length=x.shape[-1])
            reconstructed_right = torch.istft(masked_stft_right, n_fft=self.n_fft, hop_length=self.hop_length,
                                              window=self.window.to(x.device), length=x.shape[-1])

            return torch.stack([reconstructed_left, reconstructed_right], dim=1)  # Shape: [Batch, 2, Time]


if __name__ == "__main__":
    from torchinfo import summary

    n_fft = 4096
    hop_length = 1024
    context_frames = 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SimpleAmbienceExtractor(
        n_fft=n_fft,
        hop_length=hop_length,
        context_frames=context_frames
    ).to(device)

    batch_size = 8
    sample_rate = 44100
    duration = 1
    waveform = torch.randn(batch_size, 2, sample_rate * duration).to(device)
    dummy_labels = torch.randint(0, 2, (batch_size, 1), dtype=torch.float32).to(device)

    print("Model Summary:")
    summary(model, input_data=waveform)

    model.train()
    train_loss = model(waveform, dummy_labels)
    print(f"Training Loss: {train_loss:.4f}")

    model.eval()
    outputs, eval_loss = model(waveform, dummy_labels)
    print(f"Evaluation Loss: {eval_loss:.4f}")

    inference_outputs = model(waveform)
    print(f"Primary Outputs: {inference_outputs[0].squeeze()}\n")
    print(f"Ambient Outputs: {inference_outputs[1].squeeze()}\n")
