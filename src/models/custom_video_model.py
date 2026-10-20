"""
Custom Video Deepfake Detection Model (CustomVideoDeepfakeModel).
Combines our custom spatial CNN feature extractor with a Bidirectional LSTM
and temporal self-attention aggregation to identify frame-to-frame inconsistencies.
Designed and trained from scratch with random weight initialization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.custom_cnn import CustomDeepfakeCNN


class TemporalAttention(nn.Module):
    """
    Learned temporal attention mechanism to weight frames
    based on manipulation artefact prominence and temporal transitions.
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.Tanh(),
            nn.Linear(hidden_dim // 2, 1)
        )

    def forward(self, rnn_outputs):
        # rnn_outputs shape: (Batch, Seq_len, Hidden_dim)
        scores = self.attn(rnn_outputs)  # (Batch, Seq_len, 1)
        weights = F.softmax(scores, dim=1)  # (Batch, Seq_len, 1)
        context = torch.sum(weights * rnn_outputs, dim=1)  # (Batch, Hidden_dim)
        return context, weights


class CustomVideoDeepfakeModel(nn.Module):
    """
    Two-Stage Spatio-Temporal Deepfake Video Classifier:
    1. Spatial Feature Extractor: Custom 5-block CNN extracting 512-dim features per frame.
    2. Temporal Sequential Learner: 2-layer Bidirectional LSTM learning sequence dynamics.
    3. Temporal Attention: Identifies key anomalous frames.
    4. Dual Heads: Produces both sequence-level and frame-level predictions.
    """
    def __init__(self, spatial_backbone=None, hidden_dim=128, num_layers=2, dropout=0.3):
        super().__init__()
        if spatial_backbone is not None:
            self.spatial_extractor = spatial_backbone
        else:
            self.spatial_extractor = CustomDeepfakeCNN()

        self.feature_dim = 512
        self.hidden_dim = hidden_dim
        self.num_directions = 2  # Bidirectional

        # Temporal Recurrent Component: Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        rnn_out_dim = hidden_dim * self.num_directions  # 128 * 2 = 256

        # Temporal Attention Aggregation
        self.temporal_attention = TemporalAttention(rnn_out_dim)

        # Frame-Level Classification Head (evaluates each frame in sequence)
        self.frame_classifier = nn.Sequential(
            nn.Linear(rnn_out_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, 1)
        )

        # Video-Level Classification Head (evaluates overall sequence context)
        self.video_classifier = nn.Sequential(
            nn.Linear(rnn_out_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Dropout(p=0.4),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(64, 1)
        )

        self._init_temporal_weights()

    def _init_temporal_weights(self):
        """Orthogonal initialization for LSTM and Kaiming for linear heads."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                # Forget gate bias trick
                n = param.size(0)
                param.data[(n // 4):(n // 2)].fill_(1.0)

        for m in [self.frame_classifier, self.video_classifier]:
            for layer in m.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, mode="fan_out", nonlinearity="relu")
                    if layer.bias is not None:
                        nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        """
        Forward pass for a sequence of frames.
        x shape: (Batch, Num_Frames, Channels, Height, Width)
        Returns:
            video_logits: (Batch, 1) sequence-level prediction
            frame_logits: (Batch, Num_Frames, 1) individual frame predictions
            attention_weights: (Batch, Num_Frames, 1) frame importance weights
        """
        batch_size, seq_len, c, h, w = x.shape

        # Reshape to process all frames through spatial extractor in parallel
        x_reshaped = x.view(batch_size * seq_len, c, h, w)
        spatial_features = self.spatial_extractor.extract_features(x_reshaped)  # (B*T, 512)

        # Restore temporal sequence dimension: (Batch, Seq_len, 512)
        spatial_features = spatial_features.view(batch_size, seq_len, self.feature_dim)

        # Temporal sequence modeling
        rnn_out, _ = self.lstm(spatial_features)  # (Batch, Seq_len, 256)

        # Frame-level predictions across time
        frame_logits = self.frame_classifier(rnn_out)  # (Batch, Seq_len, 1)

        # Sequence attention aggregation
        context, attn_weights = self.temporal_attention(rnn_out)  # (Batch, 256), (Batch, Seq_len, 1)

        # Video-level classification
        video_logits = self.video_classifier(context)  # (Batch, 1)

        return video_logits, frame_logits, attn_weights

    def predict_video(self, x):
        """Inference helper returning probabilities."""
        with torch.no_grad():
            video_logits, frame_logits, attn_weights = self.forward(x)
            video_prob = torch.sigmoid(video_logits)
            frame_probs = torch.sigmoid(frame_logits)
            return video_prob, frame_probs, attn_weights


if __name__ == "__main__":
    video_model = CustomVideoDeepfakeModel()
    dummy_video = torch.randn(2, 8, 3, 128, 128)  # 2 videos, 8 frames each
    v_out, f_out, attn = video_model(dummy_video)
    print("CustomVideoDeepfakeModel Initialized Successfully!")
    print(f"Input video shape: {dummy_video.shape}")
    print(f"Video logits: {v_out.shape}")
    print(f"Frame logits: {f_out.shape}")
    print(f"Attention weights: {attn.shape}")
    total_params = sum(p.numel() for p in video_model.parameters() if p.requires_grad)
    print(f"Total Video Model Parameters: {total_params:,}")
