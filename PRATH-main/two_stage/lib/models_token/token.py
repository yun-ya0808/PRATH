import torch.nn as nn
import numpy as np
import torch


class TokenLearnerModuleV11(nn.Module):
    """TokenLearner module Version 1.1, using slightly different conv. layers.
  Instead of using 4 conv. layers with small channels to implement spatial
  attention, this version uses 2 grouped conv. layers with more channels. It
  also uses softmax instead of sigmoid. We confirmed that this version works
  better when having limited training data, such as training with ImageNet1K
  from scratch.
  Attributes:
    num_tokens: Number of tokens.
    dropout_rate: Dropout rate.
  """

    def __init__(self, num_tokens, dropout_rate):
        super().__init__()
        self.num_tokens = num_tokens
        self.dropout_rate = dropout_rate

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Applies learnable tokenization to the 2D inputs.
    Args:
      inputs: Inputs of shape `[bs, h, w, c]`.tensorflow 是b,h,w,c.我用pytorch就是b,c,h,w
      deterministic: Weather we are in the deterministic mode (e.g inference
        time) or not.
    Returns:
      Output of shape `[bs, n_token, c]`.
    """
        feature_shape = inputs.shape

        selected = inputs

        selected = nn.LayerNorm(inputs.size()[1:])(selected)

        for _ in range(1):
            selected = nn.Conv2d(
                feature_shape[1],
                feature_shape[1],
                kernel_size=(1, 1),
                groups=8,
                bias=False)(selected)  # Shape: [bs, h, w, channels].

        selected = nn.Conv2d(
            feature_shape[1],
            self.num_tokens,
            kernel_size=(1, 1),
            stride=(1, 1),
            bias=False)(selected)  # Shape: [bs, h, w, n_token].

        selected = selected.reshape(feature_shape[0], -1,
                                    feature_shape[2] * feature_shape[3])  # Shape: [bs, h*w, n_token].
        # selected = np.transpose(selected, [0, 2, 1])  # Shape: [bs, n_token, h*w].
        selected = torch.softmax(selected, dim=-1)

        feat = inputs
        feat = nn.Conv2d(
            feature_shape[1],
            feature_shape[1],
            kernel_size=(1, 1),
            groups=8,
            bias=False)(feat)  # Shape: [bs, h, w, channels].
        feat = feat.reshape(feature_shape[0], -1, feature_shape[2] * feature_shape[3])  # Shape: [bs, h*w, c].
        feat = feat.transpose(1, 2)

        feat = torch.einsum('...si,...id->...sd', selected, feat)

        return nn.Dropout(self.dropout_rate)(feat)


if __name__ == '__main__':
    input = torch.randn(32, 256, 16, 16)
    Token = TokenLearnerModuleV11(16, 0.1)
    output = Token(input).reshape(input.shape[0], -1, input.shape[2], input.shape[3])
    print(output.shape)
