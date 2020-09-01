from abc import ABC
import torch
import torch.nn as nn
import torch.nn.functional as F


seed = 14


class VectorQuantizer(nn.Module, ABC):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, quant_noise=1):
        super(VectorQuantizer, self).__init__()
        torch.manual_seed(seed)
        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self.embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
        self.embedding.weight.data.uniform_(-1 / self._num_embeddings, 1 / self._num_embeddings)
        self._commitment_cost = commitment_cost

        self.quant_noise = quant_noise

    def forward(self, inputs):
        # convert inputs from (B,C,H,W) -> (B,H,W,C)
        inputs = inputs.permute(0, 2, 3, 1).contiguous()
        input_shape = inputs.shape

        # Flatten input
        flat_input = inputs.view(-1, self._embedding_dim)

        p = self.quant_noise if self.training else 1
        mask = torch.zeros(flat_input.shape[0], device=inputs.device).bernoulli_(p).to(torch.bool)

        # Calculate distances
        distances = (torch.sum(flat_input[mask] ** 2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight ** 2, dim=1)
                     - 2 * torch.matmul(flat_input[mask], self.embedding.weight.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and un flatten
        quantized = torch.zeros(flat_input.shape, device=inputs.device)
        quantized[mask] = torch.matmul(encodings, self.embedding.weight)
        quantized[~mask] = flat_input[~mask]
        quantized = quantized.view(input_shape)

        # Loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self._commitment_cost * e_latent_loss

        quantized = inputs + (quantized - inputs).detach()
        avg_prob = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_prob * torch.log(avg_prob + 1e-10)))

        # convert quantized from (B,H,W,C) -> (B,C,H,W)
        return loss, quantized.permute(0, 3, 1, 2).contiguous(), perplexity, \
            None if self.training else encoding_indices.view(input_shape[:3])


class Residual(nn.Module, ABC):
    def __init__(self, in_channels, num_hidden, num_residual_hidden):
        super(Residual, self).__init__()
        torch.manual_seed(seed)
        self._block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_channels=in_channels,
                      out_channels=num_residual_hidden,
                      kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(in_channels=num_residual_hidden,
                      out_channels=num_hidden,
                      kernel_size=1, stride=1, bias=False)
        )

    def forward(self, x):
        return x + self._block(x)


class ResidualStack(nn.Module, ABC):
    def __init__(self, in_channels, num_hidden, num_residual_layers, num_residual_hidden):
        super(ResidualStack, self).__init__()
        self._num_residual_layers = num_residual_layers
        self._layers = nn.ModuleList([Residual(in_channels, num_hidden, num_residual_hidden)
                                      for _ in range(self._num_residual_layers)])

    def forward(self, x):
        for i in range(self._num_residual_layers):
            x = self._layers[i](x)
        return F.relu(x)


class Encoder(nn.Module, ABC):
    def __init__(self, in_channels, num_hidden, num_residual_layers, num_residual_hidden):
        super(Encoder, self).__init__()
        torch.manual_seed(seed)
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hidden // 2,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_2 = nn.Conv2d(in_channels=num_hidden // 2,
                                 out_channels=num_hidden,
                                 kernel_size=4,
                                 stride=2, padding=1)
        self._conv_3 = nn.Conv2d(in_channels=num_hidden,
                                 out_channels=num_hidden,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hidden,
                                             num_hidden=num_hidden,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hidden=num_residual_hidden)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = F.relu(x)
        x = self._conv_2(x)
        x = F.relu(x)
        x = self._conv_3(x)
        return self._residual_stack(x)


class Decoder(nn.Module, ABC):
    def __init__(self, in_channels, num_hidden, num_residual_layers, num_residual_hidden):
        super(Decoder, self).__init__()
        torch.manual_seed(seed)
        self._conv_1 = nn.Conv2d(in_channels=in_channels,
                                 out_channels=num_hidden,
                                 kernel_size=3,
                                 stride=1, padding=1)
        self._residual_stack = ResidualStack(in_channels=num_hidden,
                                             num_hidden=num_hidden,
                                             num_residual_layers=num_residual_layers,
                                             num_residual_hidden=num_residual_hidden)
        self._conv_trans_1 = nn.ConvTranspose2d(in_channels=num_hidden,
                                                out_channels=num_hidden // 2,
                                                kernel_size=4,
                                                stride=2, padding=1)
        self._conv_trans_2 = nn.ConvTranspose2d(in_channels=num_hidden // 2,
                                                out_channels=3,
                                                kernel_size=4,
                                                stride=2, padding=1)

    def forward(self, inputs):
        x = self._conv_1(inputs)
        x = self._residual_stack(x)
        x = self._conv_trans_1(x)
        x = F.relu(x)
        return self._conv_trans_2(x)


class QNVAE(nn.Module, ABC):
    def __init__(self, num_hidden, num_residual_layers, num_residual_hidden,
                 num_embeddings, embedding_dim, commitment_cost, quant_noise=1):
        super(QNVAE, self).__init__()
        torch.manual_seed(seed)
        self.embedding_dim = embedding_dim
        self._encoder = Encoder(3, num_hidden,
                                num_residual_layers,
                                num_residual_hidden)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hidden,
                                      out_channels=embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        self.vq_vae = VectorQuantizer(num_embeddings, embedding_dim,
                                      commitment_cost, quant_noise)
        self._decoder = Decoder(embedding_dim,
                                num_hidden,
                                num_residual_layers,
                                num_residual_hidden)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        loss, quantized, perplexity, encoding = self.vq_vae(z)
        x_recon = self._decoder(quantized)
        return loss, x_recon, perplexity, encoding

    def decode_samples(self, encoding_indices):
        batch, _, w, h = encoding_indices.shape
        encoding_indices = encoding_indices.view(-1, 1).long()
        flattened_size = encoding_indices.shape[0]
        encodings = torch.zeros(flattened_size, self.vq_vae._num_embeddings, device=encoding_indices.device)
        encodings.scatter_(1, encoding_indices, 1)
        quantized = torch.matmul(encodings, self.vq_vae.embedding.weight)
        quantized = quantized.view([batch, w, h, self.embedding_dim])
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        x_recon = self._decoder(quantized)
        return x_recon


class AE(nn.Module, ABC):
    def __init__(self, num_hidden, num_residual_layers, num_residual_hidden,
                 embedding_dim):
        super(AE, self).__init__()
        torch.manual_seed(seed)
        self._encoder = Encoder(3, num_hidden,
                                num_residual_layers,
                                num_residual_hidden)
        self._pre_vq_conv = nn.Conv2d(in_channels=num_hidden,
                                      out_channels=embedding_dim,
                                      kernel_size=1,
                                      stride=1)
        self._decoder = Decoder(embedding_dim,
                                num_hidden,
                                num_residual_layers,
                                num_residual_hidden)

    def forward(self, x):
        z = self._encoder(x)
        z = self._pre_vq_conv(z)
        x_recon = self._decoder(z)
        return None, x_recon, None, None


if __name__ == '__main__':
    pass
