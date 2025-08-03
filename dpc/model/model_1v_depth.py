import torch
import torch.nn.functional as F
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerEncoderModel(nn.Module):
    def __init__(self, input_dim):
        super(TransformerEncoderModel, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model=input_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, dim_feedforward=256)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=1)
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        return x





class ResConvBlockLayer(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size, dropout=0.25):
        super(ResConvBlockLayer, self).__init__()
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding='same')
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=hidden, kernel_size=kernel_size, padding='same')
        self.conv2 = nn.Conv1d(in_channels=hidden, out_channels=out_channels, kernel_size=kernel_size, padding='same')
        self.bn = nn.BatchNorm1d(out_channels)
        self.maxpool = nn.MaxPool1d(kernel_size=kernel_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.bn(x)
        x += residual
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        return x



class ChromosomeCNN(nn.Module):
    def __init__(self, num_snps,input_dim):
        super(ChromosomeCNN, self).__init__()
        self.res_block1 = ResConvBlockLayer(in_channels=input_dim, hidden=16, out_channels=32, kernel_size=5)
        self.res_block2 = ResConvBlockLayer(in_channels=32, hidden=32, out_channels=64, kernel_size=5)
        self.res_block3 = ResConvBlockLayer(in_channels=64, hidden=32, out_channels=16, kernel_size=5)

    def forward(self, x):
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        return x


class TransformerDecoder(nn.Module):
    def __init__(self, input_dim):
        super(TransformerDecoder, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, dim_feedforward=256)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=1)
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        return x


class PhenotypePredictor_1v(nn.Module):
    def __init__(self, num_chromosomes, snp_counts, input_dim):
        super(PhenotypePredictor_1v, self).__init__()

        self.chromosome_cnns = nn.ModuleList(
            [ChromosomeCNN(snp_counts[i],input_dim) for i in range(num_chromosomes)]
        )

        with torch.no_grad():
            sample_input = torch.randn(1, input_dim, snp_counts[0])
            sample_output = self.chromosome_cnns[0](sample_input)
            cnn_output_channels = sample_output.size(1)
            cnn_output_length = sample_output.size(2)
        self.transformer_decoder = TransformerEncoderModel(input_dim=cnn_output_channels)
        self.total_sequence_length = cnn_output_length * cnn_output_channels * num_chromosomes
        self.mlp = nn.Sequential(
            nn.Linear(self.total_sequence_length, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        outputs = []
        for i, cnn in enumerate(self.chromosome_cnns):

            chromosome_data = x[:, i, :, :].permute(0, 2, 1)
            output = cnn(chromosome_data)
            outputs.append(output)

        combined_output = torch.cat(outputs, dim=2)
        combined_output = combined_output.permute(0, 2, 1)
        combined_output = self.transformer_decoder(combined_output)
        combined_output = torch.flatten(combined_output, 1)
        combined_output = self.mlp(combined_output)
        return combined_output


def print_output_shape(module, input, output):

    if isinstance(output, tuple):
        print(f"Module: {module.__class__.__name__}, Output is a tuple with {len(output)} elements")
        for i, item in enumerate(output):
            if hasattr(item, 'shape'):
                print(f"  Element {i}: Shape {item.shape}")
            else:
                print(f"  Element {i}: Not a tensor")
    else:
        print(f"Module: {module.__class__.__name__}, Output Shape: {output.shape}")



def register_hooks(model):
    hooks = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList):
            hook = module.register_forward_hook(print_output_shape)
            hooks.append(hook)
    return hooks


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dim = 10
    chromosome_tensors = torch.randn(1, 10, 1000, dim)
    phenotypes_tensor = torch.randn(1, 1)


    num_chromosomes = 10
    snp_counts = [1000]*num_chromosomes
    model = PhenotypePredictor_1v(num_chromosomes, snp_counts, dim).to(device)
    model.eval()
    hooks = register_hooks(model)


    chromosome_tensors = chromosome_tensors.to(device)
    output = model(chromosome_tensors)
    print(output.shape)
