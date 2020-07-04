import numpy as np
import torch
import torch.nn.functional as F


class FeaturesLinear(torch.nn.Module):

    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim, padding_idx=0)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = torch.Tensor(np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long))
        torch.nn.init.xavier_uniform_(self.fc.weight.data[1:, :])

    def forward(self, x_field, x, x_val=None):  #[batch_size, 3, num_feats] [[[0,1,2,3,4,5,6],[0,1,2,3]]] (1, num_fields, num_dims) 
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + torch.as_tensor(self.offsets[x_field.flatten()], dtype=torch.long, device=x.device).reshape(x.shape)
        if x_val is None:
            return torch.sum(self.fc(x), dim=1) + self.bias
        else:
            return torch.sum(torch.mul(self.fc(x), x_val.unsqueeze(2)), dim=1) + self.bias

class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim, padding_idx=0)
        self.offsets = torch.Tensor(np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long))
        torch.nn.init.xavier_uniform_(self.embedding.weight.data[1:, :])

    def forward(self, x_field, x, x_val=None):  #[batch_size, 3, num_feats] [[[0,1,2,3,4,5,6],[0,1,2,3]]] (1, num_fields, num_dims) 
        """
        :param x: Long tensor of size ``(batch_size, nnz_feats)``
        """
        x = x + torch.as_tensor(self.offsets[x_field.flatten()], dtype=torch.long, device=x.device).reshape(x.shape)
        if x_val is None:
            xs = [self.embedding((x)*(x_field==f).to(torch.long)).sum(dim=1) 
                    for f in range(1, self.num_fields)]
        else:
            xs = [torch.mul(self.embedding((x)*(x_field==f).to(torch.long)), 
                (x_val*(x_field==f).to(torch.float)).unsqueeze(2)).sum(dim=1) 
                for f in range(1, self.num_fields)]
        embedded_x = torch.stack(xs, dim=1)

        return embedded_x 


class FieldAwareFactorizationMachine(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.num_fields = len(field_dims)
        self.embeddings = torch.nn.ModuleList([
            FeaturesEmbedding(field_dims, embed_dim) for _ in range(self.num_fields)
        ])
        #self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long) 

    def forward(self, x_field, x, x_val=None):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        xs = [self.embeddings[f](x_field, x, x_val) for f in range(self.num_fields)]  # field_num, bs, field_num, embed_dim
        ix = list()
        for i in range(self.num_fields - 2):
            for j in range(i, self.num_fields - 1):
                if i == j:
                    ix.append(xs[-1][:, i] * xs[i][:, j])
                else:
                    ix.append(xs[j][:, i] * xs[i][:, j])
        ix = torch.stack(ix, dim=1)
        return ix


class FactorizationMachine(torch.nn.Module):

    def __init__(self, reduce_sum=True):
        super().__init__()
        self.reduce_sum = reduce_sum

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        square_of_sum = torch.sum(x, dim=1) ** 2
        sum_of_square = torch.sum(x ** 2, dim=1)
        ix = square_of_sum - sum_of_square
        if self.reduce_sum:
            ix = torch.sum(ix, dim=1, keepdim=True)
        return 0.5 * ix


class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            #layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields*embed_dim)``
        """
        return self.mlp(x)


if __name__ == '__main__':
    torch.manual_seed(0)
    field_dims = [1, 4, 5]
    fl = FeaturesLinear(field_dims, 1) 
    fe = FeaturesEmbedding(field_dims, 2)
    fm = FactorizationMachine(True)
    ffm = FieldAwareFactorizationMachine(field_dims, 2)
    print(fl.fc.weight.data)
    print(fl.bias.data)
    print(fe.embedding.weight.data)
    x_field = torch.Tensor([[1,1,2,2,0,0], [1,2,1,0,0,0]]).to(torch.long)
    x = torch.tensor([[0,3,1,4,0,0], [2,2,2,0,0,0]])
    x_val = torch.Tensor([[1,1,1,0,0,0], [0.5,1,0.5,0,0,0]])
    print(x_field, x, x_val)
    print(fl.forward(x, x_val))
    print(fe.forward(x_field, x, x_val))
    print(fm(fe.forward(x_field, x, x_val)))
    print(ffm(x_field, x, x_val))

