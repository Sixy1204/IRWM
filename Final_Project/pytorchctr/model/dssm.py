import torch
from pytorchctr.model.layer import FactorizationMachine, FeaturesEmbedding, FeaturesLinear, MultiLayerPerceptron

class DSSM(torch.nn.Module):
    def __init__(self, field_dims, embed_dim, mlp_dims=(200, 200), dropout=0.5):
        super().__init__()
        self.mlp_dims = mlp_dims
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = (len(field_dims) - 1) * embed_dim
        if len(self.mlp_dims) > 0:
            self.item_mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)
            self.cntx_mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)

    def forward(self, cntx_field, cntx_feat, cntx_val, item_field, item_feat, item_val):
        cntx_embed = self.embedding(cntx_field, cntx_feat, cntx_val) # bs, field_num, embed_dim
        item_embed = self.embedding(item_field, item_feat, item_val)
        if len(self.mlp_dims) > 0:
            out = self.cntx_mlp(cntx_embed.view(-1, self.embed_output_dim)) \
                    * self.item_mlp(item_embed.view(-1, self.embed_output_dim))
        else:
            out = cntx_embed.mean(dim=1) * item_embed.mean(dim=1)

        return out.sum(dim=1)
