# KRAdapter
Official code repository for ICCV 2025: Towards Higher Effective Rank in Parameter-efficient Fine-tuning using Khatri–Rao Product

Paper: []

# Kathri-Rao product for matrix estimation

We are working on a hugginface peft implementation to allow users to easily experiment with KRAdapter. 
In the meantime, users can run our toy experiements on matrix appoximation (to be uploaded soon) or use the code below to use the Kathri-Rao product 

```python
import torch
import torch.nn as nn
from typing import Any
import math

class KRAdapterLinearLayer(nn.Module):
    def __init__(self, original_layer, scaling=2): #No rank needed in KRAdapter as it is inferred from the base layer shape
        super().__init__()
        in_features, out_features = original_layer.in_features, original_layer.out_features
        min_shape, max_shape = min(in_features, out_features), max(out_features, in_features)
        self.s = (out_features, in_features)
        self.r = int(math.sqrt(min_shape))
        self.scaling = scaling
        self.original_layer = original_layer
        self.merged = False

        self.d = 0
        while self.r * (self.r + self.d) < max_shape:
            self.d += 1
        
        self.min_shape = min_shape
        self.weight = nn.Parameter(torch.zeros(min_shape, self.r))
        self.v_weight = nn.Parameter(torch.zeros(min_shape, self.r + self.d))

        nn.init.kaiming_uniform_(self.v_weight, a=5)

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
        if self.merged:
            return self.original_layer(x)

        update = self.get_update()
        return x @ update * self.scaling + self.original_layer(x)

    def get_update(self):
        update = (self.weight.unsqueeze(1) * self.v_weight.unsqueeze(-1)).view(self.weight.shape[0], -1)
        if torch.argmin(torch.tensor(self.s)) != torch.argmin(torch.tensor(update.shape)):
            update = update.T
        update = update[:self.s[0], :self.s[1]]
        return update.T

    @torch.no_grad
    def merge(self):
        if not self.merged:
            self.original_layer.weight.data += self.get_update().data * self.scaling
            self.merged = True
    
    @torch.no_grad
    def unmerge(self):
        if self.merged:
            self.original_layer.weigh.data -= self.get_update().data * self.scaling
            self.merged = False


if __name__ == "__main__":
    size = (1024, 768)
    original_layer = nn.Linear(*size)
    for p in original_layer.parameters():
        p.requires_grad = False

    kra_layer = KRAdapterLinearLayer(original_layer)

    x = torch.randn(64, 4, size[0])
    out = kra_layer(x)
```