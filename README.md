# Shapley Explanation Networks
Implementation of the paper ["Shapley Explanation Networks"](https://openreview.net/forum?id=vsU0efpivw) at ICLR 2021.
Note that this repo heavily uses the experimental feature of [named tensors](https://pytorch.org/docs/stable/named_tensor.html) in PyTorch. As it was really confusing to implement the ideas for the authors, we find it tremendously easier to use this feature.
## Dependencies
For running only ShapNets, one would mostly only need PyTorch, NumPy, and SciPy.
## Usage
### For a Shapley Module:
```python
import torch
import torch.nn as nn
from ShapNet.utils import ModuleDimensions
from ShapNet import ShapleyModule

b_size = 3
features = 4
out = 1
dims = ModuleDimensions(
    features=features,
    in_channel=1,
    out_channel=out
)

sm = ShapleyModule(
    inner_function=nn.Linear(features, out),
    dimensions=dims
)
sm(torch.randn(b_size, features), explain=True)
```

### For a Shallow ShapNet
```python
import torch
import torch.nn as nn
from ShapNet.utils import ModuleDimensions
from ShapNet import ShapleyModule, OverlappingShallowShapleyNetwork

batch_size = 32
class_num = 10
dim = 32

overlapping_modules = [
    ShapleyModule(
        inner_function=nn.Sequential(nn.Linear(2, class_num)),
        dimensions=ModuleDimensions(
            features=2, in_channel=1, out_channel=class_num
        ),
    ) for _ in range(dim * (dim - 1) // 2)
]
shallow_shapnet = OverlappingShallowShapleyNetwork(
    list_modules=overlapping_modules
)
inputs = torch.randn(batch_size, dim, ), )
shallow_shapnet(torch.randn(batch_size, dim, ), )
output, bias = shallow_shapnet(inputs, explain=True, )
```
### For a Deep ShapNet
```python
import torch
import torch.nn as nn
from ShapNet.utils import ModuleDimensions
from ShapNet import ShapleyModule, ShallowShapleyNetwork, DeepShapleyNetwork

dim = 32
dim_input_channels = 1
class_num = 10
inputs = torch.randn(32, dim, ), )


dims = ModuleDimensions(
    features=dim,
    in_channel=dim_input_channels,
    out_channel=class_num
)
deep_shapnet = DeepShapleyNetwork(
    list_shapnets=[
        ShallowShapleyNetwork(
            module_dict=nn.ModuleDict({
                "(0, 2)": ShapleyModule(
                    inner_function=nn.Linear(2, class_num),
                    dimensions=ModuleDimensions(
                        features=2, in_channel=1, out_channel=class_num
                    )
                )},
            ),
            dimensions=ModuleDimensions(dim, 1, class_num)
        ),
    ],
)
deep_shapnet(inputs)
outputs = deep_shapnet(inputs, explain=True, )
```
### For a vision model:
```python
import numpy as np
import torch
import torch.nn as nn

# =============================================================================
# Imports {\sc ShapNet}
# =============================================================================
from ShapNet import DeepConvShapNet, ShallowConvShapleyNetwork, ShapleyModule
from ShapNet.utils import ModuleDimensions, NAME_HEIGHT, NAME_WIDTH, \
    process_list_sizes

num_channels = 3
num_classes = 10
height = 32
width = 32
list_channels = [3, 16, 10]
pruning = [0.2, 0.]
kernel_sizes = process_list_sizes([2, (1, 3), ])
dilations = process_list_sizes([1, 2])
paddings = process_list_sizes([0, 0])
strides = process_list_sizes([1, 1])

args = {
    "list_shapnets": [
        ShallowConvShapleyNetwork(
            shapley_module=ShapleyModule(
                inner_function=nn.Sequential(
                    nn.Linear(
                        np.prod(kernel_sizes[i]) * list_channels[i],
                        list_channels[i + 1]),
                    nn.LeakyReLU()
                ),
                dimensions=ModuleDimensions(
                    features=int(np.prod(kernel_sizes[i])),
                    in_channel=list_channels[i],
                    out_channel=list_channels[i + 1])
            ),
            reference_values=None,
            kernel_size=kernel_sizes[i],
            dilation=dilations[i],
            padding=paddings[i],
            stride=strides[i]
        ) for i in range(len(list_channels) - 1)
    ],
    "reference_values": None,
    "residual": False,
    "named_output": False,
    "pruning": pruning
}

dcs = DeepConvShapNet(**args)

```
