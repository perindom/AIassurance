from captum.attr import (
            IntegratedGradients, LayerActivation, 
            LayerConductance, DeepLift, 
            Occlusion, NeuronConductance)

import plotly.express as px
import torch
import numpy as np

from model import SeedNetwork
from seed import seed_data

# defining datasets
X_train, X_test, y_train, y_test = seed_data(5)

# defining number of examples
N = 10

# defining baselines for each input tensor
baseline = torch.zeros(N, 5)

# defining and laoding existing model
model = SeedNetwork()
model.load_state_dict(torch.load("../model/seedNetwork.pth"), strict=True)
model.eval()

# defining sample input
input = X_test[:N].reshape(N, 5)#torch.rand(1, 5)

# defining and applying integrated gradients on SeedModel
ig = IntegratedGradients(model)
attributions, approximation_error = ig.attribute(inputs=input,
                                                 baselines=baseline,
                                                 return_convergence_delta=True,
                                                 target=0,
                                                 n_steps=50)

# defining and applying Layer Conducatance to see the importance of neurons for a layer and given input.
lc = LayerConductance(model, model.fc2)
attributions, approximation_error = lc.attribute(inputs=input,
                                                 baselines=baseline,
                                                 return_convergence_delta=True,
                                                 target=0)

# defining and applying DeepLIFT on SeedModel
dl = DeepLift(model)
attributions, approximation_error = dl.attribute(inputs=input,
                                                 baselines=baseline,
                                                 return_convergence_delta=True,
                                                 target=0)

# defining and applying Occlusion on SeedModel
ablator = Occlusion(model)
### TODO: fix error from sliding_window_shapes. !-- perturbation causes param mismatch --!
# attributions, approximation_error = ablator.attribute(inputs=input,
#                                                  sliding_window_shapes=(1,1),
#                                                  target=1)

if __name__ == "__main__":
    # print("Attributions\n", attributions.detach().numpy())
    # print("Error", approximation_error.detach().numpy())
    
    fig = px.imshow(attributions.detach().numpy(), 
                    labels=dict(x='Feature Inputs', y='Number of Inputs', color="Attributions"))
    fig.show()