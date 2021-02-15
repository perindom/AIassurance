from captum.attr import IntegratedGradients, LayerConductance, DeepLift, NeuronConductance
from model import SeedNetwork
import torch
import numpy as np

# defining baselines for each input tensor
baseline = torch.zeros(10, 5)

# defining and laoding existing model
model = SeedNetwork()
# model.load_state_dict(torch.load("../model/seedNetwork.pth"), strict=True)
model.eval()

# defining sample input
input = torch.rand(10, 5)

# defining and applying integrated gradients on SeedModel and the
ig = IntegratedGradients(model)
attributions, approximation_error = ig.attribute(inputs=input,
                                                 baselines=baseline,
                                                 return_convergence_delta=True,
                                                 target=0,
                                                 n_steps=50)

# defining and applying Layer Conducatance to see the importance of neurons for a layer and given input.
lc = LayerConductance(model, model.fc1) #specify importance in layer 1
attributions, approximation_error = lc.attribute(inputs=input,
                                                 baselines=baseline,
                                                 return_convergence_delta=True,
                                                 target=0)

if __name__ == "__main__":
    print("Attributions\n", attributions)
    print("Error", approximation_error.numpy())