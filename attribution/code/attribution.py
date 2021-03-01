from captum.attr import (
            IntegratedGradients, LayerActivation, 
            LayerConductance, DeepLift, 
            Occlusion, NeuronConductance)

import plotly.express as px
import torch
import numpy as np

from model import *
from seed import *

# defining datasets
X_train, X_test, y_train, y_test = gather_loans()#seed_data(5)

# defining number of examples
N = 100

# defining baselines for each input tensor
baseline = torch.zeros(N, 11)

# defining and laoding existing model
model = LoanNetwork()
model.load_state_dict(torch.load("../model/loans_model.pth"), strict=True)
model.eval()

# defining sample input
indices = np.random.choice(np.arange(len(X_test)), N)
input = X_test[indices].reshape(N, 11)#torch.rand(1, 5)

# defining and applying integrated gradients on SeedModel
ig = IntegratedGradients(model)
attributions, approximation_error = ig.attribute(inputs=input,
                                                 baselines=baseline,
                                                 return_convergence_delta=True,
                                                 target=0,
                                                 n_steps=50)

# # defining and applying Layer Conducatance to see the importance of neurons for a layer and given input.
# lc = LayerConductance(model, model.fc2)
# attributions, approximation_error = lc.attribute(inputs=input,
#                                                  baselines=baseline,
#                                                  return_convergence_delta=True,
#                                                  target=0)

# # defining and applying DeepLIFT on SeedModel
# dl = DeepLift(model)
# attributions, approximation_error = dl.attribute(inputs=input,
#                                                  baselines=baseline,
#                                                  return_convergence_delta=True,
#                                                  target=0)

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
                    labels=dict(x='Feature Inputs', y='Number of Inputs', color="Attributions"),
                    x=['Gender', 'Married', 'Dependents', 'Education', 'Self Employed', 'Applicant Income', 'Coapplicant Income',
                       'Loan Amount', 'Loan Amount Term', 'Credit History', 'Property Area'])
    
    fig.show()