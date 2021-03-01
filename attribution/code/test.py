import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix
from tqdm import tqdm
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

import seed
from model import *


X_train, X_test, y_train, y_test = seed.gather_loans()#seed_data(num_input=5)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") #Enable cuda if available

# seed the model for reproducibility (ideally across all nodes, later...)
torch.manual_seed(0)

def load_model(PATH):
    model.load_state_dict(torch.load("../model/" + PATH), strict=False)

if __name__ == "__main__":
    model = LoanNetwork().to(device)
    load_model(PATH="loans_model.pth")
    model.eval() #set to evaluation mode
    
    #### predict X_test data
    predictions=[]
    with torch.no_grad():
        for i, data in enumerate(X_test):
            y_pred = model(data)
            predictions.append(y_pred.argmax().item())
            
    predictions = np.array(predictions, dtype=np.int16)
    
    ### compute metrics
    # score = accuracy_score(torch.max(y_test, 1)[1], predictions)
    score = accuracy_score(y_test, predictions)
    print(f"Accuracy with {X_test.shape[1]} inputs", score)