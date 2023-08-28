import torch
import random

def loss_s(x, y, T):
    b = x.size(0)  # Batch size
    lsn_loss = 0.0
    
    for i in range(b):
        xi = x[i]
        yi = y[i]
        
        numerator = 0.0
        denominator = 0.0
        
        for j in range(b):
            if j != i and y[j] == yi:
                numerator += torch.exp(-((xi - x[j])**2).sum() / T)
            if j != i:    
                denominator += torch.exp(-((xi - x[j])**2).sum() / T)
        lsn_loss += -torch.log(numerator / denominator)
    
    return lsn_loss / b

# Function definition
def loss(x, y, T):
    b = x.size(0)  # Batch size
    y = y.squeeze()

    x_expanded = x.unsqueeze(1)  # Expand dimensions for broadcasting
    y_expanded = y.unsqueeze(0)

    same_class_mask = y_expanded == y_expanded.t()

    squared_distances = (x_expanded - x_expanded.t()) ** 2
    exp_distances = torch.exp(-(squared_distances / T))
    exp_distances = exp_distances * (1 - torch.eye(b))
    #print(exp_distances)

    numerator = exp_distances * same_class_mask
    denominator = exp_distances

    print(denominator)

    lsn_loss = -torch.log(0.00001 + (numerator.sum(dim=1) / (0.00001 + denominator.sum(dim=1)))).mean()

    return lsn_loss

x = torch.tensor([1.5, -2.0, 0.5, 2.5, 3, 2.5, 3])
y = torch.tensor([[0], [1], [0], [1], [1], [1], [1]])

# Call the method and calculate the loss
T = 1.0  # You can adjust this temperature value
loss_value = loss(x, y, T)
loss_value_s = loss_s(x, y, T)

print("\nCalculated Loss:", loss_value.item())
print("\nCalculated Loss s:", loss_value_s.item())
