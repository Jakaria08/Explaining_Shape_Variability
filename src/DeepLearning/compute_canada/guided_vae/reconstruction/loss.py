import numpy as np 
import torch
import torch.nn as nn
import math

class SNNLCrossEntropy():
    STABILITY_EPS = 0.00001
    def __init__(self,
               temperature=100.,
               factor=-10.,
               optimize_temperature=True,
               cos_distance=True):
        
        self.temperature = temperature
        self.factor = factor
        self.optimize_temperature = optimize_temperature
        self.cos_distance = cos_distance
    
    @staticmethod
    def pairwise_euclid_distance(A, B):
        """Pairwise Euclidean distance between two matrices.
        :param A: a matrix.
        :param B: a matrix.
        :returns: A tensor for the pairwise Euclidean between A and B.
        """
        batchA = A.shape[0]
        batchB = B.shape[0]

        sqr_norm_A = torch.reshape(torch.pow(A, 2).sum(axis=1), [1, batchA])
        sqr_norm_B = torch.reshape(torch.pow(B, 2).sum(axis=1), [batchB, 1])
        inner_prod = torch.matmul(B, A.T)

        tile_1 = torch.tile(sqr_norm_A, [batchB, 1])
        tile_2 = torch.tile(sqr_norm_B, [1, batchA])
        return (tile_1 + tile_2 - 2 * inner_prod)
    
    @staticmethod
    def pairwise_cos_distance(A, B):
        
        """Pairwise cosine distance between two matrices.
        :param A: a matrix.
        :param B: a matrix.
        :returns: A tensor for the pairwise cosine between A and B.
        """
        normalized_A = torch.nn.functional.normalize(A, dim=1)
        normalized_B = torch.nn.functional.normalize(B, dim=1)
        prod = torch.matmul(normalized_A, normalized_B.transpose(-2, -1).conj())
        return 1 - prod
    
    @staticmethod
    def fits(A, B, temp, cos_distance):
        if cos_distance:
            distance_matrix = SNNLCrossEntropy.pairwise_cos_distance(A, B)
        else:
            distance_matrix = SNNLCrossEntropy.pairwise_euclid_distance(A, B)
            
        return torch.exp(-(distance_matrix / temp))
    
    @staticmethod
    def pick_probability(x, temp, cos_distance):
        """Row normalized exponentiated pairwise distance between all the elements
        of x. Conceptualized as the probability of sampling a neighbor point for
        every element of x, proportional to the distance between the points.
        :param x: a matrix
        :param temp: Temperature
        :cos_distance: Boolean for using cosine or euclidean distance
        :returns: A tensor for the row normalized exponentiated pairwise distance
                  between all the elements of x.
        """
        f = SNNLCrossEntropy.fits(x, x, temp, cos_distance) - torch.eye(x.shape[0], device='cuda:0')
        return f / (SNNLCrossEntropy.STABILITY_EPS + f.sum(axis=1).unsqueeze(1))
    
    @staticmethod
    def same_label_mask(y, y2):
        """Masking matrix such that element i,j is 1 iff y[i] == y2[i].
        :param y: a list of labels
        :param y2: a list of labels
        :returns: A tensor for the masking matrix.
        """
        return (y == y2.unsqueeze(1)).squeeze().to(torch.float32)
    
    @staticmethod
    def masked_pick_probability(x, y, temp, cos_distance):
        """The pairwise sampling probabilities for the elements of x for neighbor
        points which share labels.
        :param x: a matrix
        :param y: a list of labels for each element of x
        :param temp: Temperature
        :cos_distance: Boolean for using cosine or Euclidean distance
        :returns: A tensor for the pairwise sampling probabilities.
        """
        return SNNLCrossEntropy.pick_probability(x, temp, cos_distance) * \
                                    SNNLCrossEntropy.same_label_mask(y, y)
    
    @staticmethod
    def SNNL(x, y, temp=100, cos_distance=True):
        """Soft Nearest Neighbor Loss
        :param x: a matrix.
        :param y: a list of labels for each element of x.
        :param temp: Temperature.
        :cos_distance: Boolean for using cosine or Euclidean distance.
        :returns: A tensor for the Soft Nearest Neighbor Loss of the points
                  in x with labels y.
        """
        summed_masked_pick_prob = SNNLCrossEntropy.masked_pick_probability(x, y, temp, cos_distance).sum(axis=1)
        return -torch.log(SNNLCrossEntropy.STABILITY_EPS + summed_masked_pick_prob).mean()



class CorrelationLoss(nn.Module):
    def __init__(self):
        super(CorrelationLoss, self).__init__()

    def forward(self, z_batch, y_batch):
        # Split z_batch and y_batch into categories
        z_1 = z_batch[y_batch.flatten() == 1.0]
        z_0 = z_batch[y_batch.flatten() == 0.0]
        n_1 = len(z_1)
        n_0 = len(z_0)
        n = n_1 + n_0

        # Calculate means for the two categories
        mean_z_1 = torch.mean(z_1[:, 0])
        mean_z_0 = torch.mean(z_0[:, 0])

        # Multiplier
        mlt = math.sqrt((n_1 * n_0) / (n**2))

        # Calculate point biserial correlation
        r_pb = (mean_z_1 - mean_z_0) / torch.std(z_batch[:, 0]) * mlt

        # Calculate correlation of other dimensions with y
        other_dim_corrs = torch.zeros_like(z_batch[:, 1])
        for i in range(1, z_batch.shape[1]):
            other_dim_corrs[i-1] = (torch.mean(z_1[:, i]) - torch.mean(z_0[:, i])) / torch.std(z_batch[:, i]) * mlt

        # Loss components
        ncc_loss = 1 - torch.abs(r_pb)  # Minimize correlation
        other_dims_loss = torch.mean(torch.abs(other_dim_corrs))  # Minimize other dimension correlations

        # Combine losses with weights
        total_loss = ncc_loss + other_dims_loss

        return total_loss

# SNNL loss modified fast
class SNNLoss(nn.Module):
    def __init__(self, T):
        super(SNNLoss, self).__init__()
        self.T = T
        self.STABILITY_EPS = 0.00001

    def forward(self, x, y):
        b = x.size(0)  # Batch size
        y = y.squeeze()

        x_expanded = x.unsqueeze(1)  # Expand dimensions for broadcasting
        y_expanded = y.unsqueeze(0)

        same_class_mask = y_expanded == y_expanded.t()

        squared_distances = (x_expanded - x_expanded.t()) ** 2
        exp_distances = torch.exp(-(squared_distances / self.T))
        exp_distances = exp_distances * (1 - torch.eye(b, device='cuda:0'))
        #print(exp_distances)

        numerator = exp_distances * same_class_mask
        denominator = exp_distances

        #print(denominator)

        lsn_loss = -torch.log(self.STABILITY_EPS + (numerator.sum(dim=1) / (self.STABILITY_EPS + denominator.sum(dim=1)))).mean()

        return lsn_loss

# SNNL loss modified slow
'''
class SNNLoss(nn.Module):
    def __init__(self, T):
        super(SNNLoss, self).__init__()
        self.T = T

    def forward(self, x, y):
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

'''