import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm

def train(diffs_a, diffs_b, device, hidden_dim, num_epochs):
    norm_a = 0
    norm_b = 0
    for i in diffs_a:
        norm_a += i.norm()

    for i in diffs_b:
        norm_b += i.norm()

    norm_a /= len(diffs_a)
    norm_b /= len(diffs_b)

    steering_a = torch.randn((1, hidden_dim),
                             requires_grad=True, device=device)
    steering_b = torch.randn((1, hidden_dim),
                             requires_grad=True, device=device)

    optimizer = optim.Adam([steering_a, steering_b], lr = 1e-3)
    num_samples = len(diff_a)

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}')
        for i in tqdm(range(num_samples)):
            optimizer.zero_grad()
            loss_a = (1 - F.cosine_similarity(steering_a, diff_a[i], eps=1e-6))
            loss_b = (1 - F.cosine_similarity(steering_b, diff_b[i], eps=1e-6))
            loss = loss_a + loss_b
            loss.backward()
            optimizer.step()

    return F.normalize(steering_a) * norm_a,\
           F.normalize(steering_b) * norm_b

