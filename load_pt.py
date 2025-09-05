import torch
# Load the .pt file
# data = torch.load('/shared_mnt/github_repos/KGNET/Datasets/mid-cd69d7dc9fccd44ef4f1a5793f6c9e9f1df779894521821019f1f1de3295ee69/processed/geometric_data_processed.pt')
data = torch.load('/shared_mnt/github_repos/KGNET/Datasets/mid-cd69d7dc9fccd44ef4f1a5793f6c9e9f1df779894521821019f1f1de3295ee69/processed/pre_transform.pt')

# Inspect the loaded data
print(type(data))  # Check the type of data (e.g., dict, tensor, etc.)
print(data) 