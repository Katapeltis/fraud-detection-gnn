import torch

def get_top_k_positions(tensor_dict, key, k):
    tensor = tensor_dict[key]
    
    class_1 = torch.sum(tensor[:, 1] > tensor[:, 0]).item()
    class_0 = torch.sum(tensor[:, 0] > tensor[:, 1]).item()
    
    top_k_per_column = []
    for col in range(tensor.shape[1]):  # Iterate over each column
        column_values = tensor[:, col]
        top_k_values, top_k_indices = torch.topk(column_values, k)
        top_k_per_column.append({
            'column': col,
            'top_k_values': top_k_values.tolist(),
            'indices': top_k_indices.tolist()  # Row indices
        })
    
    return class_0, class_1, top_k_per_column

# # Example usage:
# tensor_dict = {'review': torch.tensor([[0.3931, 0.5371],
#                                      [0.3911, 0.4920],
#                                      [0.3913, 0.5365]])}
# k = 2
# class0, class1, top_k_results = get_top_k_positions(tensor_dict, 'review', k)
# print(f"Number of rows with second element > first: {count}")
# print("Top k per column:", top_k_results)

def analyze_tensor(tensor_dict, key, k):
    tensor = tensor_dict[key]
    
    # 1. Count rows where second element > first element
    count = 1
    
    # 2. Get top k indices per column (as separate lists)
    top_k_indices_col0 = torch.topk(tensor[:, 0], k).indices.tolist()
    top_k_indices_col1 = torch.topk(tensor[:, 1], k).indices.tolist()
    
    return count, [top_k_indices_col0, top_k_indices_col1]

# count, top_k_indices = analyze_tensor(tensor_dict, 'review', 2)
# print(f"Top k row indices (col0): {top_k_indices[0]}")
# print(f"Top k row indices (col1): {top_k_indices[1]}")