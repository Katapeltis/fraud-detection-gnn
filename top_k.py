import torch

def get_top_k_positions(tensor_dict, tensor_type, k):

    if tensor_type == 'heterogenous':
        tensor = tensor_dict[list(tensor_dict.keys())[0]]
    else:
        tensor = tensor_dict
    
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

def analyze_tensor(tensor_dict, tensor_type, k):
    if tensor_type == 'heterogenous':
        tensor = tensor_dict[list(tensor_dict.keys())[0]]
    else:
        tensor = tensor_dict

    class_1 = torch.sum(tensor[:, 1] > tensor[:, 0]).item()
    class_0 = torch.sum(tensor[:, 0] > tensor[:, 1]).item()
    
    if k >10000:
        print(f"\nNumber of 0: {class_0}")
        print(f"Number of 1: {class_1}")
    
    #get top k indices per column (as separate lists)
    top_k_indices_col0 = torch.topk(tensor[:, 0], k).indices.tolist()
    top_k_indices_col1 = torch.topk(tensor[:, 1], k).indices.tolist()
    
    num_nodes = tensor.shape[0]
    num_classes = tensor.shape[1]

    indices = []
    for cls in range(num_classes):
        # pick k random indices for this class
        random_idx = torch.randperm(num_nodes)[:k]
        indices.append(random_idx)
    
    #return [top_k_indices_col0, top_k_indices_col1]
    return indices
