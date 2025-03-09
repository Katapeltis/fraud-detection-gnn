from dgl.data import AmazonCoBuy, FraudYelpDataset

raw_dir='/Users/marios/fraud_dgl'

# Load the Amazon dataset
# amazon_dataset = AmazonCoBuy("computers")
# amazon_graph = amazon_dataset[0]
# print("Amazon Graph:")
# print(amazon_graph)

# Load the Yelp dataset
def load_dataset(dataset):
    if dataset == 'yelp':
        yelp_dataset = FraudYelpDataset(raw_dir=raw_dir)
        graph = yelp_dataset[0]
    return graph