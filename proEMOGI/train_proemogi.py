import argparse, os, sys
import torch
import torch.nn.functional as F
import utils, gcnIO
from proemogi import proEMOGI

from scipy.sparse import lil_matrix
import scipy.sparse as sp

import numpy as np

def predict(model, edge_list, features, mask):
    model.eval()
    pred = model.predict(features[mask], edge_list)
    return pred

def fit_model(model, features, edge_list, 
              learning_rate, weight_decay,
              epochs, y_train, train_mask,
              y_val, val_mask, output_dir):
    """Fits a constructed model to the data.
    
    Trains a given GCN model for as many epochs as specified using
    the provided session. Metrics (ACC/LOSS/AUROC/AUPR) are monitored
    using the provided training and validation labels and masks.
    Early stopping can be used but seems to have problems with
    restoring the model so far.
    
    Parameters:
    ----------
    model:              An already initialized GCN model
    epochs:             The number of epochs to train for
    dropout_rate:       The percentage of connections set to 0 during
                        training to increase generalization
    y_train:            Vector of size n where n is the number of nodes.
                        A 1 in the vector stands for a positively labelled
                        node and a 0 for a negatively labelled one
    train_mask:         A vector similar to `y_train` but symbolizes nodes
                        that belong to the training set and 0 denotes all
                        other nodes
    y_val:              Similar vector to `y_train` but for the validation set
    val_mask:           Similar vector to `train_mask` but for the validation
                        set
    
    Returns:
    The trained model.
    """
    model_save_path = os.path.join(output_dir, 'model.pth')
    opt_list = [dict(params=model.layers[0].parameters(), weight_decay=weight_decay)] + \
               [dict(params=model.layers[i].parameters(), weight_decay=0) for i in range(1, len(model.layers))]
    optimizer = torch.optim.Adam(opt_list, lr=learning_rate)  # Only perform weight-decay on first convolution.
    
    def train(y, train_mask):
        model.train()
        optimizer.zero_grad()
        out = model(features[train_mask], edge_list)
        loss = F.nll_loss(out, y[train_mask])

        loss.backward()
        optimizer.step()

        return loss.item()


    @torch.no_grad()
    def val(y, val_mask):
        model.eval()
        pred = model.predict(features[val_mask], edge_list)
        acc = pred.eq(y[val_mask]).sum().item() / val_mask.sum().item()
        return acc
    
    best_val_acc = 0
    for epoch in range(epochs):
        loss = train(y_train, train_mask)
        train_acc = val(y_train, train_mask)
        val_acc = val(y_val, val_mask)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
        print(f'Epoch: {epoch}, Loss: {loss:.4f} Train: {train_acc:.2f}, Val: {val_acc:.2f}')
    
    model.save(model_save_path)
    return model


def train_gcn(data_path, n_support, hidden_dims, learning_rate,
              weight_decay, loss_multiplier, epochs, dropout_rate,
              output_dir, logging=True):
    """Train a GCN from some hyper parameters and a HDF5 container.
    
    Construct and fit a GCN model to data stored in a HDF5 container.
    This function encapsulates all Tensorflow stuff. It preprocesses
    features and the adjacency matrix, sets up a model using TF and
    writes predictions to a file in the output directory.
    
    Parameters:
    ----------
    data_path:          Path to a HDF5 container that contains a network,
                        features and train/test splits as data sets inside.
    n_support:          The order of neighborhoods (sometimes called radius)
                        of each node that is taken into account at each layer
    hidden_dims:        The architecture of the model. A list of integers,
                        specifying the number of convolutional kernels per
                        layer and the number of layers altogether
    learning_rate:      The initial learning rate for the adam optimizer
    weight_decay:       The rate of the weight decay
    loss_multiplier:    Times that a positive node (gene) is counted more often
                        than a negative node. Useful for imbalanced data sets
                        in which one class is much more frequent than the
                        other
    epochs:             Number of epochs to train for
    dropout_rate:       Percentage of connection to be set to 0 during training
                        (not evaluation). Improves generalization and robustness
                        of most neural networks
    output_dir:         The output directory. 

    Returns:
    Predictions. A probability for each node. This is the final prediction for
    all nodes in the network after training. The predictions are a vector and
    the nodes are in the same order as in the HDF5 container.
    """
    # load data and preprocess it
    input_data_path = data_path
    data = gcnIO.load_hdf_data(input_data_path, feature_name='features')
    adj, edge_list, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, node_names, feature_names = data
    print("Read data from: {}".format(input_data_path))

#     # preprocess features
#     num_feat = features.shape[1]
#     if num_feat > 1:
#         #features = utils.preprocess_features(lil_matrix(features))
#         #features = utils.sparse_to_tuple(lil_matrix(features))
#         # we dont row-normalize the data because it doesn't seem to benefit
#         # classification and the data is assumed to be normalized anyways.
#         # For different applications, you might consider normalization here.
#         pass
#     else:
#         print("Not row-normalizing features because feature dim is {}".format(num_feat))
#         #features = utils.sparse_to_tuple(lil_matrix(features))

    # get higher support matrices
    support, num_supports = utils.get_support_matrices(adj, n_support)
    hidden_dims = [int(x) for x in hidden_dims]
    
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # initialize model and metrics
    model = proEMOGI(input_dim=features.shape[1],
                     output_dim=len(np.unique(y_train)),
                     weight_decay=weight_decay,
                     dropout_rate=dropout_rate,
                     num_hidden_layers=len(hidden_dims),
                     hidden_dims=hidden_dims,
                     pos_loss_multiplier=loss_multiplier
    ).to(device)
    features, edge_list = torch.Tensor(features).to(device), torch.Tensor(edge_list).long().to(device)
    y_train, y_val = torch.Tensor(y_train).flatten().long().to(device), torch.Tensor(y_val).flatten().long().to(device)
    
    # fit the model
    model = fit_model(model, features, edge_list,  
                      learning_rate, weight_decay,
                      epochs, y_train, train_mask, 
                      y_val, val_mask, output_dir)
    
    predictions = predict(model, edge_list, features, test_mask)
    return predictions

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train EMPGI and save to file')
    parser.add_argument('-e', '--epochs', help='Number of Epochs',
                        dest='epochs',
                        default=100,
                        type=int
                        )
    parser.add_argument('-lr', '--learningrate', help='Learning Rate',
                        dest='lr',
                        default=.001,
                        type=float
                        )
    parser.add_argument('-s', '--support', help='Neighborhood Size in Convolutions',
                        dest='support',
                        default=1,
                        type=int
                        )
    parser.add_argument('-hd', '--hidden_dims',
                        help='Hidden Dimensions (number of filters per layer). Also determines the number of hidden layers.',
                        nargs='+',
                        dest='hidden_dims',
                        default=[50, 100])
    parser.add_argument('-lm', '--loss_mul',
                        help='Number of times, false negatives are weighted higher than false positives',
                        dest='loss_mul',
                        default=30,
                        type=float
                        )
    parser.add_argument('-wd', '--weight_decay', help='Weight Decay',
                        dest='decay',
                        default=5e-2,
                        type=float
                        )
    parser.add_argument('-do', '--dropout', help='Dropout Percentage',
                        dest='dropout',
                        default=.5,
                        type=float
                        )
    parser.add_argument('-d', '--data', help='Path to HDF5 container with data',
                        dest='data',
                        type=str,
                        required=True
                        )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if not args.data.endswith('.h5'):
        print("Data is not a hdf5 container. Exit now.")
        sys.exit(-1)

    output_dir = gcnIO.create_model_dir('./')
    predictions = train_gcn(data_path=args.data,
                            n_support=args.support,
                            hidden_dims=args.hidden_dims,
                            learning_rate=args.lr,
                            weight_decay=args.decay,
                            loss_multiplier=args.loss_mul,
                            epochs=args.epochs,
                            dropout_rate=args.dropout,
                            output_dir=output_dir,
                            logging=True)
