from matplotlib import pyplot as plt
from torch.autograd import Variable
from copy import deepcopy

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch
import sys
sys.path.append('../')
from utils import *


def load_data(base_path="../data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, i=100, j=300, k=200, h1=nn.Tanh(), h2=nn.Tanh()):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, i)
        nn.init.kaiming_uniform_(self.g.weight, nonlinearity="sigmoid")
        self.enc1 = nn.Linear(i, j)
        self.enc2 = nn.Linear(j, k)

        self.encoder = nn.Sequential(self.g, 
                                nn.Sigmoid(),
                                self.enc1,
                                h1,
                                self.enc2,
                                h2)


        self.dec1 = nn.Linear(k, j)
        self.dec2 = nn.Linear(j, i)
        self.h = nn.Linear(i, num_question)

        self.decoder = nn.Sequential(self.dec1,
                                h2,
                                self.dec2,
                                h1,
                                self.h,
                                nn.Sigmoid())

    def contractive(self, inputs):
        return torch.norm(torch.autograd.functional.jacobian(self.encoder, inputs))

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################
        # TODO:                                                             #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################
        out = nn.Sequential(self.encoder, self.decoder) (inputs)

        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # TODO: Add a regularizer to the cost function. 
    
    # Tell PyTorch you are training the model.
    

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    num_student = train_data.shape[0]

    val_losses = []
    train_losses = []

    valid_acc = 0

    for epoch in range(0, num_epoch):
        train_loss = 0.
        val_loss = 0.
        model.train()
        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.)
            loss.backward()

            train_loss += loss.item() + lamb * model.contractive(inputs)
            optimizer.step()
        train_losses.append(train_loss.detach())
        model.eval()
        for i, u in enumerate(valid_data["user_id"]):
            inputs = Variable(zero_train_data[u]).unsqueeze(0)
            target = inputs.clone()
            output = model(inputs)


            nan_mask = np.isnan(train_data[u].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            loss = torch.sum((output - target) ** 2.)
            loss.backward()

            val_loss += loss.item()

        val_losses.append(val_loss)


        valid_acc = evaluate(model, zero_train_data, valid_data)
        print("Epoch: {} \tTraining Cost: {:.6f}\t Val Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, val_loss, valid_acc))
        
    #Plot and report:
    # losses = [train_losses, val_losses]
    # labels = ["Train Losses", "Val Losses"]
    # for i in range(len(losses)):
    #     plt.figure()
    #     plt.plot(range(0, epoch+1), losses[i], label=labels[i])

    #     plt.xlabel("Epoch")
    #     plt.ylabel("Cost")
    #     plt.title("Training and Validation objective changes")
    #     plt.legend()
    #     plt.grid(True)
    #     plt.show()

    return valid_acc
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,do you notice much difference between tanh and sigmoid for regression problems
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # TODO:                                                             #
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters. 10, 50, 100, 200, 500
    i_s = [5, 10, 50, 100, 200, 500]
    j_s = [5, 10, 50, 100, 200, 500]
    k_s = [5, 10, 50, 100, 200, 500]
    hidden = [nn.Tanh(), nn.Sigmoid(), nn.ReLU(), nn.Softmax(), nn.LeakyReLU(), nn.ELU()]
    # Optimal Parameters
    opt_k = -1 # 10
    opt_j = -1 # 20
    opt_i = -1 # 50
    opt_h1 = None
    opt_h2 = None
    
    learn = 0.01
    epoch = 70
    lam = 1

    best_model = None
    best_acc = 0
    # for lam in lambs:
    for h1 in hidden:
        for h2 in hidden:
            for i in i_s:
                for j in j_s:
                    for k in k_s:
                        if k <= j <= i:

                            print("i: {0}, j: {1}, k: {2}, learn: {3}, lambda: {4}, h1: {5}, h2: {6}".format(i, j, k, learn, lam, h1, h2))
                            model = AutoEncoder(train_matrix.shape[1], i=i, j=j, k=k, h1=h1, h2=h2)

                            # Set optimization hyperparameters.
                            lr = learn
                            num_epoch = epoch
                            lamb = lam

                            acc = train(model, lr, lamb, train_matrix, zero_train_matrix,
                                valid_data, num_epoch)

                            if acc > best_acc:
                                best_acc = acc
                                best_model = deepcopy(model)
                                opt_i, opt_j, opt_k = i, j, k
                                opt_h1, opt_h2 = h1, h2

    print("===== BEST =====")
    print("i: {0}, j: {1}, k: {2}, learn: {3}, lambda: {4}, h1: {5}, h2: {6}".format(opt_i, opt_j, opt_k, learn, lam, opt_h1, opt_h2))
    
    test_acc = evaluate (best_model, zero_train_matrix, test_data)

    print("test accuracy: " + str(test_acc))

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
