from neural_network import *

import numpy as np
import random
import sys
sys.path.append('../')
from utils import *
    
def nnLoadMats(train_matrix):
    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return train_matrix, zero_train_matrix

def nnHelper(train_matrix, val, k, lr, lam):
    num_epoch = 80
    
    train_matrix, zero_train_matrix = nnLoadMats(train_matrix)

    model = AutoEncoder(train_matrix.shape[1], k=k)
    train(model, lr, lam, train_matrix, zero_train_matrix,
        val, num_epoch)
    return zero_train_matrix, model

def nn_helper_eval(model, train_data, val):
    model.eval()

    pred = []


    for i, u in enumerate(val["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        pred.append(output[0][val["question_id"][i]].item())
    return pred

def main():
    np.random.seed(42)
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")
    
    preds = []
    test_preds = []
    real_pred = []
    real_test_pred = []
    for i in range(3):
        sample = np.random.choice(sparse_matrix.shape[0], sparse_matrix.shape[1], replace=True)
        new_sample = sparse_matrix[sample, ]
        
        
        zero_mat, model = nnHelper(new_sample, val_data, 10, 0.001, 0)
        preds.append(nn_helper_eval(model, zero_mat, val_data))
        test_preds.append(nn_helper_eval(model, zero_mat, test_data))
    
    for i in range(len(preds[0])):
        sum = 0
        
        for pred in preds:
            sum += pred[i]
        real_pred.append(sum / 3 >= 0.5)

    real_accuracy = np.sum((val_data["is_correct"] == np.array(real_pred))) \
           / len(val_data["is_correct"])    
    
    print("Real val accuracy: " + str(real_accuracy))

    for i in range(len(test_preds[0])):
        sum = 0

        for pred in preds:
            sum += pred[i]
        real_test_pred.append(sum / 3 >= 0.5)

    test_acc =  np.sum((test_data["is_correct"] == np.array(real_test_pred))) \
           / len(test_data["is_correct"])
    
    print("Real test accuracy: " + str(test_acc))


if __name__ == "__main__":
    main()