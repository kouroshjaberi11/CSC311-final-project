from item_response import *
from knn import *
from neural_network import *

import numpy as np
import random
import sys
sys.path.append('../')
from utils import *

def irt_helper(mat, val):
    learn = 0.05
    iteration = 4
    theta, beta, _ = irt (mat, val, learn, iteration)

    return theta, beta

def irt_eval_helper(theta, beta, val):
    pred = []
    for i, q in enumerate(val["question_id"]):
        u = val["user_id"][i]
        x = (theta[u] - beta[q])
        p_a = sigmoid(x)
        pred.append(p_a)
    return pred

def kNNHelper(mat, data, threshold=0.5):
    k = 11
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(mat.T)
    pred = []
    
    for i in range(len(data["is_correct"])):
        cur_user_id = data["user_id"][i]
        cur_question_id = data["question_id"][i]
        pred.append(mat[cur_question_id, cur_user_id])
    
    return pred
    
def nnLoadMats(train_matrix):
    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return train_matrix, zero_train_matrix

def nnHelper(train_matrix, val, k, lr, lam):
    # k = 10
    num_epoch = 20
    # lr = 0.005
    # lam = 0
    
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
    print("hi")
    np.random.seed(42)
    sparse_matrix = load_train_sparse("../data").toarray()
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    ln = sparse_matrix.shape[0]
    
    preds = []
    real_pred = []
    print(len(val_data['is_correct']))
    for i in range(3):
        sample = np.random.choice(sparse_matrix.shape[0], sparse_matrix.shape[1], replace=True)
        new_sample = sparse_matrix[sample, ]
        
        
        zero_mat, model = nnHelper(new_sample, val_data, 10, 0.005, 0)
        preds.append(nn_helper_eval(model, zero_mat, val_data))
        # elif (i == 1):
        #     print("hi")
        #     # theta, beta = irt_helper(new_sample, val_data)
        #     # preds.append(irt_eval_helper(theta, beta, val_data))
        #     zero_mat, model = nnHelper(new_sample, val_data, 50, 0.001, 0.1)
        #     preds.append(nn_helper_eval(model, zero_mat, val_data))
        # else:
        #     preds.append(kNNHelper(sparse_matrix, val_data))
    
    for i in range(len(preds[0])):
        sum = 0
        
        for pred in preds:
            sum += pred[i]
        real_pred.append(sum / 3 > 0.5)
    print(real_pred.count(0))
    print(real_pred.count(1))

    ones = 0
    zs = 0
    kones = 0
    zones = 0
    for i in range(len(preds[0])):
        if preds[0][i] < 0.5:
            zs+=1
        else:
            ones+=1

        if preds[1][i]< 0.5:
            kones+=1
        else:
            zones+=1
    print(zones)
    print(kones)
    print(ones)
    print(zs)
    real_accuracy = np.sum((val_data["is_correct"] == np.array(real_pred))) \
           / len(val_data["is_correct"])    
    
    print("Real accuracy: " + str(real_accuracy))


if __name__ == "__main__":
    main()