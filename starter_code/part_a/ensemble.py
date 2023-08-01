from item_response import *
from knn import *
from neural_network import *

import numpy as np
import random
import sys
sys.path.append('../')
from utils import *

def main():
    print("hi")
    train_data = load_train_csv("../data")
    
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    ln = len(train_data.get('user_id'))

    for i in range(3):
        sample = random.choices(range(ln), ln)

        new_sample = {
            'user_id': [train_data.get('user_id')[i] for i in sample],
            'question_id': [train_data.get('question_id')[i] for i in sample],
            'is_correct': [train_data.get('is_correct')[i] for i in sample]
        }
    


if __name__ == "__main__":
    main()