#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <cassert>
// #include <cmath>
using namespace std;

#pragma once

// Function to load the glove6b.txt file into a hash table
unordered_map<string, vector<double>> load_embedding_map(string filename) {
    unordered_map<string, vector<double>> hash_table;

    ifstream infile(filename);
    if (!infile) {
        cerr << "Error opening file " << filename << endl;
        return hash_table;
    }

    string line;
    while (getline(infile, line)) {
        stringstream ss(line);
        string word;
        ss >> word;

        vector<double> vec;
        double value;
        while (ss >> value) {
            vec.push_back(value);
        }

        hash_table[word] = vec;
    }

    infile.close();

    return hash_table;
}




// Attention 
// https://www.youtube.com/watch?v=W2rWgXJBZhU
/*
// https://glassboxmedicine.com/2019/08/15/the-transformer-attention-is-all-you-need/

nn.Embedding consists of a weight matrix W that will transform a one-hot vector into a real-valued vector.
*/

// https://jalammar.github.io/illustrated-transformer/ 
vector<double> get_positional_encoding(int embedding_size, int word_pos ){
    vector<double> position_encoding(embedding_size);

    for(int i=0; i < embedding_size; i++){
        if(i%2==0){
            position_encoding[i] = sin(word_pos/pow(10000, 2*i/embedding_size)); // size of embedded word vector ie 50
        } else{
            position_encoding[i] = cos(word_pos/pow(10000, 2*i/embedding_size));
        }
    }

    return position_encoding;
}

/*

one hot encode words, as many words as inputs to fully connected layer
fully connected kayer has 512 outputs

*/

/*


class Attention:
    def __init__(self, input_dim, attention_dim):
        self.input_dim = input_dim
        self.attention_dim = attention_dim
        self.W = np.random.randn(input_dim, attention_dim)
        self.b = np.zeros(attention_dim)
        self.V = np.random.randn(attention_dim, 1)
    
    def forward(self, X):
        self.X = X
        self.Z = np.tanh(np.dot(X, self.W) + self.b)
        self.scores = np.dot(self.Z, self.V)
        self.attention_weights = np.exp(self.scores) / np.sum(np.exp(self.scores), axis=0)
        self.context_vector = np.sum(X * self.attention_weights, axis=0)
        return self.context_vector
    
    def backward(self, dL_dy):
        dL_dZ = dL_dy * self.attention_weights.T
        dL_dV = np.dot(self.Z.T, dL_dy)
        dL_dZ *= (1 - np.power(self.Z, 2))
        dL_dW = np.dot(self.X.T, dL_dZ)
        dL_db = np.sum(dL_dZ, axis=0)
        dL_dX = np.dot(dL_dZ, self.W.T)
        return dL_dX, dL_dW, dL_db, dL_dV
        
        
*/