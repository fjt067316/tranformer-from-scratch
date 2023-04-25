#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include <cassert>
// #include <cmath>
#include "fcl.h"
#include "softmax.h"
#include "layer_norm.h"
using namespace std;

#pragma once


class Attention{
public:
    FullyConnectedLayer* k_fcl;
    FullyConnectedLayer* q_fcl;
    FullyConnectedLayer* v_fcl;
    FullyConnectedLayer* feed_forward_layer;
    FullyConnectedLayer* no_relu_feed_forward;
    Softmax* softmax;
    LayerNorm* layer_norm;
    vector<vector<double>> res;

    Attention(int vectorized_word_size){
        // https://stats.stackexchange.com/questions/421935/what-exactly-are-keys-queries-and-values-in-attention-mechanisms
        k_fcl = new FullyConnectedLayer(vectorized_word_size, vectorized_word_size, 0.001, 0);
        q_fcl = new FullyConnectedLayer(vectorized_word_size, vectorized_word_size, 0.001, 0);
        v_fcl = new FullyConnectedLayer(vectorized_word_size, vectorized_word_size, 0.001, 0);
        feed_forward_layer = new FullyConnectedLayer(vectorized_word_size, vectorized_word_size, 0.001, 0);
        no_relu_feed_forward = new FullyConnectedLayer(vectorized_word_size, vectorized_word_size, 0.001, 0, 0);
        softmax = new Softmax();
        layer_norm = new LayerNorm();
        

    }

    vector<vector<double>> single_head_attn(vector<vector<double>> pos_encoded_words){
        vector<vector<double>> k(pos_encoded_words.size(), vector<double>(50));
        vector<vector<double>> tmp(pos_encoded_words.size(),vector<double>(50, 0));

        vector<vector<double>> q(pos_encoded_words.size(), vector<double>(pos_encoded_words[0].size()));
        vector<vector<double>> v(pos_encoded_words.size(), vector<double>(pos_encoded_words[0].size()));
        res = pos_encoded_words;
        // v == pos_encoded_words
        for(int word_idx=0; word_idx<pos_encoded_words.size(); word_idx++){
            k[word_idx] = k_fcl->forward(pos_encoded_words[word_idx]);
            q[word_idx] = q_fcl->forward(pos_encoded_words[word_idx]);
            v[word_idx] = v_fcl->forward(pos_encoded_words[word_idx]);
        }


        // https://towardsdatascience.com/demystifying-efficient-self-attention-b3de61b9b0fb => "row-wise softmax" https://twitter.com/srush_nlp/status/1359582647522127889?lang=en 
        for(int word_idx=0; word_idx<pos_encoded_words.size(); word_idx++){
            for(int i=0; i<pos_encoded_words.size(); i++){ 
                for(int j =0; j<pos_encoded_words[0].size(); j++){
                    tmp[word_idx][i] += q[word_idx][j]+k[i][j]; // matrtix multiplication
                }
                tmp[word_idx][i] /= sqrt(pos_encoded_words[0].size());
            }
        }

        tmp = softmax->forward(tmp);

        vector<vector<double>> outputs(pos_encoded_words.size(), vector<double>(pos_encoded_words[0].size(), 0));

        for(int word_idx=0; word_idx<pos_encoded_words.size(); word_idx++){
            for(int i=0; i<pos_encoded_words[0].size(); i++){ 
                for(int j =0; j<pos_encoded_words.size(); j++){
                    outputs[word_idx][i] += tmp[word_idx][j] * v[j][i]; // matrtix multiplication
                }
            }
        }
        return outputs;
    }

    // https://stackoverflow.com/questions/74979359/how-is-position-wise-feed-forward-neural-network-implemented-for-transformers
    vector<vector<double>> add_and_norm(vector<vector<double>> input){

        for(int i=0; i<input.size(); i++){
            for(int j=0; j<input[0].size(); j++){
                input[i][j] += res[i][j];
            }
        }

        return layer_norm->forward(input);

    }

    vector<vector<double>> feed_forward(vector<vector<double>> input){
        res = input;
        for(int i=0; i<input.size(); i++){
            input[i] = no_relu_feed_forward->forward(feed_forward_layer->forward(input[i]));
        }
        return input;
    }
    
};