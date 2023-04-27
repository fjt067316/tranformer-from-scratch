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
    vector<vector<double>> v, q, k;

    Attention(int vectorized_word_size){
        // https://stats.stackexchange.com/questions/421935/what-exactly-are-keys-queries-and-values-in-attention-mechanisms
        k_fcl = new FullyConnectedLayer(vectorized_word_size, vectorized_word_size, 0.001, 0);
        q_fcl = new FullyConnectedLayer(vectorized_word_size, vectorized_word_size, 0.001, 0);
        v_fcl = new FullyConnectedLayer(vectorized_word_size, vectorized_word_size, 0.001, 0);
        feed_forward_layer = new FullyConnectedLayer(vectorized_word_size, vectorized_word_size, 0.001, 0);
        no_relu_feed_forward = new FullyConnectedLayer(vectorized_word_size, vectorized_word_size, 0.001, 0, 0);
        softmax = new Softmax();
        layer_norm = new LayerNorm(50, 0.001);
        

    }

    vector<vector<double>> single_head_attn(vector<vector<double>> pos_encoded_words){

        vector<vector<double>> tmp(pos_encoded_words.size(),vector<double>(50, 0));
        vector<vector<double>> q(pos_encoded_words.size(), vector<double>(pos_encoded_words[0].size()));
        vector<vector<double>> v(pos_encoded_words.size(), vector<double>(pos_encoded_words[0].size()));
        vector<vector<double>> k(pos_encoded_words.size(), vector<double>(50));
        


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

    vector<vector<double>> update_single_head(vector<vector<double>>wk, vector<vector<double>> wv, vector<vector<double>> wq){
        // k_fcl->backwards()
        vector<double> mean_dl(wk[0].size(), 0);

        for(int j=0; j<wk.size(); j++){
            for(int i=0; i<wk[0].size(); i++){
                mean_dl[i] += wk[j][i] / wk.size();
            }
        }

        k_fcl->backwards(mean_dl);


        for(int j=0; j<wv.size(); j++){
            for(int i=0; i<wv[0].size(); i++){
                mean_dl[i] += wv[j][i] / wv.size();
            }
        }

        v_fcl->backwards(mean_dl);


        for(int j=0; j<wq.size(); j++){
            for(int i=0; i<wq[0].size(); i++){
                mean_dl[i] += wq[j][i] / wq.size();
            }
        }

        q_fcl->backwards(mean_dl);

        return wq;

    }

    vector<vector<double>> attn_k_v_q(vector<vector<double>> k_in, vector<vector<double>> v_in, vector<vector<double>> q_in){
        v = v_in;
        k = k_in;
        q = q_in;
// https://towardsdatascience.com/transformers-explained-visually-part-3-multi-head-attention-deep-dive-1c1ff1024853
        
        // https://stats.stackexchange.com/questions/524043/what-are-exact-inputs-and-their-dimension-for-decoder-part-of-the-transformers
        // q_in has a diffferent row size than k_in or v_in
        vector<vector<double>> tmp(q_in.size(),vector<double>(k_in[0].size(), 0));

        res = q_in;
        // v == pos_encoded_words

        // https://towardsdatascience.com/demystifying-efficient-self-attention-b3de61b9b0fb => "row-wise softmax" https://twitter.com/srush_nlp/status/1359582647522127889?lang=en 
        for(int word_idx=0; word_idx<q_in.size(); word_idx++){
            for(int i=0; i<k_in.size(); i++){ 
                for(int j =0; j<q_in[0].size(); j++){
                    tmp[word_idx][i] += q_in[word_idx][j]+k_in[i][j]; // matrtix multiplication
                }
                tmp[word_idx][i] /= sqrt(k_in[0].size());
            }
        }

        tmp = softmax->forward(tmp);

        vector<vector<double>> outputs(q_in.size(), vector<double>(q_in[0].size(), 0));

        for(int word_idx=0; word_idx<q_in.size(); word_idx++){
            for(int i=0; i<q_in[0].size(); i++){ 
                for(int j =0; j<k_in.size(); j++){
                    outputs[word_idx][i] += tmp[word_idx][j] * v_in[j][i]; // matrtix multiplication
                }
            }
        }
        return outputs; // shape of q_in.size() x k_in[0].size()
    }

    vector<vector<vector<double>>> attn_backwards(vector<vector<double>> dLdZ){

        vector<vector<double>> dLdS(dLdZ.size(), vector<double>(v.size(), 0));

        for(int row=0; row<dLdZ.size();row++){
            for(int j=0; j<v.size(); j++){
                for(int i=0; i<v[0].size(); i++){
                    dLdS[row][j] += dLdZ[row][i]*v[j][i]; // dLdS = dLdZ * v^T
                }
            }
        }
        dLdS = softmax->backwards(dLdS);
       
       vector<vector<double>> dWq(dLdS.size(), vector<double>(k[0].size(), 0));
       for(int row=0; row<dWq.size(); row++){
            for(int col=0; col<dWq[0].size(); col++){
                for(int i=0; i<k.size(); i++){
                    dWq[row][col] = dLdS[row][i]*k[i][col]; // dWq = dLdS*k
                }
            }
       }

       vector<vector<double>> dLdV(v.size(), vector<double>(v[0].size(), 0));

       for(int r =0; r<dLdV.size();r++){
            for(int c =0; c<dLdV[0].size();c++){
                for(int i=0; i<dLdZ.size(); i++){
                    dLdV[r][c] += softmax->last_output[i][r]*dLdZ[i][c]; // softmax_output^T*dLdZ
                }
            }
       }

        vector<vector<double>> dWk(v.size(), vector<double>(k[0].size(), 0));

        for(int r =0; r<dWk.size();r++){
            for(int c =0; c<dWk[0].size();c++){
                for(int i=0; i<q.size(); i++){
                    dWk[r][c] += dLdS[i][r]*q[i][c]; // dLdWk=dLdS^T*q
                }
            }
       }
       
       vector<vector<vector<double>>> vkq;
       vkq.push_back(dWk);
       vkq.push_back(dLdV);
       vkq.push_back(dWq);
        return vkq;
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

    vector<vector<double>> update_norm(vector<vector<double>> dLdZ){

        return layer_norm->backward(dLdZ);
    }

    vector<vector<double>> update_last_norm(vector<double> dLdZ){

        vector<vector<double>> reshape_dldz(layer_norm->_input.size(), vector<double>(layer_norm->_input[0].size(), 0.0));
        
        for(int i=0; i<dLdZ.size(); i++){
            reshape_dldz[reshape_dldz.size()-1][i] = dLdZ[i];
        }

        return layer_norm->backward(reshape_dldz);
    }

    vector<vector<double>> feed_forward(vector<vector<double>> input){
        res = input;
        for(int i=0; i<input.size(); i++){
            input[i] = no_relu_feed_forward->forward(feed_forward_layer->forward(input[i]));
        }
        return input;
    }

    vector<double> feed_backwards(vector<vector<double>> dLdZ){
        vector<double> dLdZ_avg(dLdZ[0].size(), 0.0);

        for(int i=0; i<dLdZ.size(); i++){
            for(int j=0; j<dLdZ[0].size(); j++){
                dLdZ_avg[j] += dLdZ[i][j];
            }
        }

        for(int j=0; j<dLdZ[0].size(); j++){
            dLdZ_avg[j] /= dLdZ.size(); //avg it
        }
        return no_relu_feed_forward->backwards(dLdZ_avg);
    }
    
};