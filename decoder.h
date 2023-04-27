#include <iostream>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>
#include <random>
#include <cctype>

#include "transformer.h"
using namespace std;

#pragma once

class Decoder{
public:
    Attention* attn1 = new Attention(50);
    Attention* attn2 = new Attention(50);
    Attention* attn3 = new Attention(50);
    vector<vector<double>> dLdZ_encoder;
    int kv_rows;
    int count=0;

    Decoder()
    {}

    vector<vector<double>>  forwards(const string input, vector<vector<double>> kv) {
        kv_rows = kv.size();
        dLdZ_encoder = vector<vector<double>>(kv.size(), vector<double>(kv[0].size(), 0.0));
 
        vector<vector<double>> embedded_words; // will be the length of the number of words + punctuations in sentence
        // get list of encoded words
        string word = "";
        for (int i = 0; i < input.length(); i++) {
            if (input[i] == ' ' || ispunct(input[i]) || i == (input.length()-1)) {
                if (hash_table.count(word)) {
                    vector<double> embedding = hash_table[word];
                    embedded_words.push_back(embedding);
                }
                word = "";
                if(ispunct(input[i])){
                    vector<double> embedding = hash_table[string(1, input[i])];
                    embedded_words.push_back(embedding);
                }
            } else {
                word += input[i];
            }
        }

        vector<vector<double>> pos_encoded_words(embedded_words.size(), vector<double>(embedded_words[0].size()));

        for(int word_idx=0; word_idx<embedded_words.size(); word_idx++){

            vector<double> position_bias = get_positional_encoding(embedded_words[0].size(),word_idx);
            // vector<double> embedded_word = hash_table[input];
            for(int j=0; j<embedded_words[0].size(); j++){
                pos_encoded_words[word_idx][j] = embedded_words[word_idx][j] + position_bias[j]; // add position bias to take into account where in sentence word is
            }
        }


        vector<vector<double>> residual_1 = pos_encoded_words;
        vector<vector<double>> q_vals = attn1->single_head_attn(pos_encoded_words);
        vector<vector<double>> q_val_norm = attn1->add_and_norm(q_vals); 
        vector<vector<double>> residual_2 = q_val_norm;
        vector<vector<double>> mha_out = attn2->attn_k_v_q(kv, kv,q_val_norm);
        vector<vector<double>> mha_norm = attn2->add_and_norm(mha_out); 
        vector<vector<double>> residual_3 = mha_norm;
        vector<vector<double>> feed_forward_out = attn3->feed_forward(mha_norm);
        vector<vector<double>> ffw_norm = attn3->add_and_norm(feed_forward_out); 

        // take first row of ffw_norm and pass it through linear layer then softmax

        return ffw_norm; 
    }

    vector<vector<double>> backwards(vector<double> dLdZ_short){// returns dLdZ used in encoder

        // first "Feed Forward" hit in backprop is passed only row 1 of add & norm to update itself
        // then the raw add & norm derivative is also passed to the previous add and norm layer 
        vector<vector<double>> dLdZ = attn3->update_last_norm(dLdZ_short);
        attn3->feed_backwards(dLdZ); // we dont need this derivative so dont assign it
        dLdZ = attn2->update_norm(dLdZ);
        vector<vector<vector<double>>> derivatives = attn2->attn_backwards(dLdZ);

        vector<vector<double>> dWk = derivatives[0];
        vector<vector<double>> dLdV = derivatives[1];
        vector<vector<double>> dWq = derivatives[2];
        for(int i=0; i<dLdZ_encoder.size(); i++){
            for(int j=0; j<dLdZ_encoder[0].size(); j++){
                dLdZ_encoder[i][j] += dWk[i][j] + dLdV[i][j]; // average weights back to encoder
                count++;
            }
        }

        dLdZ = attn1->update_norm(dWq);
        derivatives = attn1->attn_backwards(dLdZ);

        dWk = derivatives[0];
        dLdV = derivatives[1];
        dWq = derivatives[2];

        attn1->update_single_head(dWk, dLdV, dWq);
        
        return dLdZ_encoder;
    } 

};


