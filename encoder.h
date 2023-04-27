#include <iostream>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>
#include <random>
#include <cctype>

#include "helper.h"
#include "fcl.h"
#include "softmax.h"
#include "transformer.h"
#include "attention.h"
#include "layer_norm.h"
using namespace std;

#pragma once


vector<vector<double>> encoder(string input) {

    // string input = "hello my name is josh what is your name?";
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

    Attention* attn = new Attention(50);
// https://glassboxmedicine.com/2019/08/15/the-transformer-attention-is-all-you-need/

    vector<vector<double>> attn_out = attn->single_head_attn(pos_encoded_words);
    vector<vector<double>> sub_l1 = attn->add_and_norm(attn_out);
    vector<vector<double>> feed_forward_out = attn->feed_forward(sub_l1);
    vector<vector<double>> sub_l2 = attn->add_and_norm(feed_forward_out);
// #####################    ENCODER FINISHED    #################

    // pos_encoded_words.size() is the number of words+puncts in input

    // encoding finished
    // encoded_word will become encoded_words and have an array of encodings for every word in a input

    return sub_l2;
}
