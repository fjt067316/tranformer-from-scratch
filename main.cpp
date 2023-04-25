#include <iostream>
#include <thread>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
#include <cmath>
#include <random>
#include <cctype>

#include "fcl.h"
#include "softmax.h"
#include "transformer.h"
#include "attention.h"
#include "layer_norm.h"
using namespace std;

// Load the glove6b.txt file into a hash table
unordered_map<string, vector<double>> hash_table = load_embedding_map("glove.6B/glove.6B.50d.txt"); 

vector<vector<double>>  encoder(string input);
vector<vector<double>>  decoder(string input, vector<vector<double>> kv);

int main(){
    vector<vector<double>> encoder_out = encoder("hello my name is josh what is your name?");
    cout << encoder_out.size() << endl;
    vector<vector<double>> decoder_out = decoder("my name is", encoder_out);
    cout << decoder_out.size() << endl;

    return 1;
}

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

vector<vector<double>>  decoder(const string input, vector<vector<double>> kv){
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
    vector<vector<double>> q_vals = attn->single_head_attn(pos_encoded_words);

    vector<vector<double>> q_val_norm = attn->add_and_norm(q_vals); 
    vector<vector<double>> mha_out = attn->attn_k_v_q(kv, kv,q_val_norm);
    vector<vector<double>> mha_norm = attn->add_and_norm(mha_out); 
    vector<vector<double>> feed_forward_out = attn->feed_forward(mha_norm); 
    vector<vector<double>> ffw_norm = attn->add_and_norm(feed_forward_out); 

    return ffw_norm;

    
}
