#include <iostream>
#include <thread>
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
#include "decoder.h"
#include "encoder.h"
using namespace std;

// Load the glove6b.txt file into a hash table
unordered_map<string, int> word_to_index;
unordered_map<int, string> index_to_word;
// vector<vector<double>>  encoder(string input);
// vector<vector<double>>  decoder(string input, vector<vector<double>> kv, string outputs);

int main(){
    create_fr_word_maps("fr_words.txt",word_to_index, index_to_word);
    vector<std::string> en_sentences;
    vector<std::string> fr_sentences;

    readCSV( "en-fr.csv", en_sentences, fr_sentences);

    FullyConnectedLayer* linear = new FullyConnectedLayer(50, 336520, 0.0001, 0); //336520
    Softmax1d* softmax = new Softmax1d();
    Decoder* decoder = new Decoder();
    Encoder* encoder = new Encoder();

    vector<vector<double>> encoder_out = encoder->forwards(en_sentences[0]);
    // take first row of decoder_out and pass it through linear layer then softmax

    // vector<vector<double>> decoder_out = decoder(fr_sentences[0], encoder_out);
    vector<vector<double>> decoder_out = decoder->forwards(fr_sentences[0], encoder_out);
    vector<double> linear_out = linear->forward(decoder_out[decoder_out.size()-1]);
    vector<double> predictions = softmax->forward(linear_out);

    double max = INT32_MIN;
    int pred_idx = 0;
    for(int i=0; i < predictions.size(); i++){
        if(predictions[i] > max){
            max = predictions[i];
            pred_idx = i;
        }
    }
    cout << pred_idx << " " << max << endl;
    cout << index_to_word[pred_idx] << endl;
    // cout << word_to_index["salut"] << endl;
    vector<double> dLdZ(predictions.size(), 0);
    dLdZ[word_to_index["salut"]] = -1 / (predictions[word_to_index["salut"]]+1e-8);
    dLdZ = softmax->backwards(dLdZ);
    dLdZ = linear->backwards(dLdZ);
    // vector<vector<double>> dLdZ_2d = decoder->backwards(dLdZ);
    // cout << dLdZ_2d.size() << endl;
    // encoder->backwards(dLdZ_2d);


    return 1;
}
