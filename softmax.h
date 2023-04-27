#include <vector>
#include <cmath>
#include <cstdlib>
#include <random>
#include <iostream>
#include <cassert>

using namespace std;

#pragma once

class Softmax1d {
public:
    vector<double> last_input;
    // string tag = "smax";


    Softmax1d(){
        return;
    }

    vector<double> forward(vector<double> input) { // 1d input
        last_input = input;
        double max = INT32_MIN;
        // input.print();

        for (int i = 0; i < input.size(); i++) {
            max = (max > input[i]) ? max : input[i];
        }

        double sum = 0.0;
        for (int i = 0; i < input.size(); i++) {
            sum += exp(input[i]-max);
        }

        for (int i = 0; i < input.size(); i++) {
            input[i] = exp(input[i]-max) / sum;
        }

        return input;
    }

    vector<double> backwards(vector<double> dLdZ){ // cross_entropy dLdZ = -1/p
    // https://github.com/AlessandroSaviolo/CNN-from-Scratch/blob/master/src/layer.py
    // https://victorzhou.com/blog/intro-to-cnns-part-2/
        // dLdZ.print();
        vector<double> dLdZ_exp(dLdZ.size());
        vector<double> dout_dt(dLdZ.size()); // dout_dt is dLdZ next layer
        double sum_exp = 0.0;
        int label_idx;

        for(int i=0; i < last_input.size(); i++){
            dLdZ_exp[i] = exp(last_input[i]); 
            sum_exp += dLdZ_exp[i];       
            if(dLdZ[i] < 0){ // answer selected will be negative
                label_idx = i;
            }
        }
        // i is the label index
        for(int i=0; i < last_input.size(); i++){
            dout_dt[i] = -dLdZ_exp[label_idx]*dLdZ_exp[i] / (sum_exp*sum_exp);
        }
        
        dout_dt[label_idx] = dLdZ_exp[label_idx] * (sum_exp - dLdZ_exp[label_idx]) / (sum_exp * sum_exp);

        for(int i=0; i < last_input.size(); i++){
            dout_dt[i] *= dLdZ[label_idx];
        }
        return dout_dt;
    }
};

// THIS is a modified softmax which takes in a 2d vector input and computes a row wise softmax on the vector
class Softmax {
public:
    vector<vector<double>> last_input;
    vector<vector<double>> last_output;

    Softmax(){
        return;
    }

    vector<vector<double>> forward(vector<vector<double>> input) { // 2D input
        last_input = input;
        vector<vector<double>> output(input.size(), vector<double>(input[0].size()));
        vector<double> row_max(input.size(), INT32_MIN);

        for (int i = 0; i < input.size(); i++) {
            for (int j = 0; j < input[0].size(); j++) {
                row_max[i] = (row_max[i] > input[i][j]) ? row_max[i] : input[i][j];
            }
        }
        
        for (int i = 0; i < input.size(); i++) {
            double sum = 0.0;
            for (int j = 0; j < input[0].size(); j++) {
                sum += exp(input[i][j]-row_max[i]);
            }
            for (int j = 0; j < input[0].size(); j++) {
                output[i][j] = exp(input[i][j]-row_max[i]) / sum;
            }
        }

        last_output = output;

        return output;
    }

    vector<vector<double>> backwards(vector<vector<double>> dLdZ){ // cross_entropy dLdZ = -1/p
        vector<vector<double>> dLdZ_exp(dLdZ.size(), vector<double>(dLdZ[0].size()));
        vector<vector<double>> dout_dt(dLdZ.size(), vector<double>(dLdZ[0].size()));
        vector<double> sum_exp(dLdZ.size(), 0.0);
        vector<int> output_idx;

        for(int i=0; i < last_input.size(); i++){
            for(int j=0; j < last_input[0].size(); j++){
                dLdZ_exp[i][j] = exp(last_input[i][j]); 
                sum_exp[i] += dLdZ_exp[i][j];
                if(dLdZ[i][j] < 0){ // answer selected will be negative
                    output_idx[i] = j;
            }       
            }
        }

        for(int i=0; i < last_input.size(); i++){
            for(int j=0; j < last_input[0].size(); j++){
                dout_dt[i][j] = -dLdZ_exp[i][output_idx[i]]*dLdZ_exp[i][j] / (sum_exp[i]*sum_exp[i]);
            }
        }
        for(int i=0; i < last_input.size(); i++){
            dout_dt[i][output_idx[i]] = dLdZ_exp[i][output_idx[i]]*(sum_exp[i]-dLdZ_exp[i][output_idx[i]])/(sum_exp[i]*sum_exp[i]);
        }
        
        for(int i=0; i < last_input.size(); i++){
            for(int j=0; j < last_input[0].size(); j++){
                dout_dt[i][j] *= dLdZ[i][output_idx[i]];
            }
        }

        return dout_dt;
    }

private:
    double output_delta(int i, int j, int size) {
        if (i == j) {
            return 1 - (1 / size);
        } else {
            return -(1 / size);
        }
    }
};
