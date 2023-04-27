#include <vector>
#include <cmath>
#include <cstdlib>
#include <random>
#include <iostream>
#include <cassert>

using namespace std;

#pragma once
// https://www.youtube.com/watch?v=1JmZ5idFcVI 
// Layer norm in transformers might be done row by row and not by entire layer ????
class LayerNorm {
public:
    vector<double> gamma;
    vector<double>  beta;
    vector<vector<double>> _input;
    float learning_rate;
    double variance;

    LayerNorm(int inputs_cols, float learning_rate=0.001) : 
    gamma(inputs_cols, 1),
    beta(inputs_cols, 0)
     {}

    vector<vector<double>> forward(vector<vector<double>> input) {
        _input = input;
        // Compute mean and variance of the input
        double mean = 0.0;
        variance = 0.0;
        for (int i = 0; i < input.size(); i++) {
            for(int j=0; j<input[0].size(); j++){
                mean += input[i][j];
            }
        }
        mean /= input.size()*input[0].size();

        for (int i = 0; i < input.size(); i++) {
            for(int j=0; j<input[0].size(); j++){
                variance += pow((input[i][j]-mean), 2);
            }
        }

        variance /= input.size()*input[0].size();

        for (int i = 0; i < input.size(); i++) {
            for(int j=0; j<input[0].size(); j++){
                input[i][j] = (input[i][j]-mean)/(variance+1e-9);
                input[i][j] *= gamma[j];
                input[i][j] += beta[j];
            }
        }
        return input;
    }

    vector<vector<double>> backward(vector<vector<double>> dLdZ) {
        vector<double> dLdG(gamma.size(),0);
        vector<double> dLdB(beta.size(),0);
        double factor = dLdZ.size();


        for (int i = 0; i < dLdZ.size(); i++) {
            for(int j=0; j < dLdZ[0].size(); j++){
                dLdG[j] += (_input[i][j]*dLdZ[i][j]);
            }
        }
        
        for (int i = 0; i < dLdZ.size(); i++) {
            for(int j=0; j < dLdZ[0].size(); j++){
                dLdB[j] += dLdZ[i][j];
            }
        }

        for(int j=0; j<gamma.size(); j++){
            dLdB[j] /= factor;
            dLdG[j] /= factor;
        }

        for (int i = 0; i < dLdZ.size(); i++) {
            for(int j=0; j < dLdZ[0].size(); j++){
                dLdZ[i][j] = dLdZ[i][j]*gamma[j] / sqrt(variance+1e-8);
            }
        }

        for(int i=0; i<beta.size(); i++) {
            beta[i] -= learning_rate * dLdB[i];
            gamma[i] -= learning_rate * dLdG[i];
        }

        return dLdZ;
    }

};
