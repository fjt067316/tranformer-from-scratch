#include <vector>
#include <cmath>
#include <cstdlib>
#include <random>
#include <iostream>
#include <cassert>
#include <queue> 
#include "activation_functions.h"
using namespace std;

#pragma once

class AdamFCL{
public:
    vector<vector<double>>  m_dw, v_dw;
    vector<double> m_db, v_db;
    double beta1, beta2, epsilon, learning_rate, initial_b1, initial_b2;
    int t = 0;
    int counter=0;
    int iterations = 100;
    // double decay_rate = 0.8;
    // double gamma_init = 0.00001;
    // double clip_norm = 1;


    AdamFCL(double rows, double cols, double learning_rate, double beta1=0.9, double beta2=0.999, double epsilon=1e-8) :
    m_dw(rows, vector<double>(cols, 0)),
    v_dw(rows, vector<double>(cols, 0)),
    m_db(rows, 0),
    v_db(rows, 0)
    {
       
        this->beta1 = beta1;
        this->initial_b1 = beta1;
        this->initial_b2 = beta2;
        this->beta2 = beta2;
        this->epsilon = epsilon;
        this->learning_rate = learning_rate;
    }

    void update( vector<vector<double>> *w, vector<double> *b, vector<vector<double>> dw, vector<double> db) { // t is current timestep
        // dw, db are what we would usually update params with gradient descent
        // printArray(m_dw[0], 10);
        // printArray(dw[0], 10);
        // counter++;
        // if((counter % iterations) == 0){
        //     // beta1 *= initial_b1;
        //     // beta2 *= initial_b2;
        // }
        this->t++;

        // momentum beta 1
        // weights
        for(int i=0; i< (*w).size(); i++){
            for(int j=0; j< (*w)[0].size(); j++){
                m_dw[i][j] = beta1 * m_dw[i][j] + (1 - beta1) * dw[i][j]; // biased momentum estimate
                v_dw[i][j] = beta2 * v_dw[i][j] + (1 - beta2) * pow(dw[i][j], 2); // bias corrected momentum estimate
            }
        }
                // printArray(m_dw[0], 10);

        // cout << m_dw[7][7] << endl;
        // print_vector(m_dw);
        // biases
        for(int i=0; i< (*w).size(); i++){
            m_db[i] = beta1 * m_db[i] + (1 - beta1) * db[i]; 
            v_db[i] = beta2 * v_db[i] + (1 - beta2) * pow(db[i], 2);
        }
        

        // rms beta 2
        // weights
        // biases
        vector<vector<double>> m_dw_corr(m_dw.size(), vector<double>(m_dw[0].size()));
        vector<vector<double>> v_dw_corr(m_dw.size(), vector<double>(m_dw[0].size()));
        vector<double> m_db_corr(m_dw.size());
        vector<double> v_db_corr(m_dw.size());

        double denom_mw = (1 - pow(beta1, t));
        double denom_vw = (1 - pow(beta2, t));
        // cout << denom_mw << endl;
        for (int i = 0; i < m_dw.size(); i++) {
            for (int j = 0; j < m_dw[i].size(); j++) {
                m_dw_corr[i][j] = m_dw[i][j] / denom_mw;
                v_dw_corr[i][j] = v_dw[i][j] / denom_vw;
            }
        }

        // bias correction
        double denom_mb = (1 - pow(beta1, t));
        double denom_vb = (1 - pow(beta2, t));

        for (int j = 0; j < m_dw.size(); j++) {
            m_db_corr[j] = m_db[j] / denom_mb;
            v_db_corr[j] = v_db[j] / denom_vb;
        }
        double clip_threshold = 0;
        // printArray(m_dw_corr[0], 10);
        // update weights and biases 
        // double gamma = gamma_init * decay_rate;
        for(int i=0; i< (*w).size(); i++){
            for(int j=0; j< (*w)[i].size(); j++){
                (*w)[i][j] -= learning_rate * (m_dw_corr[i][j] / (sqrt(v_dw_corr[i][j]) + epsilon));
                if(clip_threshold > 0){
                    (*w)[i][j] = min(max((*w)[i][j], -clip_threshold), clip_threshold); 
                }
            }
        }
        // printArray(m_db_corr, 10);
        // learning_rate = learning_rate * sqrt(denom_vb) / denom_mb;

        // b -= learning_rate *  dL/dZ
        for(int i=0; i < (*b).size(); i++){
            (*b)[i] -=learning_rate * (m_db_corr[i] / (sqrt(v_db_corr[i]) + epsilon)); 
            if(clip_threshold > 0){
                (*b)[i] = min(max((*b)[i], -clip_threshold), clip_threshold); 
            }
        }
        
        return;
    }
};


class FullyConnectedLayer {
public:
    // const string tag{"fcl"};
    int input_size;
    int output_size;
    vector<vector<double>> weights; // matrix of shape (output_size, input_size)
    vector<vector<bool>> prune_mask; // matrix of shape (output_size, input_size)

    vector<double> bias; // vector of shape (output_size)
    vector<double> input_matrix;
    // AdamFCL* adam;
    AdamFCL adam;
    bool adam_optimizer;
    double learning_rate;
    bool use_relu;


    FullyConnectedLayer(int input_size, int output_size, double learning_rate, bool adam_optimizer, bool use_relu=1) :
        input_size(input_size),
        output_size(output_size),
        prune_mask(output_size, vector<bool>(input_size, 1)),
        weights(output_size, vector<double>(input_size)),
        bias(output_size, 0.0),
        adam(output_size, input_size, learning_rate)
    {
        this->input_size = input_size;
        this->output_size = output_size;
        this->learning_rate = learning_rate;
        // adam = new AdamFCL(output_size, input_size);

        // Initialize weights with random values
        random_device rd;
        mt19937 gen(rd());
        normal_distribution<double> dist(0.0, sqrt(2.0 / input_size));

        for (auto& row : this->weights) {
            generate(row.begin(), row.end(), [&](){ return dist(gen); });
        }
    }

    vector<double> forward(vector<double> input_matrix) {
        this->input_matrix = input_matrix;
        vector<double> outputs(output_size);

        for (int i = 0; i < output_size; i++) {
            for (int j = 0; j < input_size; j++) {
                outputs[i] += this->weights[i][j] * input_matrix[j] * prune_mask[i][j];
            }
            outputs[i] += this->bias[i];
        }

        if(use_relu){
            relu(&outputs);
        }
        return outputs;
    }
    
    vector<double> backwards(vector<double> dLdZ) {
        // dLdA == dLdZ*relu_derivative(dLdZ)==relu(dLdZ) because of how relu works a*drelu(a) == relu(a)
        // relu(&dLdZ, true); 
        // dLdZ.print();
        // print_vector(weights);

        // calculate next layer dLdZ
        vector<double> next_dLdZ(input_size);

        for(int c=0; c < input_size; c++){
            for(int r=0; r < dLdZ.size(); r++){
                next_dLdZ[c] += weights[r][c]*dLdZ[r]*prune_mask[r][c];
            }
        }
        // calculate dLdW to update weights
        vector<vector<double>> dLdW(this->output_size, vector<double>(this->input_size));
        
        for(int r=0; r < dLdZ.size(); r++){
            for(int c=0; c < input_size; c++){
                dLdW[r][c] = dLdZ[r]*input_matrix[c];
            }
        }

        // print_vector(dLdW);
        // input_matrix.print();

        if(adam_optimizer){
            adam.update(&weights, &bias, dLdW, dLdZ);
        } else {
            // w -= learning_rate * dL/dW
            for(int i=0; i< weights.size(); i++){
                for(int j=0; j< weights[i].size(); j++){
                    weights[i][j] -= learning_rate * dLdW[i][j];
                }
            }
            // b -= learning_rate *  dL/dZ
            for(int i=0; i<bias.size(); i++){
                bias[i] -= learning_rate* 10 * dLdZ[i]; // make learning rate for bias larger because its smaller number smaller condition more likely to convergeto 0
            }

        }
        // print_vector(weights);
        // printArray(bias, 10);
        return next_dLdZ;
    }
};