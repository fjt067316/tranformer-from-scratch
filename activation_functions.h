#include <vector>
#include <cmath>
#include <cstdlib>
#include <random>
#include <iostream>
using namespace std;
#pragma once


void relu(vector<double> *input, bool derivative=false) {
    if(derivative){
        for (int i = 0; i < input->size(); i++){
            (*input)[i] = ((*input)[i] > 0) ? 1 : 0;
        }
    } else {
        for (int i = 0; i < input->size(); i++){
            (*input)[i] = max((*input)[i], 0.001*(*input)[i]);
        }
    }
}