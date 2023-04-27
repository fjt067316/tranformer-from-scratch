#include <iostream>
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <unordered_map>

#pragma once
using namespace std;

void printVector(const vector<vector<double>>& v) {
    for (const auto& row : v) {
        for (const auto& element : row) {
            cout << element << " ";
        }
        cout << endl;
    }
}

void create_fr_word_maps(const char* filename, unordered_map<string, int>& word_to_index, unordered_map<int, string>& index_to_word) {
    // Open the file
    FILE* file = fopen(filename, "r");

    // Make sure the file was opened successfully
    if (file == nullptr) {
        std::cerr << "Error: could not open file " << filename << std::endl;
        return;
    }

    char word[100];
    int index = 0;

    // Read the file line by line and populate the hash maps
    while (fgets(word, 100, file) != nullptr) {
        // Remove the trailing newline character
        word[strcspn(word, "\n")] = '\0';

        // Add the word and index to the hash maps
        word_to_index[word] = index;
        index_to_word[index] = word;

        index++;
    }

    // Close the file
    fclose(file);
}

void readCSV(string fname, vector<string>& en_sentences, vector<string>& fr_sentences)
{
    ifstream file(fname);
    string line;

    while (std::getline(file, line))
    {
        // Split the line into English and French sentences using the comma separator
        size_t comma_pos = line.find(",");
        string en_sentence = line.substr(0, comma_pos);
        string fr_sentence = line.substr(comma_pos + 1);

        // Convert English and French sentences to lower case
        transform(en_sentence.begin(), en_sentence.end(), en_sentence.begin(), ::tolower);
        transform(fr_sentence.begin(), fr_sentence.end(), fr_sentence.begin(), ::tolower);

        // Add the English and French sentences to their respective vectors
        en_sentences.push_back(en_sentence);
        fr_sentences.push_back(fr_sentence);
    }
}
