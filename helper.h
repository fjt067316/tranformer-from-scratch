#include <iostream>
#include <vector>
#include <iostream>
#include <cstdio>
#include <cstring>
#include <unordered_map>


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
