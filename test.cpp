#include <iostream>
#include <cmath>
#include <omp.h>
#include <fstream>
#include <vector>
#include <sstream>

using namespace std;

void split(const std::string& str, char delimiter = ',') {
    std::vector<std::string> result;
    std::stringstream ss(str); // Convert the string into a stringstream
    std::string token;
    
    // Extract tokens separated by the delimiter
    while (std::getline(ss, token, delimiter)) {
        cout << token << endl;
        result.push_back(token); // Add the token to the result vector
    }

}

int main(){
    split("1,2,3,4,5", ',');
}