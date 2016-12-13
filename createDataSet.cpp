#include <iostream>
#include <fstream>
#include <math.h>
#include <limits>
#include <stdlib.h>

using namespace std;

int main(void) {
    double posx, posy, spam;
    double step =0;

    fstream outputFile;
    outputFile.open("dataSet.csv", ios::out);

    if( outputFile.is_open() ) {
        while(step < 8*2*3.14) {
            posx = step;
            posy = sin(step);
            spam = posy + 0.5;
            outputFile <<  posx << "," <<  posy << "," << spam << endl;
            cout <<  posx << "," <<  posy << "," << spam << endl;
            step += 0.01;
        }
        outputFile.close();
    } else {
        cout << endl << "Error - Could not open file: 'dataSet.csv' " << endl;
    }


    return 0;
}