#! /bin/bash
rm dataSet.csv
touch dataSet.csv
clang++ -Wall -Wextra -o createDataSet createDataSet.cpp
./createDataSet
rm createDataSet
echo "Finished creating the data set\n"