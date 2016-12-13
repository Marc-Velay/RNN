#include <iostream>
#include <fstream>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <vector>
#include <string>

#define NB_INPUTS 1
#define NB_TARGETS 2

using namespace std;


class dataEntry         //stores a data item
{
public:	
	
	vector<double> pattern;	//input patterns
	vector<double> target;		//target result

	dataEntry(vector<double> p, vector<double> t): pattern(p), target(t) {}
		
	~dataEntry()
	{				
		pattern.clear();
		target.clear();
	}

};


class trainingDataSet       //Training Sets Storage - stores shortcuts to data items
{
public:

	std::vector<dataEntry*> trainingSet;
	std::vector<dataEntry*> generalizationSet;
	std::vector<dataEntry*> validationSet;

	trainingDataSet(){}
	
	void clear()
	{
		trainingSet.clear();
		generalizationSet.clear();
		validationSet.clear();
	}
};

//dataset retrieval approach enum
enum { NONE, STATIC};

//data reader class
class dataReader
{
	
public:
	std::string classList;
	
//private members
//----------------------------------------------------------------------------------------------------------------
private:

	//data storage
	std::vector<dataEntry*> data;
	int nInputs;
	int nTargets;
	//current data set
	trainingDataSet tSet;

	//data set creation approach and total number of dataSets
	int creationApproach;
	int numTrainingSets;
	int trainingDataEndIndex;

	
//public methods
//----------------------------------------------------------------------------------------------------------------
public:

	dataReader(): nInputs(NB_INPUTS), nTargets(NB_TARGETS), creationApproach(NONE), numTrainingSets(-1) {}
	~dataReader();
	
	bool loadDataFile( const char* filename);
	void setCreationApproach();
	int getNumTrainingSets();
	
	trainingDataSet* getTrainingDataSet();
	std::vector<dataEntry*>& getAllDataEntries();

//private methods
//----------------------------------------------------------------------------------------------------------------
private:
	
	void createStaticDataSet();
	void processLine( std::string &line );
	vector<double> toClass(char t);
  	size_t strcpy_s(char *d, size_t n, char const *s);	
};