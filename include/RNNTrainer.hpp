#include <iostream>
#include <vector>
#include <unordered_map>
#include <string>
#include "opencv2/core.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include <fstream>
#include "dataReader.hpp"
#include "cellStructure.hpp"
#include "matrix_maths.hpp"

using namespace std;
using namespace cv;

#define NB_LAYERS 3
#define HIDDEN_NEURONS 512
#define HIDDEN_WEIGHT_DECAY 1e-6
#define HIDDEN_DROPOUT_RATE 1.0

#define SOFTMAX_WEIGHT_DECAY 1e-6

#define NB_INPUTS 1
#define NB_TARGETS 2

extern float training_percent;
extern int maxEpochs;

extern double lrate_weights;
extern double lrate_bias;
extern int iter_per_epo;

extern std::vector<HiddenLayerConfig> hiddenConfig;
extern SoftmaxLayerConfig softmaxConfig;

double getSetAccuracy( std::vector<dataEntry*>& set, std::vector<Hl> &HiddenLayers, Smr &smr);
double getSetMSE( std::vector<dataEntry*>& set, std::vector<Hl> &HiddenLayers, Smr &smr);
void feedForward(vector<dataEntry*> trainingSet, std::vector<Hl> &HiddenLayers, Smr &smr);
void getNetworkCost(dataEntry* trainingPoint, std::vector<Hl> &hLayers, Smr &smr);
void testNetwork(vector<dataEntry*> trainingSet, std::vector<Hl> &hLayers, Smr &smr);
Mat resultPredict(vector<dataEntry*> trainingSet, int start, int end, std::vector<Hl> &hLayers, Smr &smr);


class RNNTrainer {

    public:
        RNNTrainer();
        ~RNNTrainer();
        void configNet();
        void rnnInitParams(std::vector<Hl> &HiddenLayers, Smr &smr);
        void trainNetwork(trainingDataSet* tSet, std::vector<Hl> &HiddenLayers, Smr &smr);
        std::vector<Hl> HiddenLayers;
        Smr smr;
        
    private:
        void weightMatRandomInit(Smr &smr, int nclasses, int nfeatures);
        void weightMatRandomInit(Hl &ntw, int inputsize, int hiddensize);

        void runTrainingEpoch(vector<dataEntry*> trainingSet, std::vector<Hl> &HiddenLayers, Smr &smr);


        double trainingSetAccuracy;
        double validationSetAccuracy;
        double generalizationSetAccuracy;
        double trainingSetMSE;
        double validationSetMSE;
        double generalizationSetMSE;
};