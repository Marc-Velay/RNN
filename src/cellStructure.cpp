#include "../include/RNNTrainer.hpp"

double getSetAccuracy( std::vector<dataEntry*>& set, std::vector<Hl> &HiddenLayers, Smr &smr) {     //Get set accuracy
        /*double incorrectResults = 0;

        for( int i = 0; i < (int)set.size(); i++) {
                feedForward(set[i]->pattern);

                bool correctResultFlag = true;

                for(int j = 0; j < NB_TARGETS; j++) {
                        if( clampOutput(outputNeurons[j]) != set[i]->target[j]) {
                                correctResultFlag = false;
                        }
                }

                if(!correctResultFlag) {
                        incorrectResults++;
                }
        }

        return 100 - (incorrectResults/set.size() *100);*/
        return 0.0;
}



double getSetMSE( std::vector<dataEntry*>& set, std::vector<Hl> &HiddenLayers, Smr &smr) {
        /*double mse = 0;

        for (int i = 0; i < (int) set.size(); i++) {
                feedForward( set[i]->pattern );

                for( int j = 0; j < NB_TARGETS; j++) {
                        mse += pow((outputNeurons[j] - set[i]->target[j]), 2);
                }
        }

        return mse/(NB_TARGETS * set.size());*/
        return 0.0;
}