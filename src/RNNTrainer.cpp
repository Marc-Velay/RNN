#include "../include/RNNTrainer.hpp"


int maxEpochs = 2;
int epoch = 0;
double lrate_weights = 3e-3;
double lrate_bias = 1e-3;
int iter_per_epo = 100;
float training_percent = 0.8;
double trainingSetAccuracy =0.0;
double validationSetAccuracy =0.0;
double generalizationSetAccuracy =0.0;
double trainingSetMSE =0.0;
double validationSetMSE =0.0;
double generalizationSetMSE =0.0;
std::vector<HiddenLayerConfig> hiddenConfig;
SoftmaxLayerConfig softmaxConfig;

void RNNTrainer::configNet() {
    for(int i = 0; i < NB_LAYERS; i++){
        hiddenConfig.push_back(HiddenLayerConfig(HIDDEN_NEURONS, HIDDEN_WEIGHT_DECAY, HIDDEN_DROPOUT_RATE));
            
    }
    softmaxConfig.NumClasses = NB_INPUTS;
    softmaxConfig.WeightDecay = SOFTMAX_WEIGHT_DECAY;
    cout << "created config net" << endl;
}

void RNNTrainer::weightMatRandomInit(Hl &ntw, int inputsize, int hiddensize){
    double epsilon = 0.12;
    //forward
    // weight between hidden layer with previous layer
    ntw.U_l = Mat::ones(hiddensize, inputsize, CV_64FC1);
    randu(ntw.U_l, Scalar(-1.0), Scalar(1.0));
    ntw.U_l = ntw.U_l * epsilon;
    ntw.U_lgrad = Mat::zeros(hiddensize, inputsize, CV_64FC1);
    ntw.U_ld2 = Mat::zeros(ntw.U_l.size(), CV_64FC1);
    ntw.lr_U = lrate_weights;
    
    // weight between current time t with previous time t-1
    ntw.W_l = Mat::ones(hiddensize, hiddensize, CV_64FC1);
    randu(ntw.W_l, Scalar(-1.0), Scalar(1.0));
    ntw.W_l = ntw.W_l * epsilon;
    ntw.W_lgrad = Mat::zeros(hiddensize, hiddensize, CV_64FC1);
    ntw.W_ld2 = Mat::zeros(ntw.W_l.size(), CV_64FC1);
    ntw.lr_W = lrate_weights;

    //backward
    /*
    ntw.U_r = Mat::ones(hiddensize, inputsize, CV_64FC1);
    randu(ntw.U_r, Scalar(-1.0), Scalar(1.0));
    ntw.U_r = ntw.U_r * epsilon;
    ntw.U_rgrad = Mat::zeros(hiddensize, inputsize, CV_64FC1);
    ntw.U_rd2 = Mat::zeros(ntw.U_r.size(), CV_64FC1);
    ntw.lr_U = lrate_weights;
    
    ntw.W_r = Mat::ones(hiddensize, hiddensize, CV_64FC1);
    randu(ntw.W_r, Scalar(-1.0), Scalar(1.0));
    ntw.W_r = ntw.W_r * epsilon;
    ntw.W_rgrad = Mat::zeros(hiddensize, hiddensize, CV_64FC1);
    ntw.W_rd2 = Mat::zeros(ntw.W_r.size(), CV_64FC1);
    ntw.lr_W = lrate_weights;
    */
}

void RNNTrainer::weightMatRandomInit(Smr &smr, int nclasses, int nfeatures){
    double epsilon = 0.12;
    //link from forward hidden layer
    smr.W_l = Mat::ones(nclasses, nfeatures, CV_64FC1);
    randu(smr.W_l, Scalar(-1.0), Scalar(1.0));
    smr.W_l = smr.W_l * epsilon;
    smr.W_lgrad = Mat::zeros(nclasses, nfeatures, CV_64FC1);
    smr.W_ld2 = Mat::zeros(smr.W_l.size(), CV_64FC1);

    //link from backward hidden layer
    /*
    smr.W_r = Mat::ones(nclasses, nfeatures, CV_64FC1);
    randu(smr.W_r, Scalar(-1.0), Scalar(1.0));
    smr.W_r = smr.W_r * epsilon;
    smr.W_rgrad = Mat::zeros(nclasses, nfeatures, CV_64FC1);
    smr.W_rd2 = Mat::zeros(smr.W_r.size(), CV_64FC1);
    */
    
    smr.cost = 0.0;
    smr.lr_W = lrate_weights;
}

void RNNTrainer::rnnInitParams(std::vector<Hl> &HiddenLayers, Smr &smr){
    
    // Init Hidden layers
    if(hiddenConfig.size() > 0){
        Hl newHiddenLayer; 
        weightMatRandomInit(newHiddenLayer, NB_INPUTS, hiddenConfig[0].NumHiddenNeurons);
        HiddenLayers.push_back(newHiddenLayer);
        for(int i = 1; i < (int)hiddenConfig.size(); i++){
            Hl newHiddenLayer2;
            weightMatRandomInit(newHiddenLayer2, hiddenConfig[i - 1].NumHiddenNeurons, hiddenConfig[i].NumHiddenNeurons);
            HiddenLayers.push_back(newHiddenLayer2);
        }
    }
    // Init Softmax layer
    if(hiddenConfig.size() == 0){
        weightMatRandomInit(smr, softmaxConfig.NumClasses, NB_TARGETS);
    }else{
        weightMatRandomInit(smr, softmaxConfig.NumClasses, hiddenConfig[(int)hiddenConfig.size() - 1].NumHiddenNeurons);
    }
}


void RNNTrainer::trainNetwork(trainingDataSet* tSet, std::vector<Hl> &HiddenLayers, Smr &smr) {
    epoch = 1;
    while(epoch <= maxEpochs) {
        cout << "trainingSetAccuracy: " << trainingSetAccuracy << endl;
        runTrainingEpoch(tSet->trainingSet, HiddenLayers, smr);

        generalizationSetAccuracy = getSetAccuracy( tSet->generalizationSet, HiddenLayers, smr);
        generalizationSetMSE = getSetMSE( tSet->generalizationSet, HiddenLayers, smr);

        cout << "Epoch :" << epoch;
        cout << " TSet Acc:" << trainingSetAccuracy << "%, MSE: " << trainingSetMSE ;
        cout << " GSet Acc:" << generalizationSetAccuracy << "%, MSE: " << generalizationSetMSE << endl << endl;

        epoch++;
    }
}


void RNNTrainer::runTrainingEpoch(vector<dataEntry*> trainingSet, std::vector<Hl> &HiddenLayers, Smr &smr) {
    double incorrectPatterns = 0;
    double mse = 0;
    feedForward(trainingSet, HiddenLayers, smr);
    //backpropagate(trainingSet[i]->target, HiddenLayers, smr);

    //bool patternCorrect = true;

    /*for(int j = 0; j < NB_TARGETS; j++) {
        if(NN->clampOutput(NN->outputNeurons[j]) != trainingSet[i]->target[j] ) {
            patternCorrect = false;
        }
        mse += pow((NN->outputNeurons[j] - trainingSet[i]->target[j]), 2);
    }
    if(!patternCorrect) incorrectPatterns++;*/

    //trainingSetAccuracy = 100 - (incorrectPatterns/trainingSet.size() * 100);
    //trainingSetMSE = mse / ( NN->nOutput * trainingSet.size() );
}

void feedForward(vector<dataEntry*> trainingSet, std::vector<Hl> &HiddenLayers, Smr &smr) {
    Mat v_smr_W_l = Mat::zeros(smr.W_l.size(), CV_64FC1);
    Mat smrW_ld2 = Mat::zeros(smr.W_l.size(), CV_64FC1);
    //Mat v_smr_W_r = Mat::zeros(smr.W_l.size(), CV_64FC1);
    //Mat smrW_rd2 = Mat::zeros(smr.W_l.size(), CV_64FC1);
    std::vector<Mat> v_hl_W_l;
    std::vector<Mat> hlW_ld2;
    std::vector<Mat> v_hl_U_l;
    std::vector<Mat> hlU_ld2;
    //std::vector<Mat> v_hl_W_r;
    //std::vector<Mat> hlW_rd2;
    //std::vector<Mat> v_hl_U_r;
    //std::vector<Mat> hlU_rd2;
    for(int i = 0; i < (int)HiddenLayers.size(); ++i){
        Mat tmpW = Mat::zeros(HiddenLayers[i].W_l.size(), CV_64FC1);
        Mat tmpU = Mat::zeros(HiddenLayers[i].U_l.size(), CV_64FC1);
        v_hl_W_l.push_back(tmpW);
        v_hl_U_l.push_back(tmpU);
        hlW_ld2.push_back(tmpW);
        hlU_ld2.push_back(tmpU);
        //v_hl_W_r.push_back(tmpW);
        //v_hl_U_r.push_back(tmpU);
        //hlW_rd2.push_back(tmpW);
        //hlU_rd2.push_back(tmpU);
    }
    double Momentum_w = 0.5;
    double Momentum_u = 0.5;
    double Momentum_d2 = 0.5;
    Mat lr_W;
    Mat lr_U;
    double mu = 1e-2;
    int k = 0;

        for(; k < (int)trainingSet.size(); k++){
            if(k > 300) {Momentum_w = 0.95; Momentum_u = 0.95; Momentum_d2 = 0.90;}
            cout<<"epoch: "<<epoch<<", iter: "<< k <<endl;    
            getNetworkCost(trainingSet[k], HiddenLayers, smr);

            // softmax update
            smrW_ld2 = Momentum_d2 * smrW_ld2 + (1.0 - Momentum_d2) * smr.W_ld2;
            lr_W = smr.lr_W / (smrW_ld2 + mu);
            v_smr_W_l = v_smr_W_l * Momentum_w + (1.0 - Momentum_w) * smr.W_lgrad.mul(lr_W);
            smr.W_l -= v_smr_W_l;

            /*
            smrW_rd2 = Momentum_d2 * smrW_rd2 + (1.0 - Momentum_d2) * smr.W_rd2;
            lr_W = smr.lr_W / (smrW_rd2 + mu);
            v_smr_W_r = v_smr_W_r * Momentum_w + (1.0 - Momentum_w) * smr.W_rgrad.mul(lr_W);
            smr.W_r -= v_smr_W_r;
            */

            // hidden layer update
            for(int i = 0; i < (int)HiddenLayers.size(); i++){
                hlW_ld2[i] = Momentum_d2 * hlW_ld2[i] + (1.0 - Momentum_d2) * HiddenLayers[i].W_ld2;
                hlU_ld2[i] = Momentum_d2 * hlU_ld2[i] + (1.0 - Momentum_d2) * HiddenLayers[i].U_ld2;
                lr_W = HiddenLayers[i].lr_W / (hlW_ld2[i] + mu);
                lr_U = HiddenLayers[i].lr_U / (hlU_ld2[i] + mu);
                v_hl_W_l[i] = v_hl_W_l[i] * Momentum_w + (1.0 - Momentum_w) * HiddenLayers[i].W_lgrad.mul(lr_W);
                v_hl_U_l[i] = v_hl_U_l[i] * Momentum_u + (1.0 - Momentum_u) * HiddenLayers[i].U_lgrad.mul(lr_U);
                HiddenLayers[i].W_l -= v_hl_W_l[i];
                HiddenLayers[i].U_l -= v_hl_U_l[i];

                /*
                hlW_rd2[i] = Momentum_d2 * hlW_rd2[i] + (1.0 - Momentum_d2) * HiddenLayers[i].W_rd2;
                hlU_rd2[i] = Momentum_d2 * hlU_rd2[i] + (1.0 - Momentum_d2) * HiddenLayers[i].U_rd2;
                lr_W = HiddenLayers[i].lr_W / (hlW_rd2[i] + mu);
                lr_U = HiddenLayers[i].lr_U / (hlU_rd2[i] + mu);
                v_hl_W_r[i] = v_hl_W_r[i] * Momentum_w + (1.0 - Momentum_w) * HiddenLayers[i].W_rgrad.mul(lr_W);
                v_hl_U_r[i] = v_hl_U_r[i] * Momentum_u + (1.0 - Momentum_u) * HiddenLayers[i].U_rgrad.mul(lr_U);
                HiddenLayers[i].W_r -= v_hl_W_r[i];
                HiddenLayers[i].U_r -= v_hl_U_r[i];
                */
            }
            //sampleX.clear();
            //std::vector<Mat>().swap(sampleX);
        }
        
        
        cout<<"Test training data: "<<endl;;
        testNetwork(trainingSet, HiddenLayers, smr);
        cout<<"Test testing data: "<<endl;;
        testNetwork(trainingSet, HiddenLayers, smr);
        
    
    v_smr_W_l.release();
    v_hl_W_l.clear();
    std::vector<Mat>().swap(v_hl_W_l);
    v_hl_U_l.clear();
    std::vector<Mat>().swap(v_hl_U_l);
    hlW_ld2.clear();
    std::vector<Mat>().swap(hlW_ld2);
    hlU_ld2.clear();
    std::vector<Mat>().swap(hlU_ld2);
    /*
    v_smr_W_r.release();
    v_hl_W_r.clear();
    std::vector<Mat>().swap(v_hl_W_r);
    v_hl_U_r.clear();
    std::vector<Mat>().swap(v_hl_U_r);
    hlW_rd2.clear();
    std::vector<Mat>().swap(hlW_rd2);
    hlU_rd2.clear();
    std::vector<Mat>().swap(hlU_rd2);
    */
}



void getNetworkCost(dataEntry* trainingPoint, std::vector<Hl> &hLayers, Smr &smr){

    int T = 1; //trainingPoint->pattern.size();
    int nSamples = 1;//(int)trainingPoint->pattern[0].size();
    //cout << "started network cost" << endl;
    //cin.ignore();
    Mat pattern(trainingPoint->pattern);
    Mat target(trainingPoint->target);
    Mat tmp, tmp2;
    // hidden layer forward
    std::vector<std::vector<Mat> > nonlin_l;
    std::vector<std::vector<Mat> > acti_l;
    std::vector<std::vector<Mat> > bernoulli_l;

    /*
    //hidden layer backward
    std::vector<std::vector<Mat> > nonlin_r;
    std::vector<std::vector<Mat> > acti_r;
    std::vector<std::vector<Mat> > bernoulli_r;
    */

    std::vector<Mat> tmp_vec;
    acti_l.push_back(tmp_vec);
    //acti_r.push_back(tmp_vec); 
    for(int i = 0; i < T; ++i){
        //acti_r[0].push_back(x[i]);
        acti_l[0].push_back(pattern);
        tmp_vec.push_back(tmp);
    }

    for(int i = 1; i <= (int)hiddenConfig.size(); ++i){
        // for each hidden layer
        acti_l.push_back(tmp_vec);
        nonlin_l.push_back(tmp_vec);
        bernoulli_l.push_back(tmp_vec);

        /*
        acti_r.push_back(tmp_vec);
        nonlin_r.push_back(tmp_vec);
        bernoulli_r.push_back(tmp_vec);
        */
        // from left to right
        for(int j = 0; j < T; ++j){
            // for each time slot
            Mat tmpacti = hLayers[i - 1].U_l * acti_l[i - 1][j];
            if(j > 0) tmpacti += hLayers[i - 1].W_l * acti_l[i][j - 1];
            
            //if(i > 1) tmpacti += hLayers[i - 1].U_l * acti_r[i - 1][j];
            
            tmpacti.copyTo(nonlin_l[i - 1][j]);
            tmpacti = ReLU(tmpacti);
            if(hiddenConfig[i - 1].DropoutRate < 1.0){
                Mat bnl = getBernoulliMatrix(tmpacti.rows, tmpacti.cols, hiddenConfig[i - 1].DropoutRate);
                tmp = tmpacti.mul(bnl);
                tmp.copyTo(acti_l[i][j]);
                bnl.copyTo(bernoulli_l[i - 1][j]);
            }else tmpacti.copyTo(acti_l[i][j]);
        }
        /*
        // from right to left
        for(int j = T - 1; j >= 0; --j){
            // for each time slot
            Mat tmpacti = hLayers[i - 1].U_r * acti_r[i - 1][j];
            if(j < T - 1) tmpacti += hLayers[i - 1].W_r * acti_r[i][j + 1];
            if(i > 1) tmpacti += hLayers[i - 1].U_r * acti_l[i - 1][j];
            tmpacti.copyTo(nonlin_r[i - 1][j]);
            tmpacti = ReLU(tmpacti);
            if(hiddenConfig[i - 1].DropoutRate < 1.0){
                Mat bnl = getBernoulliMatrix(tmpacti.rows, tmpacti.cols, hiddenConfig[i - 1].DropoutRate);
                tmp = tmpacti.mul(bnl);
                tmp.copyTo(acti_r[i][j]);
                bnl.copyTo(bernoulli_r[i - 1][j]);
            }else tmpacti.copyTo(acti_r[i][j]);
        }
        */
    }

    // softmax layer forward
    std::vector<Mat> p;
    for(int i = 0; i < T; ++i){
        Mat M = smr.W_l * acti_l[acti_l.size() - 1][i];
        //M += smr.W_r * acti_r[acti_r.size() - 1][i];
        M -= repeat(reduce(M, 0, CV_REDUCE_MAX), M.rows, 1);
        M = exp(M);
        Mat tmpp = divide(M, repeat(reduce(M, 0, CV_REDUCE_SUM), M.rows, 1));
        p.push_back(tmpp);
    }

    std::vector<Mat> groundTruth;
    for(int i = 0; i < T; ++i){
        Mat tmpgroundTruth = Mat::zeros(softmaxConfig.NumClasses, nSamples, CV_64FC1);
        for(int j = 0; j < nSamples; j++){
            tmpgroundTruth.ATD(target.ATD(i, j), j) = 1.0;
        }
        groundTruth.push_back(tmpgroundTruth);
    }

    double J1 = 0.0;
    for(int i = 0; i < T; i++){
        J1 +=  - sum1(groundTruth[i].mul(log(p[i])));
    }
    J1 /= nSamples;
    double J2 = (sum1(pow(smr.W_l, 2.0)) /*+ sum1(pow(smr.W_r, 2.0))*/) * softmaxConfig.WeightDecay /*/ 2*/;
    double J3 = 0.0; 
    double J4 = 0.0;
    for(int hl = 0; hl < (int)hLayers.size(); hl++){
        J3 += (sum1(pow(hLayers[hl].W_l, 2.0)) /*+ sum1(pow(hLayers[hl].W_r, 2.0))*/) * hiddenConfig[hl].WeightDecay /*/ 2*/;
    }
    for(int hl = 0; hl < (int)hLayers.size(); hl++){
        J4 += (sum1(pow(hLayers[hl].U_l, 2.0)) /*+ sum1(pow(hLayers[hl].U_r, 2.0))*/) * hiddenConfig[hl].WeightDecay /*/ 2*/;
    }
    smr.cost = J1 + J2 + J3 + J4;

    //cout << "calculated cost" << endl;
    //cin.ignore();
    /*if(!is_gradient_checking) 
        cout<<", J1 = "<<J1<<", J2 = "<<J2<<", J3 = "<<J3<<", J4 = "<<J4<<", Cost = "<<smr.cost<<endl;*/

    // softmax layer backward
    /*
    tmp = - (groundTruth[0] - p[0]) * acti_l[acti_l.size() - 1][0].t();
    for(int i = 1; i < T; ++i){
        tmp += - (groundTruth[i] - p[i]) * acti_l[acti_l.size() - 1][i].t();
    }
    smr.W_lgrad =  tmp / nSamples + softmaxConfig.WeightDecay * smr.W_l;
    tmp = pow((groundTruth[0] - p[0]), 2.0) * pow(acti_l[acti_l.size() - 1][0].t(), 2.0);
    for(int i = 1; i < T; ++i){
        tmp += pow((groundTruth[i] - p[i]), 2.0) * pow(acti_l[acti_l.size() - 1][i].t(), 2.0);
    }
    smr.W_ld2 = tmp / nSamples + softmaxConfig.WeightDecay;

    tmp = - (groundTruth[0] - p[0]) * acti_r[acti_r.size() - 1][0].t();
    for(int i = 1; i < T; ++i){
        tmp += - (groundTruth[i] - p[i]) * acti_r[acti_r.size() - 1][i].t();
    }
    smr.W_rgrad =  tmp / nSamples + softmaxConfig.WeightDecay * smr.W_r;
    tmp = pow((groundTruth[0] - p[0]), 2.0) * pow(acti_r[acti_r.size() - 1][0].t(), 2.0);
    for(int i = 1; i < T; ++i){
        tmp += pow((groundTruth[i] - p[i]), 2.0) * pow(acti_r[acti_r.size() - 1][i].t(), 2.0);
    }
    smr.W_rd2 = tmp / nSamples + softmaxConfig.WeightDecay;
    */


    // hidden layer backward
    
    std::vector<std::vector<Mat> > delta_l(acti_l.size());
    std::vector<std::vector<Mat> > delta_ld2(acti_l.size());
    /*
    std::vector<std::vector<Mat> > delta_r(acti_r.size());
    std::vector<std::vector<Mat> > delta_rd2(acti_r.size());
    */

    for(int i = 0; i < delta_l.size(); i++){
        delta_l[i].clear();
        delta_ld2[i].clear();
        /*
        delta_r[i].clear();
        delta_rd2[i].clear();
        */
        Mat tmpmat;
        for(int j = 0; j < T; j++){
            delta_l[i].push_back(tmpmat);
            delta_ld2[i].push_back(tmpmat);
            /*
            delta_r[i].push_back(tmpmat);
            delta_rd2[i].push_back(tmpmat);
            */
        }
    }
    
    // Last hidden layer
    // Do BPTT backward pass for the forward hidden layer
    for(int i = T - 1; i >= 0; i--){
        tmp = -smr.W_l.t() * (groundTruth[i] - p[i]);
        tmp2 = pow(smr.W_l.t(), 2.0) * pow((groundTruth[i] - p[i]), 2.0);
        if(i < T - 1){
            tmp += hLayers[(int)hLayers.size() - 1].W_l.t() * delta_l[delta_l.size() - 1][i + 1];
            tmp2 += pow(hLayers[(int)hLayers.size() - 1].W_l.t(), 2.0) * delta_ld2[delta_ld2.size() - 1][i + 1];
        }
        tmp.copyTo(delta_l[delta_l.size() - 1][i]);
        tmp2.copyTo(delta_ld2[delta_ld2.size() - 1][i]);
        delta_l[delta_l.size() - 1][i] = delta_l[delta_l.size() - 1][i].mul(dReLU(nonlin_l[nonlin_l.size() - 1][i]));
        delta_ld2[delta_ld2.size() - 1][i] = delta_ld2[delta_ld2.size() - 1][i].mul(pow(dReLU(nonlin_l[nonlin_l.size() - 1][i]), 2.0));
        if(hiddenConfig[(int)hiddenConfig.size() - 1].DropoutRate < 1.0){
            delta_l[delta_l.size() - 1][i] = delta_l[delta_l.size() -1][i].mul(bernoulli_l[bernoulli_l.size() - 1][i]);
            delta_ld2[delta_ld2.size() - 1][i] = delta_ld2[delta_ld2.size() -1][i].mul(pow(bernoulli_l[bernoulli_l.size() - 1][i], 2.0));
        } 
    }

    // Do BPTT backward pass for the backward hidden layer
    /*
    for(int i = 0; i < T; i++){
        tmp = -smr.W_r.t() * (groundTruth[i] - p[i]);
        tmp2 = pow(smr.W_r.t(), 2.0) * pow((groundTruth[i] - p[i]), 2.0);
        if(i > 0){
            tmp += hLayers[hLayers.size() - 1].W_r.t() * delta_r[delta_r.size() - 1][i - 1];
            tmp2 += pow(hLayers[hLayers.size() - 1].W_r.t(), 2.0) * delta_rd2[delta_rd2.size() - 1][i - 1];
        }
        tmp.copyTo(delta_r[delta_r.size() - 1][i]);
        tmp2.copyTo(delta_rd2[delta_rd2.size() - 1][i]);
        delta_r[delta_r.size() - 1][i] = delta_r[delta_r.size() - 1][i].mul(dReLU(nonlin_r[nonlin_r.size() - 1][i]));
        delta_rd2[delta_rd2.size() - 1][i] = delta_rd2[delta_rd2.size() - 1][i].mul(pow(dReLU(nonlin_r[nonlin_r.size() - 1][i]), 2.0));
        if(hiddenConfig[hiddenConfig.size() - 1].DropoutRate < 1.0){
            delta_r[delta_r.size() - 1][i] = delta_r[delta_r.size() -1][i].mul(bernoulli_r[bernoulli_r.size() - 1][i]);
            delta_rd2[delta_rd2.size() - 1][i] = delta_rd2[delta_rd2.size() -1][i].mul(pow(bernoulli_r[bernoulli_r.size() - 1][i], 2.0));
        } 
    }
    */
    // hidden layers
    for(int i = delta_l.size() - 2; i > 0; --i){
        // Do BPTT backward pass for the forward hidden layer
        for(int j = T - 1; j >= 0; --j){
            tmp = hLayers[i].U_l.t() * delta_l[i + 1][j];
            tmp2 = pow(hLayers[i].U_l.t(), 2.0) * delta_ld2[i + 1][j];
            if(j < T - 1){
                tmp += hLayers[i - 1].W_l.t() * delta_l[i][j + 1];
                tmp2 += pow(hLayers[i - 1].W_l.t(), 2.0) * delta_ld2[i][j + 1];
            }
            /*
            tmp += hLayers[i].U_r.t() * delta_r[i + 1][j];
            tmp2 += pow(hLayers[i].U_r.t(), 2.0) * delta_rd2[i + 1][j];
            */
            tmp.copyTo(delta_l[i][j]);
            tmp2.copyTo(delta_ld2[i][j]);
            delta_l[i][j] = delta_l[i][j].mul(dReLU(nonlin_l[i - 1][j]));
            delta_ld2[i][j] = delta_ld2[i][j].mul(pow(dReLU(nonlin_l[i - 1][j]), 2.0));
            if(hiddenConfig[i - 1].DropoutRate < 1.0){
                delta_l[i][j] = delta_l[i][j].mul(bernoulli_l[i - 1][j]);
                delta_ld2[i][j] = delta_ld2[i][j].mul(pow(bernoulli_l[i - 1][j], 2.0));
            }
        }

        // Do BPTT backward pass for the backward hidden layer
        /*
        for(int j = 0; j < T; ++j){
            tmp = hLayers[i].U_r.t() * delta_r[i + 1][j];
            tmp2 = pow(hLayers[i].U_r.t(), 2.0) * delta_rd2[i + 1][j];
            if(j > 0){
                tmp += hLayers[i - 1].W_r.t() * delta_r[i][j - 1];
                tmp2 += pow(hLayers[i - 1].W_r.t(), 2.0) * delta_rd2[i][j - 1];
            }
            tmp += hLayers[i].U_l.t() * delta_l[i + 1][j];
            tmp2 += pow(hLayers[i].U_l.t(), 2.0) * delta_ld2[i + 1][j];
            tmp.copyTo(delta_r[i][j]);
            tmp2.copyTo(delta_rd2[i][j]);
            delta_r[i][j] = delta_r[i][j].mul(dReLU(nonlin_r[i - 1][j]));
            delta_rd2[i][j] = delta_rd2[i][j].mul(pow(dReLU(nonlin_r[i - 1][j]), 2.0));
            if(hiddenConfig[i - 1].DropoutRate < 1.0){
                delta_r[i][j] = delta_r[i][j].mul(bernoulli_r[i - 1][j]);
                delta_rd2[i][j] = delta_rd2[i][j].mul(pow(bernoulli_r[i - 1][j], 2.0));
            }
        }
        */
    }
    //cout << "finished calc weight diffs" << endl;
    //cin.ignore();

    for(int i = (int)hiddenConfig.size() - 1; i >= 0; i--){
        // forward part.
        if(i == 0){
            tmp = delta_l[i + 1][0] * acti_l[i][0].t();
            tmp2 = delta_ld2[i + 1][0] * pow(acti_l[i][0].t(), 2.0);
            for(int j = 1; j < T; ++j){
                tmp += delta_l[i + 1][j] * acti_l[i][j].t();
                tmp2 += delta_ld2[i + 1][j] * pow(acti_l[i][j].t(), 2.0);
            }

    //cout << "last layer" << endl;
    //cin.ignore();
        }else{
            tmp = delta_l[i + 1][0] * (acti_l[i][0].t() /*+ acti_r[i][0].t()*/);
            tmp2 = delta_ld2[i + 1][0] * (pow(acti_l[i][0].t(), 2.0) /*+ pow(acti_r[i][0].t(), 2.0)*/);
            for(int j = 1; j < T; ++j){
                tmp += delta_l[i + 1][j] * (acti_l[i][j].t() /*+ acti_r[i][j].t()*/);
                tmp2 += delta_ld2[i + 1][j] * (pow(acti_l[i][j].t(), 2.0) /*+ pow(acti_r[i][j].t(), 2.0)*/);
            }
        }

    //cout << "layer: " << i << endl;
    //cin.ignore();
        hLayers[i].U_lgrad = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].U_l;
        hLayers[i].U_ld2 = tmp2 / nSamples + hiddenConfig[i].WeightDecay;

    //cout << "updated lgrad n ld" << endl;
    //cin.ignore();
        tmp = delta_l[i + 1][T - 1] * acti_l[i + 1][T - 1].t();
        tmp2 = delta_ld2[i + 1][T - 1] * pow(acti_l[i + 1][T - 1].t(), 2.0);
        for(int j = T - 2; j > 0; j--){
            tmp += delta_l[i + 1][j] * acti_l[i + 1][j - 1].t();
            tmp2 += delta_ld2[i + 1][j] * pow(acti_l[i + 1][j - 1].t(), 2.0);
        }

    //cout << "calculated deltas" << endl;
    //cin.ignore();
        hLayers[i].W_lgrad = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].W_l;
        hLayers[i].W_ld2 = tmp2 / nSamples + hiddenConfig[i].WeightDecay;

    
    //cout << "calculated weight vars" << endl;
    //cin.ignore();
        // backward part.
        /*
        if(i == 0){
            tmp = delta_r[i + 1][0] * acti_r[i][0].t();
            tmp2 = delta_rd2[i + 1][0] * pow(acti_r[i][0].t(), 2.0);
            for(int j = 1; j < T; ++j){
                tmp += delta_r[i + 1][j] * acti_r[i][j].t();
                tmp2 += delta_rd2[i + 1][j] * pow(acti_r[i][j].t(), 2.0);
            }
        }else{
            tmp = delta_r[i + 1][0] * (acti_l[i][0].t() + acti_r[i][0].t());
            tmp2 = delta_rd2[i + 1][0] * (pow(acti_l[i][0].t(), 2.0) + pow(acti_r[i][0].t(), 2.0));
            for(int j = 1; j < T; ++j){
                tmp += delta_r[i + 1][j] * (acti_l[i][j].t() + acti_r[i][j].t());
                tmp2 += delta_rd2[i + 1][j] * (pow(acti_l[i][j].t(), 2.0) + pow(acti_r[i][j].t(), 2.0));
            }
        }
        hLayers[i].U_rgrad = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].U_r;
        hLayers[i].U_rd2 = tmp2 / nSamples + hiddenConfig[i].WeightDecay;

        tmp = delta_r[i + 1][0] * acti_r[i + 1][1].t();
        tmp2 = delta_rd2[i + 1][0] * pow(acti_r[i + 1][1].t(), 2.0);
        for(int j = 1; j < T - 1; j++){
            tmp += delta_r[i + 1][j] * acti_r[i + 1][j + 1].t();
            tmp2 += delta_rd2[i + 1][j] * pow(acti_r[i + 1][j + 1].t(), 2.0);
        }
        hLayers[i].W_rgrad = tmp / nSamples + hiddenConfig[i].WeightDecay * hLayers[i].W_r;
        hLayers[i].W_rd2 = tmp2 / nSamples + hiddenConfig[i].WeightDecay;
        */
    }
    // destructor
    acti_l.clear();
    std::vector<std::vector<Mat> >().swap(acti_l);
    nonlin_l.clear();
    std::vector<std::vector<Mat> >().swap(nonlin_l);
    delta_l.clear();
    std::vector<std::vector<Mat> >().swap(delta_l);
    delta_ld2.clear();
    std::vector<std::vector<Mat> >().swap(delta_ld2);
    bernoulli_l.clear();
    std::vector<std::vector<Mat> >().swap(bernoulli_l);
    /*
    acti_r.clear();
    std::vector<std::vector<Mat> >().swap(acti_r);
    nonlin_r.clear();
    std::vector<std::vector<Mat> >().swap(nonlin_r);
    delta_r.clear();
    std::vector<std::vector<Mat> >().swap(delta_r);
    delta_rd2.clear();
    std::vector<std::vector<Mat> >().swap(delta_rd2);
    bernoulli_r.clear();
    std::vector<std::vector<Mat> >().swap(bernoulli_r);
    */

    p.clear();
    std::vector<Mat>().swap(p);
    groundTruth.clear();
    std::vector<Mat>().swap(groundTruth);
}

Mat resultPredict(vector<dataEntry*> trainingSet, int start, int end, std::vector<Hl> &hLayers, Smr &smr){

    int T = end-start; //trainingPoint->pattern.size();
    //cout << "started network cost" << endl;
    //cin.ignore();
    std::vector<Mat > pattern;
    for (int i = start; i < end; i++){
        Mat tmp(trainingSet[i]->pattern);
        pattern.push_back(tmp);
    }
    
    //int mid = (int)(T /2.0);
    Mat tmp;
    // hidden layer forward
    std::vector<std::vector<Mat> > acti_l;
    //std::vector<std::vector<Mat> > acti_r;
    std::vector<Mat> tmp_vec;
    acti_l.push_back(tmp_vec);
    //acti_r.push_back(tmp_vec); 
    for(int i = 0; i < T; ++i){
        //acti_r[0].push_back(x[i]);
        acti_l[0].push_back(pattern[i]);
        tmp_vec.push_back(tmp);
    }
    for(int i = 1; i <= hiddenConfig.size(); ++i){
        // for each hidden layer
        acti_l.push_back(tmp_vec);
        //acti_r.push_back(tmp_vec);
        // from left to right
        for(int j = 0; j < T; ++j){
            // for each time slot
            Mat tmpacti = hLayers[i - 1].U_l * acti_l[i - 1][j];
            if(j > 0) tmpacti += hLayers[i - 1].W_l * acti_l[i][j - 1];
            //if(i > 1) tmpacti += hLayers[i - 1].U_l * acti_r[i - 1][j];
            tmpacti = ReLU(tmpacti);
            if(hiddenConfig[i - 1].DropoutRate < 1.0) tmpacti = tmpacti.mul(hiddenConfig[i - 1].DropoutRate);
            tmpacti.copyTo(acti_l[i][j]);
        }
        // from right to left
        /*
        for(int j = T - 1; j >= 0; --j){
            // for each time slot
            Mat tmpacti = hLayers[i - 1].U_r * acti_r[i - 1][j];
            if(j < T - 1) tmpacti += hLayers[i - 1].W_r * acti_r[i][j + 1];
            if(i > 1) tmpacti += hLayers[i - 1].U_r * acti_l[i - 1][j];
            tmpacti = ReLU(tmpacti);
            if(hiddenConfig[i - 1].DropoutRate < 1.0) tmpacti = tmpacti.mul(hiddenConfig[i - 1].DropoutRate);
            tmpacti.copyTo(acti_r[i][j]);
        }
        */
    }
    tmp_vec.clear();
    std::vector<Mat>().swap(tmp_vec);
    // softmax layer forward
    Mat M = smr.W_l * acti_l[acti_l.size() - 1][0];
    //M += smr.W_r * acti_r[acti_r.size() - 1][mid];
    Mat result = Mat::zeros(1, M.cols, CV_64FC1);

    double minValue, maxValue;
    Point minLoc, maxLoc;
    for(int i = 0; i < M.cols; i++){
        minMaxLoc(M(Rect(i, 0, 1, M.rows)), &minValue, &maxValue, &minLoc, &maxLoc);
        result.ATD(0, i) = (int)maxLoc.y;
    }
    acti_l.clear();
    std::vector<std::vector<Mat> >().swap(acti_l);
    /*
    acti_r.clear();
    std::vector<std::vector<Mat> >().swap(acti_r);
    */
    return result;
}

void testNetwork(vector<dataEntry*> trainingSet, std::vector<Hl> &HiddenLayers, Smr &smr) {

    // Test use test set
    // Because it may leads to lack of memory if testing the whole dataset at 
    // one time, so separate the dataset into small pieces of batches (say, batch size = 20).
    // 
    int batchSize = 50;
    Mat result = Mat::zeros(1, (int)trainingSet.size(), CV_64FC1);

    std::vector<dataEntry*> tmpBatch;
    int batch_amount = (int)trainingSet.size() / batchSize;
    for(int i = 0; i < batch_amount; i++){
        for(int j = 0; j < batchSize; j++){
            tmpBatch.push_back(trainingSet[i * batchSize + j]);
        }
        /*
        std::vector<Mat> sampleX;
        getDataMat(tmpBatch, sampleX, re_wordmap);
        */
        Mat resultBatch = resultPredict(trainingSet, i * batchSize, i * batchSize + batchSize, HiddenLayers, smr);
        Rect roi = Rect(i * batchSize, 0, batchSize, 1);
        resultBatch.copyTo(result(roi));
        tmpBatch.clear();
        sampleX.clear();
    }
    if((int)trainingSet.size() % batchSize){
        for(int j = 0; j < (int)trainingSet.size() % batchSize; j++){
            tmpBatch.push_back(trainingSet[batch_amount * batchSize + j]);
        }
        std::vector<Mat> sampleX;
        getDataMat(tmpBatch, sampleX, re_wordmap);
        Mat resultBatch = resultPredict(sampleX, HiddenLayers, smr);
        Rect roi = Rect(batch_amount * batchSize, 0, (int)trainingSet.size() % batchSize, 1);
        resultBatch.copyTo(result(roi));
        ++ batch_amount;
        tmpBatch.clear();
        //sampleX.clear();
    }
    Mat sampleY = Mat::zeros(1, y.size(), CV_64FC1);
    getLabelMat(y, sampleY);

    Mat err;
    sampleY.copyTo(err);
    err -= result;
    int correct = err.cols;
    for(int i=0; i<err.cols; i++){
        if(err.ATD(0, i) != 0) --correct;
    }
    cout<<"######################################"<<endl;
    cout<<"## test result. "<<correct<<" correct of "<<err.cols<<" total."<<endl;
    cout<<"## Accuracy is "<<(double)correct / (double)(err.cols)<<endl;
    cout<<"######################################"<<endl;
    result.release();
    err.release();
}


RNNTrainer::RNNTrainer() {
    cout << "creating net" << endl;
    configNet();   
}

RNNTrainer::~RNNTrainer() {

}

int main(void){
    printf("Starting training !\n\n");
    srand( (unsigned int) time(0) );

    dataReader d;
    d.loadDataFile("dataSet.csv");
    d.setCreationApproach();
    cout << "Initialised dataset and approach" << endl;
    //create neural network
    RNNTrainer rnn;

    rnn.rnnInitParams(rnn.HiddenLayers, rnn.smr);

    //train neural network on data sets
	for (int i=0; i < d.getNumTrainingSets(); i++ )
	{
        cout << "training network on datasets" << endl;
		rnn.trainNetwork( d.getTrainingDataSet(), rnn.HiddenLayers, rnn.smr);
	}

    //save the weights
    //char * file = "weights.csv";
	//mlp.saveWeights(file);


    cout << endl << endl << "Finished training, weights saved to: 'weights.csv'" << endl;
    return 0;           
}