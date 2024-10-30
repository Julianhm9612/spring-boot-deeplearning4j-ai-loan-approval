package com.example.spring_boot_deeplearning4j_ai_loan_approval.model;

import java.io.File;

import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.SplitTestAndTrain;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.NormalizerStandardize;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions.LossFunction;

import jakarta.annotation.PostConstruct;

public class LoanApprovalModel {

    @PostConstruct
    public static void main(String[] args) throws Exception {
        // Load dataset
        int numLinesToSkip = 0;
        char delimiter = ',';
        CSVRecordReader recordReader = new CSVRecordReader(numLinesToSkip, delimiter);
        recordReader.initialize(new FileSplit(new ClassPathResource("loan_data.csv").getFile()));

        int labelIndex = 4; // Index of the label (approve/reject)
        int numClasses = 2; // Approve or Reject
        int batchSize = 210;
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses);

        DataSet allData = iterator.next();
        allData.shuffle(42);

        // Normalize the data
        DataNormalization normalizer = new NormalizerStandardize();
        // normalizer.fit(iterator);
        // iterator.setPreProcessor(normalizer);
        normalizer.fit(allData);
        normalizer.transform(allData);

        SplitTestAndTrain testAndTrain = allData.splitTestAndTrain(0.65);
        DataSet trainingData = testAndTrain.getTrain();
        DataSet testData = testAndTrain.getTest();

        // Define the network configuration
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            .seed(12345)
            // .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new
                Sgd.Builder() // Configure training options
                    .learningRate(0.01)
                    .build())
            .list()
            .layer(0, new DenseLayer.Builder().nIn(4).nOut(3).activation(Activation.RELU).weightInit(WeightInit.XAVIER).l2(0.001).build())
            .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                    .activation(Activation.SOFTMAX)
                    .nIn(3).nOut(2).build())
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));

        // Train the model
        for (int i = 0; i < 1000; i++) {
            model.fit(trainingData);
        }

        // Train the model using built-in functionality
        // model.fit(trainingData);

        INDArray features = testData.getFeatures();
        INDArray output = model.output(features);
        Evaluation eval = new Evaluation(2);
        eval.eval(testData.getLabels(), output);

        System.out.println(eval.stats());

        // Save the model
        model.save(new File("loan_approval_model.zip"), true);
    }
}
