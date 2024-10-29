package com.example.spring_boot_deeplearning4j_ai_loan_approval.model;

import java.io.File;

import org.datavec.api.records.reader.impl.csv.CSVRecordReader;
import org.datavec.api.split.FileSplit;
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.linalg.activations.Activation;
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
        int batchSize = 5;
        DataSetIterator iterator = new RecordReaderDataSetIterator(recordReader, batchSize, labelIndex, numClasses);

        // Normalize the data
        DataNormalization normalizer = new NormalizerStandardize();
        normalizer.fit(iterator);
        iterator.setPreProcessor(normalizer);

        // Define the network configuration
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
            // .iterations(1000)
            .seed(12345)
            .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
            .updater(new
                Sgd.Builder() // Configure training options
                    .learningRate(0.01)
                    .build())
            // .activation(Activation.RELU)
            // .weightInit(WeightInit.XAVIER)
            // .learningRate(0.01)
            .l2(0.0001)
            .list()
            // .layer(0, new DenseLayer.Builder().nIn(4).nOut(3).build())
            .layer(0, new DenseLayer.Builder().nIn(4).nOut(3).activation(Activation.RELU).weightInit(WeightInit.XAVIER).build())
            .layer(1, new OutputLayer.Builder(LossFunction.NEGATIVELOGLIKELIHOOD)
                    .activation(Activation.SOFTMAX)
                    .nIn(3).nOut(2).build())
            .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        model.setListeners(new ScoreIterationListener(100));

        // Train the model
        for (int i = 0; i < 1000; i++) {
            iterator.reset();
            model.fit(iterator);
        }

        // Save the model
        model.save(new File("loan_approval_model.zip"), true);
    }
}
