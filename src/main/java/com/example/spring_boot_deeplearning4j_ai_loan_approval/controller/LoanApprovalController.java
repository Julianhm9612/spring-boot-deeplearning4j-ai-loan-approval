package com.example.spring_boot_deeplearning4j_ai_loan_approval.controller;

import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.springframework.web.bind.annotation.*;

import com.example.spring_boot_deeplearning4j_ai_loan_approval.dto.LoanApplication;

import java.io.File;
import java.io.IOException;

@RestController
@RequestMapping("/loan")
public class LoanApprovalController {

    private final MultiLayerNetwork model;

    public LoanApprovalController() throws IOException {
        // Load the trained model
        model = MultiLayerNetwork.load(new File("loan_approval_model.zip"), true);
    }

    @PostMapping("/approve")
    public String approveLoan(@RequestBody LoanApplication loanApplication) {
        // Prepare input data
        INDArray input = Nd4j.create(new double[]{
                loanApplication.getCreditScore(),
                loanApplication.getIncome(),
                loanApplication.getLoanAmount(),
                loanApplication.getEmploymentStatus()
        }, 1, 4);

        // Make prediction
        INDArray output = model.output(input);
        int prediction = Nd4j.argMax(output, 1).getInt(0);

        return prediction == 1 ? "Approved" : "Rejected";
    }
}
