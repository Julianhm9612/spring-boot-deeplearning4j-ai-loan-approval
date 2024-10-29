package com.example.spring_boot_deeplearning4j_ai_loan_approval.dto;

public class LoanApplication {
    private double creditScore;
    private double income;
    private double loanAmount;
    private int employmentStatus;

    public double getCreditScore() {
        return creditScore;
    }

    public void setCreditScore(double creditScore) {
        this.creditScore = creditScore;
    }

    public double getIncome() {
        return income;
    }

    public void setIncome(double income) {
        this.income = income;
    }

    public double getLoanAmount() {
        return loanAmount;
    }

    public void setLoanAmount(double loanAmount) {
        this.loanAmount = loanAmount;
    }

    public int getEmploymentStatus() {
        return employmentStatus;
    }

    public void setEmploymentStatus(int employmentStatus) {
        this.employmentStatus = employmentStatus;
    }

    
}