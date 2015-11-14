package com.hyx.ml.bpnn;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Created by yixuanhe on 11/13/15.
 */
/*
 * the java implement of BPNN
 * this implement only have 3 layers: input, hidden, output
 * might have more than 3 layers in the future
 */
public class BPNN {
    // the hidden and output layer
    Layer hiddenLayer;
    Layer outputLayer;

    // the number of output
    int outputNumber;
    // the number of cells in hidden layer
    int hiddenNumber;
    // the number of input feature
    int inputNumber;
    // the active function
    ActiveFunction active;

    // the hidden layer output
    double[] hiddenOutput;
    // the input layer output
    double[] inputOutput;
    // the input value
    double[] input;

    /*
     * input_number : the number of input feature
     * hidden_number : the number of cells in hidden layers
     * output_number : the number of output
     * active : the active function
     * rate : the learning rate
     */
    public BPNN(int inputNumber, int hiddenNumber, int outputNumber, ActiveFunction active, double rate, double threshold){
        this.hiddenLayer = new Layer(hiddenNumber, inputNumber, active, rate, threshold);
        this.outputLayer = new Layer(outputNumber, hiddenNumber, active, rate, threshold);
        this.outputNumber = outputNumber;
        this.hiddenNumber = hiddenNumber;
        this.inputNumber = inputNumber;
        this.active = active;
    }

    public boolean[] predict(double[] value){
        double[] hiddenOutput = hiddenLayer.calOutput(value);
        outputLayer.calOutput(hiddenOutput);

        return outputLayer.getActive();
    }

    public double[] getOutput(double[] value){
        inputOutput = hiddenLayer.calOutput(value).clone();
        hiddenOutput = outputLayer.calOutput(inputOutput).clone();
        input = value;

        return hiddenOutput;
    }

    public double update(double[] value, double[] tag, boolean flag){
        getOutput(value);

        double[] err = new double[1];
        err[0] = 0;

        double[] val = new double[hiddenNumber];
        double[][] O2HGradient = getOutputToHiddenGradient(tag, err, val);
        double[][] H2IGradient = getHiddenToInputGradient(tag, err, val);

        outputLayer.update(O2HGradient);
        hiddenLayer.update(H2IGradient);

        if(flag) {
            double start = loss(tag, hiddenOutput);
            System.out.println("loss : " + start);
        }

        return err[0];
    }

    public double[][] getOutputToHiddenGradient(double[] tag, double[] err, double[] val){
        double[][] gradient = new double[outputNumber][hiddenNumber];

        for (int i = 0; i < hiddenNumber; i++){
            val[i] = 0;
        }

        for(int i = 0; i < outputNumber; i++){
            for (int j = 0; j < hiddenNumber; j++) {
                double delta = active.derivative(outputLayer.getWeight(i), inputOutput) * (tag[i] - hiddenOutput[i]);
                gradient[i][j] = delta * inputOutput[j];

                /*
                 * this code is used to test whether the gradient is right and should not appear when you use it

                double start = loss(tag, hiddenOutput);
                double epsilon = 0.01;
                outputLayer.addWeight(i, j, epsilon);
                double end = loss(tag, outputLayer.calOutput(hiddenLayer.calOutput(input)));
                double test_g = (end - start)/epsilon;
                outputLayer.minusWeight(i, j, epsilon);

                 * test is over
                 */

                err[0] += Math.abs(gradient[i][j]);
                val[j] += delta*outputLayer.getWeight(i)[j];
             }
         }

        return gradient;
    }

    public double[][] getHiddenToInputGradient(double[] tag, double[] err, double[] val){
        double[][] gradient = new double[hiddenNumber][inputNumber];

        for (int i = 0; i < hiddenNumber; i++){
            for (int j = 0; j < inputNumber; j++){
                double delta = val[i] * active.derivative(hiddenLayer.getWeight(i), input);
                gradient[i][j] = delta*input[j];

                 /*
                 * this code is used to test whether the gradient is right and should not appear when you use it

                double start = loss(tag, hiddenOutput);
                double epsilon = 0.001;
                hiddenLayer.addWeight(i, j, epsilon);
                double[] out = outputLayer.calOutput(hiddenLayer.calOutput(input));
                double end = loss(tag, out);
                double test_g = (end - start)/epsilon;
                hiddenLayer.minusWeight(i, j, epsilon);

                 * test is over
                 */

                err[0] += Math.abs(gradient[i][j]);
            }
        }

        return gradient;
    }

    public double loss(double[] tag, double[] v){
        double result = 0;
        int length = tag.length;
        for (int i = 0; i < length; i++){
            result += (tag[i]-v[i])*(tag[i]-v[i]);
        }

        return result/2;
    }

    public void train(double[][] X, double[][] Y){
        double err = 1;
        int length = X.length;
        while (err != 0){
            boolean flag = false;
            for (int i = 0; i < length; i++){
                err = update(X[i], Y[i], i==0);
                if (i == 0)
                    System.out.println("error : " + err);
                if (err < 0.001){
                    flag = true;
                    break;
                }
            }
            if (flag)
                break;
        }
    }

    public static void main(String[] args){
        Sigmoid sigmoid = new Sigmoid();
        BPNN bp = new BPNN(32, 15, 4, sigmoid, 0.25, 0.5);


        Random random = new Random();
        List<Integer> list = new ArrayList<Integer>();
        for (int i = 0; i != 1000; i++) {
            int value = random.nextInt();
            list.add(value);
        }

        double[][] X = new double[200][32];
        double[][] Y = new double[200][4];

        for (int i = 0; i < 200; i++) {
            for (int j = 0; j  < 32; j++){
                X[i][j] = random.nextInt(10);
            }
            for (int j = 0; j < 4; j++){
                Y[i][j] = random.nextInt(4);
            }
        }



        bp.train(X, Y);

    }
}
