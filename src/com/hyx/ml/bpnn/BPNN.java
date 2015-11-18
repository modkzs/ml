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

    //the gradient in last epoch, used to speed up gd
    double[][] o2hgradient;
    double[][] h2igradient;

    //some parameters in bpnn
    private static double RATE = 0.25;
    private static double THRESHOLD = 0.5;
    private static double MONENTUM = 0.05;

    //the error threshold where whole alg stop
    private static double ERR = 0.000000001;
    private static double LOSS = 0.00001;
    // the processed data number in a batch process
    private final static int STEP = 15;

    //the momentum factor
    double momentum;

    /*
     * input_number : the number of input feature
     * hidden_number : the number of cells in hidden layers
     * output_number : the number of output
     * active : the active function
     * rate : the learning rate
     * threshold : the threshold using to judge whether a cell is active
     * momentum : momentum factor
     */
    public BPNN(int inputNumber, int hiddenNumber, int outputNumber, ActiveFunction active, double rate, double threshold, double momentum){
        this.hiddenLayer = new Layer(hiddenNumber, inputNumber, active, rate, threshold);
        this.outputLayer = new Layer(outputNumber, hiddenNumber, active, rate, threshold);
        this.outputNumber = outputNumber;
        this.hiddenNumber = hiddenNumber;
        this.inputNumber = inputNumber;
        this.active = active;
        this.momentum = momentum;

        this.o2hgradient = new double[inputNumber][hiddenNumber+1];
        this.h2igradient = new double[hiddenNumber][inputNumber+1];
    }

    public BPNN(int inputNumber, int hiddenNumber, int outputNumber, ActiveFunction active){
        this.hiddenLayer = new Layer(hiddenNumber, inputNumber, active, RATE, THRESHOLD);
        this.outputLayer = new Layer(outputNumber, hiddenNumber, active, RATE, THRESHOLD);
        this.outputNumber = outputNumber;
        this.hiddenNumber = hiddenNumber;
        this.inputNumber = inputNumber;
        this.active = active;
        this.momentum = MONENTUM;

        this.o2hgradient = new double[inputNumber][hiddenNumber+1];
        this.h2igradient = new double[hiddenNumber][inputNumber+1];
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

    /*
     * This method is used for autoEncoder, which should not be used in other place
     */
    public double[] getHiddenOutput(double[] value){
        getOutput(value);
        return hiddenOutput;
    }

    public double[] update(double[][] values, double[][] tags, int STEP){
        int length = values.length;
        double[][] O2HGradient = new double[inputNumber][hiddenNumber+1];
        double[][] H2IGradient = new double[hiddenNumber][inputNumber+1];

        double[] err = new double[1];
        double start = 0;
        for(int i = 0; i < length; i++) {
            double[] value = values[i];
            double[] tag = tags[i];

            getOutput(value);

            double[] val = new double[hiddenNumber + 1];
            getOutputToHiddenGradient(tag, err, val, O2HGradient);
            getHiddenToInputGradient(tag, err, val, H2IGradient);

            this.o2hgradient = O2HGradient;
            this.h2igradient = H2IGradient;

            outputLayer.update(O2HGradient);
            hiddenLayer.update(H2IGradient);


            start += loss(tag, hiddenOutput);
        }

        double[] result = new double[2];
        result[0] = err[0];result[1] = start;
        return result;
    }

    public void getOutputToHiddenGradient(double[] tag, double[] err, double[] val, double[][] gradient){
        for (int i = 0; i <= hiddenNumber; i++){
            val[i] = 0;
        }

        for(int i = 0; i < outputNumber; i++){
            double delta = active.derivative(outputLayer.getWeight(i), inputOutput) * (tag[i] - hiddenOutput[i]);
            for (int j = 0; j < hiddenNumber; j++) {
                gradient[i][j] += momentum * delta * inputOutput[j] + (1-momentum) * o2hgradient[i][j];

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
            gradient[i][hiddenNumber] += delta;
        }
    }

    public void getHiddenToInputGradient(double[] tag, double[] err, double[] val, double[][] gradient){
        for (int i = 0; i < hiddenNumber; i++){
            double delta = val[i] * active.derivative(hiddenLayer.getWeight(i), input);
            for (int j = 0; j < inputNumber; j++){
                gradient[i][j] += momentum * delta * input[j] + (1-momentum) * h2igradient[i][j];

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
            gradient[i][inputNumber] += delta;
        }
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
        double[] data;
        int length = X.length;
        while (true){
            boolean flag = false;
            for (int i = 0; i < length; i+=STEP){
                int len;
                if (i + STEP <= length)
                    len = STEP;
                else
                    len = length - i;

                double[][] train_X = new double[len][inputNumber];
                double[][] train_Y = new double[len][outputNumber];
                for (int j = 0; j < len; j++){
                    train_X[j] = X[i+j];
                    train_Y[j] = Y[i+j];
                }
                data = update(train_X, train_Y, len);

                if (i == 0) {
                    System.out.println(i + "th error : " + data[0]);
                    System.out.println(i + "th loss : " + data[1]);
                }
                if (data[0] < ERR*len && data[1] < LOSS*len){
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
        BPNN bp = new BPNN(32, 15, 4, sigmoid, 1, 0.5, 0.05);


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
