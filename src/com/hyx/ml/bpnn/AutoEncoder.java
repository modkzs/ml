package com.hyx.ml.bpnn;

import java.util.Random;

/**
 * Created by yixuanhe on 11/14/15.
 */

/*
 * this is a auto encoder implied by bpnn
 */
public class AutoEncoder {
    private BPNN bpnn;
    int inputLen;

    public AutoEncoder(int inputLen, int codeLen){
        Sigmoid sigmoid = new Sigmoid();

        this.bpnn = new BPNN(inputLen, codeLen, inputLen, sigmoid, 0.05, 0.5, 0.05);
        this.inputLen = inputLen;

    }

    // used to construct a auto encoder, which is training process of bpnn
    public void construct(){
        int num = 1000;
        double[][] X = new double[num][inputLen];
        Random rand = new Random();

        for (int i = 0; i < num; i++){
            for (int j = 0; j < inputLen; j++){
                X[i][j] = rand.nextInt(2);
            }
        }

        bpnn.train(X, X);
    }

    public double[] getCode(double[] X){
        return bpnn.getHiddenOutput(X);
    }

    public static void main(String[] args){
        AutoEncoder autoEncoder = new AutoEncoder(32, 16);

        double X[] = new double[32];

        Random rand = new Random();
        autoEncoder.construct();


        for (int i = 0; i < 32; i++) {
            X[i] = rand.nextInt(2);
            System.out.print(X[i] + "");
        }
        System.out.println();


        double[] Y = autoEncoder.getCode(X);
        for (double y : Y)
            System.out.print(y + " ");
    }

}
