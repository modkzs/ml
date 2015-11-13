package com.hyx.ml;

/**
 * Created by yixuanhe on 11/13/15.
 */
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class BPNNTest {

    /**
     * @param args
     * @throws IOException
     */
    public static void main(String[] args) throws IOException {
        Sigmoid sigmoid = new Sigmoid();
        BPNN bp = new BPNN(32, 15, 4, sigmoid, 0.25, 0.5);

        Random random = new Random();
        List<Integer> list = new ArrayList<Integer>();

        int num = 1000;

        for (int i = 0; i != num; i++) {
            int value = random.nextInt();
            list.add(value);
        }

        double[][] X = new double[num][32];
        double[][] Y = new double[num][4];


        for (int i = 0; i < num; i++) {
            int value = list.get(i);
            double[] real = new double[4];
            if (value >= 0)
                if ((value & 1) == 1)
                    Y[i][0] = 1;
                else
                    Y[i][1] = 1;
            else if ((value & 1) == 1)
                Y[i][2] = 1;
            else
                Y[i][3] = 1;


            double[] binary = new double[32];
            int index = 31;
            do {
                binary[index--] = (value & 1);
                value >>>= 1;
            } while (value != 0);
            X[i] = binary;
        }

        bp.train(X, Y);
        int wrong = 0;

        for (int i = 0; i < num; i++) {
            int value = list.get(i);
            double[] real = new double[4];
            if (value >= 0)
                if ((value & 1) == 1)
                    Y[i][0] = 1;
                else
                    Y[i][1] = 1;
            else if ((value & 1) == 1)
                Y[i][2] = 1;
            else
                Y[i][3] = 1;


            double[] binary = new double[32];
            int index = 31;
            do {
                binary[index--] = (value & 1);
                value >>>= 1;
            } while (value != 0);
            X[i] = binary;
        }

        for (int j = 0; j < num; j++){
            double[] result = bp.getOutput(X[j]);

            double max = -Integer.MIN_VALUE;
            int idx = -1;

            for (int i = 0; i != result.length; i++) {
                if (result[i] > max) {
                    max = result[i];
                    idx = i;
                }
            }

            if(Y[j][idx] == 0){
                wrong += 1;
            }

        }
        System.out.print(wrong);
    }

}
