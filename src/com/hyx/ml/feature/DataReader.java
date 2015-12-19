package com.hyx.ml.feature;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

/**
 * Created by yixuanhe on 11/20/15.
 */
public class DataReader {
    public static Data dataRead(String fileName, int num, int featurenum) throws IOException {
        BufferedReader in=new BufferedReader(new FileReader(fileName));

        double[][] X = new double[num][featurenum];
        double[][] Y = new double[num][1];

        String line;
        int n = 0;
        while ((line = in.readLine()) != null){
            String[] feature = line.split(",");
            int len = feature.length;

            for (int i = 0; i < len - 1; i++){
                X[n][i] = Double.parseDouble(feature[i]);
            }

            Y[n][0] = Double.parseDouble(feature[len-1]);
            n++;
        }
        in.close();

        return new Data(X, Y);
    }

    public static Data2 intDataRead(String fileName, int num) throws IOException {
        BufferedReader in=new BufferedReader(new FileReader(fileName));

        double[][] X = new double[num][300];
        int[][] Y = new int[num][1];

        String line;
        int n = 0;
        while ((line = in.readLine()) != null){
            String[] feature = line.split(",");
            int len = feature.length;

            for (int i = 0; i < len - 1; i++){
                X[n][i] = Double.parseDouble(feature[i]);
            }

            Y[n][0] = Integer.parseInt(feature[len - 1]);
            n += 1;
        }
        in.close();

        return new Data2(X, Y);
    }

    public static void main(String[] args){
    }
}
