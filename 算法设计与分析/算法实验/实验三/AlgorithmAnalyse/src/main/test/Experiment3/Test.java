package Experiment3;

import java.util.Scanner;

public class Test {
    static int n;
    static int popNum = 100;
    static int length;
    static double bestRes;
    static int[] codes;
    static int[] bestCodes;
    static int[][] a;
    static int[][] iniPop;
    static double[] fitness;
    static GAOperations gaOperations = new GAOperations();

    static void initial() {
        length = n;
        a = new int[n][n];
        fitness = new double[popNum];
        codes = new int[n];
        bestCodes = new int[n];
        iniPop = new int[popNum][n];
        for (int i = 0; i < n; i++) {
            bestCodes[i] = i + 1;
        }

    }

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);
        n = sc.nextInt();
        initial();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                a[i][j] = sc.nextInt();
            }
        }
        long tt = System.currentTimeMillis();
        bestRes = GAOperations.computeFitness(bestCodes, length, a);
        gaOperations.randomInitialization(popNum, length, bestCodes, 0, codes, iniPop);
        int T = 100000, cgf = 0;
        for (int t = 1; t <= T; t++) {
            if (cgf > 2000) break;
            cgf++;
            for (int i = 0; i < popNum; i++) {
                fitness[i] = GAOperations.computeFitness(iniPop[i], length, a);
                if (fitness[i] > bestRes) {
                    cgf = 0;
                    bestRes = fitness[i];
                    if (length >= 0) System.arraycopy(iniPop[i], 0, bestCodes, 0, length);
                }
            }
            GAOperations.disturbance(iniPop, popNum, length, 0);
            GAOperations.roundBet(popNum, length, iniPop, fitness);
        }
        int BR = 0;
        for (int i = 0; i < length - 1; i++)
            BR += a[bestCodes[i] - 1][bestCodes[i + 1] - 1];
        BR += a[bestCodes[length - 1] - 1][bestCodes[0] - 1];
        System.out.print("解： " + BR + "\n路径： ");
        for (int i = 0; i < length; i++)
            System.out.print(bestCodes[i] + " ");
        System.out.println("\n用时： " + (System.currentTimeMillis() - tt) + "ms");
    }
}
