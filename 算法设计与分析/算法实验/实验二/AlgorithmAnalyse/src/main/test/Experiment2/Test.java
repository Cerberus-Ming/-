package Experiment2;

import java.util.Random;

public class Test {

    @org.junit.Test
    public void test() {

        Random random = new Random();
        int[] numList = {4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        for (int num : numList) {
            int[][] matrix = new int[num + 1][num + 1];
            for (int i = 0; i < num + 1; i++) {
                for (int j = 0; j < num + 1; j++) {
                    if (i == 0 || j == 0 || j == i) {
                        matrix[i][j] = -1;
                    } else {
                        matrix[i][j] = random.nextInt(1, 81);
                    }
                }
            }
            BB4TSP bb4TSP = new BB4TSP();
            Back4TSP back4TSP = new Back4TSP();
            long t1 = System.currentTimeMillis();
            bb4TSP.bb4TSP(matrix, num);
            long t2 = System.currentTimeMillis();
            long t3 = System.currentTimeMillis();
            back4TSP.backtrack4TSP(matrix, num);
            long t4 = System.currentTimeMillis();
            System.out.println("城市数：" + num + "\n回溯法计算用时：" + (t4 - t3) + " ，解：" + back4TSP.bestc+ "\n分支限界法用时：" + (t2 - t1) + " ，解：" + bb4TSP.getMinCost());
        }
    }
}


