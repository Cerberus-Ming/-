package Experiment3;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class GAOperations {
	/**
	 * 随机产生初始解，思路：先产生，后修复（也可以边产生边修复，如产生的位置的代码计数过多，则重新随机产生）.
	 *
	 * @param popNum 种群大小
	 * @param length  每一个个体长度.
	 * @param iniPop  产生的初始种群.
	 * @param codes   编码序列.
	 * @param codeNum   编码的数量.
	 * @param codeCount  每一个编码的计数.
	 */
	public void randomInitialization(int popNum, int length, int[] codes, int codeNum, int[] codeCount, int[][] iniPop) {
		int i, j;
		int[] nJs = new int[codeNum];//统计每个编码产生的数量
		Random random = new Random();
		//随机产生编码，并去重，修复
		for(i = 0; i < popNum; i++) {
			for (j = 0; j < codeNum; j++) {
				nJs[j] = 0;
			}
			for (j = 0;j < length;j++) {
				int pos = -1;
				do {
					if(pos != -1) {
						nJs[pos]--;
					}
					int code = random.nextInt(codeNum) + 1;
					pos = getCodePos(code,codeNum,codes);
					nJs[pos]++;
					iniPop[i][j] = code;
				} while(nJs[pos] > codeCount[pos]);
			}
		}
	}

	/**
	 *
	 * @param pop 个体
	 * @param length  个体长度.
	 * @param a 邻接矩阵
	 */
	public static double computeFitness(int[] pop, int length, int[][] a)
	{
		//计算个体适应度
		double fitness = 0.0;
		for (int i = 1; i < length; i++) {
			fitness = fitness + a[pop[i-1]-1][pop[i] - 1];
		}
		fitness += a[pop[length - 1] - 1][pop[0] -1];
		fitness = 1/fitness;
		return fitness;
	}

	/**
	 *
	 * @param popNum 个体 个数
	 * @param length  个体长度.
	 * @param iniPop1  种群
	 * @param fitness 每一个个体的适应度
	 */
	public static void roundBet(int popNum, int length, int[][] iniPop1, double[] fitness)
	{
		//轮盘赌
		Random random = new Random();
		int[][] iniPop2 = new int[popNum][length];
		double sum = 0.0;
		for (int i = 0; i < popNum; i++) {
			sum += fitness[i];
		}
		for (int i = 0; i < popNum; i++) {
			double fit = random.nextDouble() * sum;
			for (int j = 0; j < popNum; j++) {
				fit = fit - fitness[j];
				if (fit < 0) {
					for(int k = 0; k < length; k++) {
						iniPop2[i][k] = iniPop1[j][k];
					}
					break;
				}
			}
		}
		for (int i = 0; i < popNum; i++) {
			for (int j = 0; j < length; j++) {
				iniPop1[i][j] = iniPop2[i][j];
			}
		}
	}


	/**
	 *
	 * @param iniPop  种群
	 * @param popNum 个体 个数
	 * @param length  个体长度.
	 * @param disPos  随机交换的位置数
	 */
	public static void disturbance(int [][] iniPop, int popNum, int length, int disPos)
	{
		//扰动
		Random random = new Random();
		//相邻个体两两交叉
		for (int i = 0; i < popNum; i=i+2) {
			if (random.nextDouble() <= 0.5) {
				for (int j = 0; j < disPos; j++) {
					int temp = iniPop[i][j];
					iniPop[i][j] = iniPop[i+1][j];
					iniPop[i+1][j] = temp;
				}

				int m = i-1;
				while(m++ < i+1) {
					int[] njs = new int[length];
					for (int j = 0; j<length; j++) {
						njs[j] = 0;
					}
					for (int j = 0; j<length; j++) {
						int pos = iniPop[m][j] - 1;
						njs[pos]++;
						if (njs[pos] > 1) {
							iniPop[m][j] = 0;
						}
					}
					List<Integer> list = new ArrayList<Integer>();
					for (int j = 0; j<length; j++) {
						if(njs[j] == 0) {
							list.add(j+1);
						}
					}
					//打乱顺序
					Collections.shuffle(list);
					int k = 0;
					for (int j = 0; j<length; j++) {
						if (iniPop[m][j] == 0) {
							iniPop[m][j] = list.get(k++);
						}
					}
				}
			}
		}
		//随机交换两个基因，即变异
		for (int i = 0; i < popNum; i++) {
			if(random.nextDouble() <= 0.005) {
				int pos1 = random.nextInt(length);
				int pos2 = random.nextInt(length);
				while (pos1 == pos2) {
					pos2 = random.nextInt(length);
				}
				int temp = iniPop[i][pos1];
				iniPop[i][pos1] = iniPop[i][pos2];
				iniPop[i][pos2] = temp;
			}
		}
	}

	/**
	 * 获取code在codes中的位置
	 * @param code  编码
	 * @param codeNum 总编码数
	 * @param codes  编码矩阵.
	 */
	public static int getCodePos(int code, int codeNum, int[] codes)
	{
		int pos = 0;
		for(; pos < codeNum; pos++)
		{
			if(code == codes[pos])
			{
				return pos;
			}
		}
		return -1;
	}
}
