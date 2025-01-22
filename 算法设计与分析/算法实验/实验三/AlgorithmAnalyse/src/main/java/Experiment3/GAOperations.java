package Experiment3;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;

public class GAOperations {
	/**
	 * ���������ʼ�⣬˼·���Ȳ��������޸���Ҳ���Ա߲������޸����������λ�õĴ���������࣬���������������.
	 *
	 * @param popNum ��Ⱥ��С
	 * @param length  ÿһ�����峤��.
	 * @param iniPop  �����ĳ�ʼ��Ⱥ.
	 * @param codes   ��������.
	 * @param codeNum   ���������.
	 * @param codeCount  ÿһ������ļ���.
	 */
	public void randomInitialization(int popNum, int length, int[] codes, int codeNum, int[] codeCount, int[][] iniPop) {
		int i, j;
		int[] nJs = new int[codeNum];//ͳ��ÿ���������������
		Random random = new Random();
		//����������룬��ȥ�أ��޸�
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
	 * @param pop ����
	 * @param length  ���峤��.
	 * @param a �ڽӾ���
	 */
	public static double computeFitness(int[] pop, int length, int[][] a)
	{
		//���������Ӧ��
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
	 * @param popNum ���� ����
	 * @param length  ���峤��.
	 * @param iniPop1  ��Ⱥ
	 * @param fitness ÿһ���������Ӧ��
	 */
	public static void roundBet(int popNum, int length, int[][] iniPop1, double[] fitness)
	{
		//���̶�
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
	 * @param iniPop  ��Ⱥ
	 * @param popNum ���� ����
	 * @param length  ���峤��.
	 * @param disPos  ���������λ����
	 */
	public static void disturbance(int [][] iniPop, int popNum, int length, int disPos)
	{
		//�Ŷ�
		Random random = new Random();
		//���ڸ�����������
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
					//����˳��
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
		//��������������򣬼�����
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
	 * ��ȡcode��codes�е�λ��
	 * @param code  ����
	 * @param codeNum �ܱ�����
	 * @param codes  �������.
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
