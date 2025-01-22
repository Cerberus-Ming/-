package Experiment2;

public class Back4TSP {

	int NoEdge = -1;
	int bigInt = Integer.MAX_VALUE;
	int[][] a; // �ڽӾ���
	int cc = 0; // �洢��ǰ����
	int bestc = bigInt;// ��ǰ���Ŵ���
	int[] x; // ��ǰ��
	int[] bestx;// ��ǰ���Ž�
	int n = 0; // �������

	/**
	 * ���ݷ�
	 * @param i ��ʼ���
	 */
	private void backtrack(int i) {
		if (i > n) {
			bestc = cc; // ���µ�ǰ���Ŵ���
			bestx = x.clone(); // ���µ�ǰ���Ž�
		} else {
			for (int j = i; j <= n; j++) {
				if (j != i) {
					swap(i, j); // ��������˳�򣬳��Բ�ͬ�ķ���˳��
				}
				if (check(i) && cc < bestc) {
					backtrack(i + 1); // �ݹ��������������һ������
					if (i == n) {
						cc -= a[x[n]][1]; // ����ʱ��ȥ�ص���ʼ���еĴ���
					}
					cc -= a[x[i - 1]][x[i]]; // ����ʱ��ȥ��ǰ����֮ǰ�Ĵ���
				}
				if (j != i) {
					swap(i, j); // �ָ�����˳��
				}
			}
		}
	}

	/**
	 * ��������˳��
	 *
	 * @param i
	 * @param j
	 */
	private void swap(int i, int j) {
		int temp = x[i];
		x[i] = x[j];
		x[j] = temp;
	}

	/**
	 * �ж��Ƿ�����Լ�����������м�֦
	 *
	 * @param pos ��ǰ���ʵĳ���λ��
	 * @return �Ƿ�����Լ������
	 */
	public boolean check(int pos) {
		if (pos != n) {
			if (a[x[pos - 1]][x[pos]] != -1) {
				cc += a[x[pos - 1]][x[pos]]; // �ۼӵ�ǰ���к�ǰһ������֮��Ĵ���
				return true;
			}
		} else {
			if ((a[x[pos - 1]][x[pos]] != -1) && (a[x[pos]][1] != -1)) {
				cc += a[x[pos - 1]][x[pos]] + a[x[pos]][1]; // �ۼ����һ�����к���ʼ����֮��Ĵ���
				return true;
			}
		}
		return false; // ������Լ�����������м�֦
	}

	/**
	 * ���ݷ��������������
	 *
	 * @param b   ������������ڽӾ���
	 * @param num ��������
	 */
	public void backtrack4TSP(int[][] b, int num) {
		n = num; // ���ó�������
		x = new int[n + 1]; // ��ʼ�����з�������
		for (int i = 0; i <= n; i++)
			x[i] = i; // ��ʼ��������Ϊ���б��
		bestx = new int[n + 1]; // ��ʼ����ǰ���Ž�
		a = b; // �����ڽӾ���
		backtrack(2); // �ӵڶ������п�ʼ���������������
	}

}
