package Experiment2;

public class Back4TSP {

	int NoEdge = -1;
	int bigInt = Integer.MAX_VALUE;
	int[][] a; // 邻接矩阵
	int cc = 0; // 存储当前代价
	int bestc = bigInt;// 当前最优代价
	int[] x; // 当前解
	int[] bestx;// 当前最优解
	int n = 0; // 顶点个数

	/**
	 * 回溯法
	 * @param i 初始深度
	 */
	private void backtrack(int i) {
		if (i > n) {
			bestc = cc; // 更新当前最优代价
			bestx = x.clone(); // 更新当前最优解
		} else {
			for (int j = i; j <= n; j++) {
				if (j != i) {
					swap(i, j); // 交换城市顺序，尝试不同的访问顺序
				}
				if (check(i) && cc < bestc) {
					backtrack(i + 1); // 递归深度优先搜索下一个城市
					if (i == n) {
						cc -= a[x[n]][1]; // 回溯时减去回到起始城市的代价
					}
					cc -= a[x[i - 1]][x[i]]; // 回溯时减去当前城市之前的代价
				}
				if (j != i) {
					swap(i, j); // 恢复城市顺序
				}
			}
		}
	}

	/**
	 * 交换城市顺序
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
	 * 判断是否满足约束条件，进行剪枝
	 *
	 * @param pos 当前访问的城市位置
	 * @return 是否满足约束条件
	 */
	public boolean check(int pos) {
		if (pos != n) {
			if (a[x[pos - 1]][x[pos]] != -1) {
				cc += a[x[pos - 1]][x[pos]]; // 累加当前城市和前一个城市之间的代价
				return true;
			}
		} else {
			if ((a[x[pos - 1]][x[pos]] != -1) && (a[x[pos]][1] != -1)) {
				cc += a[x[pos - 1]][x[pos]] + a[x[pos]][1]; // 累加最后一个城市和起始城市之间的代价
				return true;
			}
		}
		return false; // 不满足约束条件，进行剪枝
	}

	/**
	 * 回溯法解决旅行商问题
	 *
	 * @param b   旅行商问题的邻接矩阵
	 * @param num 城市数量
	 */
	public void backtrack4TSP(int[][] b, int num) {
		n = num; // 设置城市数量
		x = new int[n + 1]; // 初始化城市访问序列
		for (int i = 0; i <= n; i++)
			x[i] = i; // 初始访问序列为城市编号
		bestx = new int[n + 1]; // 初始化当前最优解
		a = b; // 设置邻接矩阵
		backtrack(2); // 从第二个城市开始进行深度优先搜索
	}

}
