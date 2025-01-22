package Experiment1;

public class GamePassProbability {

	public double calculatePassProbability(int[] p, int num) {
		// 初始化变量，注意要用double类型的变量，否则测试不通过
		double passProbability = 0.0d;
		double[][] dp = new double[num + 1][num + 1];
		dp[0][0] = 1d;

		// 初始化二维数组
		for (int i = 1; i <= num; i++) {
			dp[i][i] = dp[i - 1][i - 1] * p[i - 1] * 0.01d;
		}

		// 动态规划计算赢得至少j场比赛的概率
		for (int j = 0; j <= num - 1; j++) {
			for (int i = j + 1; i <= num; i++) {
				if (j != 0) {
					// 更新动态规划数组
					dp[i][j] = dp[i - 1][j] * (1 - 0.01d * p[i - 1]) + dp[i - 1][j - 1] * p[i - 1] * 0.01d;
				} else {
					dp[i][j] = dp[i - 1][0] * (1 - 0.01d * p[i - 1]);
				}
			}
		}

		// 加和计算成功晋级段位的概率
		for (int j = (int) Math.ceil(0.7 * num); j <= num; j++) {
			passProbability = passProbability + dp[num][j];
		}

		// 四舍五入并输出结果
		double res = (double) Math.round(passProbability * 100000) / 100000;
		System.out.println(res);
		return res;
	}.

}
