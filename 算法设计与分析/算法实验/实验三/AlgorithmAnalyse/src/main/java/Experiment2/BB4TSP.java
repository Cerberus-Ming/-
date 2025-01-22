package Experiment2;

import java.util.Collections;
import java.util.LinkedList;
import java.util.Vector;

public class BB4TSP {

	int NoEdge = -1; //表示没有边
	private int minCost = Integer.MAX_VALUE; //当前最小代价
	private int[][] minDegreePerNode = {}; //每个节点的最小入度与出度
	private LinkedList<HeapNode> heap = new LinkedList<>(); //存储活节点
	private Vector<Integer> bestH = new Vector<>(); //最优节点排列

	public int getMinCost() {
		return minCost;
	}

	public void setMinCost(int minCost) {
		this.minCost = minCost;
	}

	/**
	 * 计算部分解的下界
	 * @param cityArrange 城市的排列
	 * @param cMatrix     邻接矩阵，第0行，0列不算
	 * @throws IllegalArgumentException
	 */
	public int computeLB(Vector<Integer> cityArrange, int level, int[][] cMatrix) {
		int lb = 0;
		int average;
		int n = cMatrix.length - 1;
		for (int i = 1; i <= level - 1; i++) {
			lb += cMatrix[cityArrange.get(i)][cityArrange.get(i + 1)];
		}
		if (level == n && cMatrix[cityArrange.get(level)][1] != -1) lb += cMatrix[cityArrange.get(level)][1];
		else if (level == n && cMatrix[cityArrange.get(level)][1] == -1) lb = Integer.MAX_VALUE;
		else if (level != n) {
			for (int i = level; i <= n; i++) {
				int node = cityArrange.get(i);
				if (i != n) {
					average = (minDegreePerNode[node][0] + minDegreePerNode[cityArrange.get(i + 1)][1]) / 2;
					lb += average;
				} else lb += minDegreePerNode[1][1];
			}
		}
		return lb;
	}

	/**
	 * 分支界限法解决TSP问题
	 *
	 * @param cMatrix 邻接矩阵，第0行，0列不算
	 * @param n       城市个数
	 * @throws IllegalArgumentException
	 */
	public int bb4TSP(int[][] cMatrix, int n) {
		//构造初始节点
		Vector<Integer> cityArrange = new Vector<>(); //城市排列
		cityArrange.add(0);
		for (int i = 1; i <= n; i++) cityArrange.add(i);
		int level = 2; //0-level的城市是已经排好的
		minDegreePerNode = calculate(cMatrix);
		int lcost = computeLB(cityArrange, level, cMatrix); //代价的下界
		HeapNode currentNode = new HeapNode(cityArrange, lcost, 1);
		heap.add(currentNode);
		while (level <= n) {
			//参考优先队列，不停扩展节点,选取下一个节点
			if (level == n) {
				if (currentNode.lcost < minCost) {
					bestH = currentNode.cityArrange;
					minCost = currentNode.lcost;
					break;
				}
			} else {
				for (int i = level; i <= n; i++) {
					if ((cMatrix[cityArrange.get(level - 1)][cityArrange.get(i)] != -1)) {
						Collections.swap(currentNode.cityArrange, level, i);
						int temp = computeLB(currentNode.cityArrange, level, cMatrix);
						HeapNode node = new HeapNode(currentNode.cityArrange, temp, level + 1);
						if (level + 1 == n) node.lcost = computeLB(currentNode.cityArrange, level + 1, cMatrix);
						heap.add(node);
						Collections.swap(currentNode.cityArrange, level, i);
					}
				}
			}
			if (heap.isEmpty()) break;
			Collections.sort(heap);
			currentNode = heap.pop();
			level = currentNode.level;
		}
		return minCost;
	}

	/**
	 * 计算矩阵中每一行每一列的最小值
	 *
	 * @param cMatrix 邻接矩阵
	 * @return 表示每一行每一列最小值的矩阵
	 */
	private int[][] calculate(int[][] cMatrix) {
		int[][] minDegreePerNode = new int[cMatrix.length][2];
		for (int i = 0; i < cMatrix.length; i++) {
			int minRow = Integer.MAX_VALUE;
			int minColumn = Integer.MAX_VALUE;
			for (int j = 0; j < cMatrix.length; j++) {
				if (cMatrix[i][j] < minRow && cMatrix[i][j] > 0) minRow = cMatrix[i][j];
				if (cMatrix[j][i] < minColumn && cMatrix[j][i] > 0) minColumn = cMatrix[j][i];
			}
			minDegreePerNode[i][0] = minRow;
			minDegreePerNode[i][1] = minColumn;
		}
		return minDegreePerNode;
	}

	/**
	 * 定义分支界限法采用的数据结构
	 */
	@SuppressWarnings("rawtypes")
	public static class HeapNode implements Comparable {
		Vector<Integer> cityArrange = new Vector<>(); //城市排列
		int lcost; // 下界
		int level; //所在层  0-level的城市是已经排好的

		//构造方法
		public HeapNode(Vector<Integer> node, int lb, int lev) {
			cityArrange.addAll(0, node);
			lcost = lb;
			level = lev;
		}

		@Override
		public int compareTo(Object x) { //升序排列, 每一次pollFirst
			int xu = ((HeapNode) x).lcost;
			if (lcost < xu) return -1;
			if (lcost == xu) return 0;
			return 1;
		}

		public boolean equals(Object x) {
			return lcost == ((HeapNode) x).lcost;
		}

	}


}
