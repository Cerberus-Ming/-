package Experiment2;

import java.util.Collections;
import java.util.LinkedList;
import java.util.Vector;

public class BB4TSP {

	int NoEdge = -1; //��ʾû�б�
	private int minCost = Integer.MAX_VALUE; //��ǰ��С����
	private int[][] minDegreePerNode = {}; //ÿ���ڵ����С��������
	private LinkedList<HeapNode> heap = new LinkedList<>(); //�洢��ڵ�
	private Vector<Integer> bestH = new Vector<>(); //���Žڵ�����

	public int getMinCost() {
		return minCost;
	}

	public void setMinCost(int minCost) {
		this.minCost = minCost;
	}

	/**
	 * ���㲿�ֽ���½�
	 * @param cityArrange ���е�����
	 * @param cMatrix     �ڽӾ��󣬵�0�У�0�в���
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
	 * ��֧���޷����TSP����
	 *
	 * @param cMatrix �ڽӾ��󣬵�0�У�0�в���
	 * @param n       ���и���
	 * @throws IllegalArgumentException
	 */
	public int bb4TSP(int[][] cMatrix, int n) {
		//�����ʼ�ڵ�
		Vector<Integer> cityArrange = new Vector<>(); //��������
		cityArrange.add(0);
		for (int i = 1; i <= n; i++) cityArrange.add(i);
		int level = 2; //0-level�ĳ������Ѿ��źõ�
		minDegreePerNode = calculate(cMatrix);
		int lcost = computeLB(cityArrange, level, cMatrix); //���۵��½�
		HeapNode currentNode = new HeapNode(cityArrange, lcost, 1);
		heap.add(currentNode);
		while (level <= n) {
			//�ο����ȶ��У���ͣ��չ�ڵ�,ѡȡ��һ���ڵ�
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
	 * ���������ÿһ��ÿһ�е���Сֵ
	 *
	 * @param cMatrix �ڽӾ���
	 * @return ��ʾÿһ��ÿһ����Сֵ�ľ���
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
	 * �����֧���޷����õ����ݽṹ
	 */
	@SuppressWarnings("rawtypes")
	public static class HeapNode implements Comparable {
		Vector<Integer> cityArrange = new Vector<>(); //��������
		int lcost; // �½�
		int level; //���ڲ�  0-level�ĳ������Ѿ��źõ�

		//���췽��
		public HeapNode(Vector<Integer> node, int lb, int lev) {
			cityArrange.addAll(0, node);
			lcost = lb;
			level = lev;
		}

		@Override
		public int compareTo(Object x) { //��������, ÿһ��pollFirst
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
