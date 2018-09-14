---
title: "排序算法"
layout: page
date: 2018-09-06 00:00
---

[TOC]

## 冒泡排序
冒泡排序算法的运作如下：
- 比较相邻的元素，如果前一个比后一个大，就把它们两个调换位置。
- 对每一对相邻元素作同样的工作，从开始第一对到结尾的最后一对。这步做完后，最后的元素会是最大的数。
- 针对所有的元素重复以上的步骤，除了最后一个。
- 持续每次对越来越少的元素重复上面的步骤，直到没有任何一对数字需要比较。
```
// 冒泡排序
// 最差时间复杂度 ---- O(n^2)
// 最优时间复杂度 ---- 如果能在内部循环第一次运行时,使用一个旗标来表示有无需要交换的可能,可以把最优时间复杂度降低到O(n)
// 平均时间复杂度 ---- O(n^2)
// 所需辅助空间 ------ O(1)
// 稳定性 ------------ 稳定
void bubble_sort(vector<int>& x) {
	for (int i = 0; i < x.size() - 1; ++i) {	//i只是一个计数作用，不是下标，而且只需要遍历x.size()-1次就可以了
	for (int j = 0; j < x.size() - i - 1; ++j) { //这里j要小于x.size()-i-1，因为只要遍历到倒数第二个元素就可以了
			if (x[j] > x[j + 1]) 
				swap(x[j], x[j + 1]);
			
		}
	}
}
```

## 选择排序
选择排序也是一种简单直观的排序算法。它的工作原理很容易理解：初始时在序列中找到最小（大）元素，放到序列的起始位置作为已排序序列；然后，再从剩余未排序元素中继续寻找最小（大）元素，放到已排序序列的末尾。以此类推，直到所有元素均排序完毕。

注意选择排序与冒泡排序的区别：冒泡排序通过依次交换相邻两个顺序不合法的元素位置，从而将当前最小（大）元素放到合适的位置；而选择排序每遍历一次都记住了当前最小（大）元素的位置，最后仅需一次交换操作即可将其放到合适的位置。

```
// 选择排序
// 最差时间复杂度 ---- O(n^2)
// 最优时间复杂度 ---- O(n^2)
// 平均时间复杂度 ---- O(n^2)
// 所需辅助空间 ------ O(1)
// 选择排序是不稳定的排序算法，不稳定发生在最小元素与A[i]交换的时刻。
void SelectSort(vector<int>& x) {
	for (int i = 0; i < x.size(); ++i) {
		int max_pos = 0;
		for (int j = 0; j < x.size() - i; ++j){
			max_pos = x[max_pos] < x[j] ? j : max_pos;
		}
	    swap(x[max_pos],x[x.size() - 1 - i]);
	}
}
```
## 插入排序
插入排序是一种简单直观的排序算法。它的工作原理非常类似于我们抓扑克牌
<img src="/wiki/static/images/pca/charu.png" alt="joey"/>
对于未排序数据(右手抓到的牌)，在已排序序列(左手已经排好序的手牌)中从后向前扫描，找到相应位置并插入。
插入排序在实现上，通常采用in-place排序（即只需用到O(1)的额外空间的排序），因而在从后向前扫描过程中，需要反复把已排序元素逐步向后挪位，为最新元素提供插入空间。
```
// 插入排序
// 最差时间复杂度 ---- 最坏情况为输入序列是降序排列的,此时时间复杂度O(n^2)
// 最优时间复杂度 ---- 最好情况为输入序列是升序排列的,此时时间复杂度O(n)
// 平均时间复杂度 ---- O(n^2)
// 所需辅助空间 ------ O(1)
// 稳定性 ------------ 稳定
void InsertSort(vector<int> & x) {
	for (int i = 1; i < x.size(); ++i) {
		int j = i;
		while (j>0&& x[j] < x[j - 1]) {  //注意j>0 要在x[j] < x[j - 1] 前面！！要先判断j>0
			swap(x[j], x[j - 1]);
			j--;
		}

	}
}
```

## 希尔排序
希尔排序，也叫递减增量排序，是插入排序的一种更高效的改进版本。
希尔排序是基于插入排序的以下两点性质而提出改进方法的：
- 插入排序在对几乎已经排好序的数据操作时，效率高，即可以达到线性排序的效率
- 但插入排序一般来说是低效的，因为插入排序每次只能将数据移动一位

希尔排序通过将比较的全部元素分为几个区域来提升插入排序的性能。这样可以让一个元素可以一次性地朝最终位置前进一大步。然后算法再取越来越小的步长进行排序，算法的最后一步就是普通的插入排序，但是到了这步，需排序的数据几乎是已排好的了（此时插入排序较快）。
```
// 最差时间复杂度 ---- 根据步长序列的不同而不同。已知最好的为O(n(logn)^2)
// 最优时间复杂度 ---- O(n)
// 平均时间复杂度 ---- 根据步长序列的不同而不同。
// 所需辅助空间 ------ O(1)
// 稳定性 ------------ 不稳定 希尔排序是不稳定的排序算法，虽然一次插入排序是稳定的，不会改变相同元素的相对顺序，但在不同的插入排序过程中，相同的元素可能在各自的插入排序中移动，最后其稳定性就会被打乱。
void ShellSort(vector<int> &x) {
	int gap = x.size() / 2;
	while (gap >= 1) {
		for (int i = 0; i < gap; ++i) {  //一共有gap组
			for (int j = i+gap; j < x.size(); j = j + gap) { //每组执行一次直接插入排序
				int k = j;
				while (k-gap>=0 && x[k] < x[k - gap]) {  
					swap(x[k], x[k - gap]);
					k-=gap;
				}
			}
		}
		gap /= 2;
	}
}
```
## 归并排序
归并排序的实现分为递归实现与非递归(迭代)实现。递归实现的归并排序是算法设计中分治策略的典型应用，我们将一个大问题分割成小问题分别解决，然后用所有小问题的答案来解决整个大问题。非递归(迭代)实现的归并排序首先进行是两两归并，然后四四归并，然后是八八归并，一直下去直到归并了整个数组。
归并排序算法主要依赖归并(Merge)操作。归并操作指的是将两个已经排序的序列合并成一个序列的操作。
```
// 最差时间复杂度 ---- O(nlogn)
// 最优时间复杂度 ---- O(nlogn)
// 平均时间复杂度 ---- O(nlogn)
// 所需辅助空间 ------ O(n)
// 稳定性 ------------ 稳定
void Merge(vector<int>&x, int left, int mid, int right) {
	int len = right - left + 1;
	vector<int> temp(len,0);
	int i = left, j = mid + 1, index = 0;
	while (i <= mid && j <=right) {
		temp[index++] = x[i] > x[j] ? x[j++] : x[i++]; // x[i] > x[j]保证归并排序的稳定性
	}
	while (i <= mid)
		temp[index++] = x[i++]; //别忘了i++
	while (j <= right)
		temp[index++] = x[j++];//别忘了j++
	for (i = 0; i < len; ++i) {
		x[left++] = temp[i];
	}
}
void MergeSort(vector<int>&x, int left, int right) {
	if (left == right)// 当待排序的序列长度为1时，递归开始回溯，进行merge操作
		return;
	int mid = (left + right) / 2;
	MergeSort(x, left, mid);
	MergeSort(x, mid+1, right);
	Merge(x, left, mid, right);
}
void MergeSortInteration(vector<int>&x) {
	int left, mid, right;
	for (int i = 1; i < x.size(); i*=2) { //i从1开始，每次i*=2，表示先11合并，再22合并，再44合并...
		int left = 0;
		while (left + i < x.size()) {
			mid = left + i - 1;
			right = mid + i <= x.size() - 1 ? mid + i : x.size() - 1; // 后一个子数组大小可能不够
			Merge(x, left, mid, right);
			left = right + 1;
		}
	}
}
```

## 快速排序
快速排序是由东尼·霍尔所发展的一种排序算法。在平均状况下，排序n个元素要O(nlogn)次比较。在最坏状况下则需要O(n^2)次比较，但这种状况并不常见。事实上，快速排序通常明显比其他O(nlogn)算法更快，因为它的内部循环可以在大部分的架构上很有效率地被实现出来。
快速排序使用分治策略(Divide and Conquer)来把一个序列分为两个子序列。步骤为：
- 从序列中挑出一个元素，作为"基准"(pivot).
- 把所有比基准值小的元素放在基准前面，所有比基准值大的元素放在基准的后面（相同的数可以到任一边），这个称为分区(partition)操作。
- 对每个分区递归地进行步骤1~2，递归的结束条件是序列的大小是0或1，这时整体已经被排好序了。

```
// 最差时间复杂度 ---- 每次选取的基准都是最大（或最小）的元素，导致每次只划分出了一个分区，需要进行n-1次划分才能结束递归，时间复杂度为O(n^2)
// 最优时间复杂度 ---- 每次选取的基准都是中位数，这样每次都均匀的划分出两个分区，只需要logn次划分就能结束递归，时间复杂度为O(nlogn)
// 平均时间复杂度 ---- O(nlogn)
// 所需辅助空间 ------ 主要是递归造成的栈空间的使用(用来保存left和right等局部变量)，取决于递归树的深度，一般为O(logn)，最差为O(n)       
// 稳定性 ---------- 不稳定
int Partition(vector<int>&x, int left, int right) { 
	//int index = RandomInRange(left,right);
	//swap(x[index],x[right]);
	//直接以最后一个元素作为基准
	int small = left - 1;
	for (int i = left; i < right; ++i) { //i<right 即可，因为i==right 是最后一个元素，即基准
		if (x[i] < x[right]) {
			small++;
			swap(x[i], x[small]);
		}
	}
	small++;
	swap(x[small], x[right]);
	return small;
}
void QuickSort(vector<int>&x,int left,int right) {
	if (left >= right)
		return;
	int index = Partition(x, left, right);
	if (index > left)
		QuickSort(x, left, index-1);
	if (index < right)
		QuickSort(x, index + 1, right);
}
```

## 堆排序
堆排序是指利用堆这种数据结构所设计的一种选择排序算法。堆是一种近似完全二叉树的结构（通常堆是通过一维数组来实现的），并满足性质：以最大堆为例，其中父结点的值总是大于它的孩子节点。
其基本思想为(大顶堆):
- 将初始待排序关键字序列(R1,R2....Rn)构建成大顶堆，建堆时要从第一个非叶子节点开始向上建立，建堆时注意交换之后都可能造成被交换的孩子节点不满足堆的性质，因此每次交换之后要重新对被交换的孩子节点进行调整
- 将堆顶元素R[1]与最后一个元素R[n]交换，此时得到新的无序区(R1,R2,......Rn-1)和新的有序区(Rn)
- 由于交换后新的堆顶R[1]可能违反堆的性质，因此需要对当前无序区(R1,R2,......Rn-1)调整为新堆，然后再次将R[1]与无序区最后一个元素交换，得到新的无序区(R1,R2....Rn-2)和新的有序区(Rn-1,Rn)。不断重复此过程直到有序区的元素个数为n-1，则整个排序过程完成
```
// 最差时间复杂度 ---- O(nlogn)
// 最优时间复杂度 ---- O(nlogn)
// 平均时间复杂度 ---- O(nlogn)
// 所需辅助空间 ------ O(1)
// 稳定性 ------------ 不稳定 堆排序是不稳定的排序算法，不稳定发生在堆顶元素与A[i]交换的时刻。
void build_heap(vector<int> &nums, int i, int heap_tail_index){//对以节点i为根节点的子树进行调整，调整为大顶堆
	int left = 2 * i + 1;
	int right = 2 * i + 2;
	if (left <= heap_tail_index && nums[left] > nums[i])
	{
		swap(nums[left], nums[i]);					//vector这个对象存在栈中，然后栈中有指向vector所存数据的地址，数据保存在堆中。
		build_heap(nums, left, heap_tail_index);	//交换之后都可能造成被交换的孩子节点不满足堆的性质，因此每次交换之后要重新对被交换的孩子节点进行调整
	}

	if (right <= heap_tail_index && nums[right] > nums[i])
	{
		swap(nums[right], nums[i]);
		build_heap(nums, right, heap_tail_index);
	}
}
void heap_sort(vector<int> &nums)
{
	for (int i = (nums.size() - 1) / 2; i >= 0; i--)//应该是(nums.size()-1)/2，而不是nums.size()/2 -1
	{												//先初始化最大堆 ，从下往上调整
		build_heap(nums, i, nums.size()-1);
	}
	for (int i = nums.size() - 1; i >= 0;)			//依次调整堆的大小（堆末尾元素对应的数组索引i) i--
	{
		swap(nums[0], nums[i]);
		i--; 
		build_heap(nums, 0, i);
	}
}
```
**注：swap(vector[x1],vector[x2]) 可以交换vector两元素。vector这个对象存在栈中，然后栈中有指向vector所存数据的地址，数据保存在堆中。swap交换的并不是vector的数据空间，而是对vector对象所在栈内存空间的16字节的地址进行了交换。**
## 参考
[常用排序算法总结](http://www.cnblogs.com/eniac12/p/5329396.html)
[堆排序](https://www.cnblogs.com/0zcl/p/6737944.html)
[vector数据存在栈中还是堆中](https://blog.csdn.net/kkkkkkkkq/article/details/80335471)
[STL vector swap 交换操作是这样的](https://blog.csdn.net/xi_niuniu/article/details/46877513)
