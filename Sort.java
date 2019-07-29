/**
 * Created by wsbty on 2019/7/3.
 */
public class Sort {
    public static void main(String args[]){
        //int [] Array = new int[] {5,2,8,1,4,9,7,6,3};
        int [] Array = new int[] {5,4,3,2,1,6};
        quickSort(Array,0,Array.length-1);
        //heapSort(Array);
        for(int i = 0;i < Array.length;i++){
            System.out.print(Array[i]+" ");
        }
    }
    /*
    插入排序
    它的工作原理是通过构建有序序列，对于未排序数据，在已排序序列中从后向前扫描，找到相应位置并插入。
    空间效率：O(1)
    时间效率：向有序子表中逐个插入元素进行了n-1趟，每次分为比较关键字和移动元素
    最坏情况，元素顺序和最终结果相反，时间为O(n2)
    最好情况 已经有序，只需要比较一次而不用移动，时间为O(n)
    平均时间为n2/4 ,时间为O(n2)
    是稳定的算法
     */
    private static void InsertSort(int[] array) { //2 1
        // TODO Auto-generated method stub
        int i,j,temp;
        for(i = 1; i < array.length; i++) {
            temp = array[i];
            j = i - 1;
            while(j >= 0 && temp < array[j]){
                array[j+1] = array[j];
                j--;
            }
            array[j+1] = temp;

        }
    }

    /*
    希尔排序
    先将整个待排序的记录序列分割成为若干子序列分别进行直接插入排序
    举例8 9 1 7 2 3 5 4 6 0 10个数gap = 10/2=5，整个数组分为5组，83 85 14 76 20，每组插入排序
    空间效率：O(1)
    时间效率：O(nlogn)
    不稳定
     */
    private static void ShellSort(int arr[]){
        int len = arr.length;
        int temp, j, gap = len / 2;
        while (gap > 0){
            for(int i = gap;i<len;i++){
                temp = arr[i];
                j = i - gap;
                while(j >= 0 && temp < arr[j]){
                    arr[j+gap] = arr[j];//记录后移，查找插入位置
                    j = j -gap;
                }
                arr[j+gap] = temp;//不用后移找到位置
            }
            gap = gap / 2;
        }
    }

    /*
    冒泡排序
    将最小的元素交换到待排序列中的第一个位置上
    空间效率：O(1)
    时间效率：O(n2)
    最好情况 比较次数 n-1 移动次数 0 时间效率O(n)
    最坏情况，初始序列逆序，n-1趟排序 第i趟进行n-i次关键字比较且移动三次数据交换位置，时间效率O(n2)
    稳定的，i>j时 值相同 不交换
     */
    private static void BubbleSort(int [] array){
        int n = array.length;
        for(int i = 0;i < n - 1;i++){
            boolean flag = false;//表示本趟冒泡是否发生交换的标志
            for(int j = n - 1;j > i;j--){//一趟冒泡过程
                if(array[j-1] > array[j]){//若为逆序
                    swap(array,j-1,j);//交换
                    flag = true;
                }
            }
            if(flag == false)
                return;//本趟遍历后没有发生交换，说明已经有序
        }
    }

    /*
    快速排序
    在序列中任取一个元素pivot作为基准，通过一趟排序分为 小于pivot和大于等于pivot的两部分 pivot在最终位置上 两个子序列继续
    空间效率：平均O(log2n) 最坏O(n)
    时间效率：最坏O(n2) 平均O(nlog2n)
    不稳定的，若右端存在两个值一样的，都小于基准，一交换相对位置会变
    快排优化：
     */
    private static void quickSort(int [] arr,int low,int high){
        if(low<high){
            int pivot = partition(arr,low,high);
            quickSort(arr, low, pivot-1);
            quickSort(arr, pivot+1, high);
        }
    }

    private static int partition(int[] arr, int low, int high) {
        // TODO Auto-generated method stub
        int pivot = arr[low];
        while(low < high){
            while(low < high && arr[high] >= pivot)
                --high;
            arr[low] = arr[high];
            while(low < high && arr[low] <= pivot)
                ++low;
            arr[high] = arr[low];
        }
        arr[low] = pivot;

        return low;
    }

    /*
    选择排序
    第i趟排序从 i--n 中选一个最小的与arr[i] 交换
    空间效率：O(1)
    时间效率：平均O(n2)
    不稳定， （2） 2 1 第一趟 1 2 （2） 相对位置变化
     */
    private static void SelectSort(int arr[]){
        int n = arr.length;
        int min = 0;
        for(int i = 0;i < n;i++){
            min = i;
            for(int j = i + 1;j < n;j++){
                if(arr[j] < arr[min])
                    min = j;
            }
            swap(arr,i,min);
        }
    }

    /*
    堆排序
    空间效率：O(1)
    时间效率：建堆时间O(n) n-1次调整 每次调整O(h) h树高 平均O(nlog2n)
    不稳定
     */
    private static void heapSort(int arr[]){//堆排序
        BuildMaxHeap(arr);
        for(int i = arr.length-1;i > 0;i--){
            swap(arr,0,i);
            AdjustDown(arr,0,i);
        }
    }
    private static void BuildMaxHeap(int[] arr) {
        // TODO Auto-generated method stub
        for(int i = arr.length/2 - 1;i >= 0;i--){
            AdjustDown(arr,i,arr.length);
        }
    }
    //
    private static void AdjustDown(int[] arr, int k, int len) {
        // TODO Auto-generated method stub
        int temp = arr[k];
        for(int i = 2*k+1;i<len;i=i*2+1){//沿值较大的子节点向下筛选
            if(i+1 < len && arr[i] < arr[i+1])
                i++;//取值较大的子节点下标
            if(temp >= arr[i])//筛选结束
                break;
            else{
                arr[k] = arr[i];//把A[i] 调整到双亲节点上
                k = i;//修改k的值继续向下筛选
            }
        }
        arr[k] = temp;//被筛选节点的值放入最终位置
    }

    /*
    归并排序
    空间效率：O(n)
    时间效率：每趟归并O(n) 一共log2n 趟 平均O(nlog2n)
    稳定，因为merge操作不会改变相对位置
     */
    public static void mergeSort(int[] a, int start, int end){
        if(start<end){//当子序列中只有一个元素时结束递归
            int mid=(start+end)/2;//划分子序列
            mergeSort(a, start, mid);//对左侧子序列进行递归排序
            mergeSort(a, mid+1, end);//对右侧子序列进行递归排序
            merge(a, start, mid, end);//合并
        }
    }

    public static void merge(int []a,int left,int mid,int right){
        int []tmp=new int[a.length];//辅助数组
        int p1=left,p2=mid+1,k=left;//p1、p2是检测指针，k是存放指针

        while(p1<=mid && p2<=right){
            if(a[p1]<=a[p2])
                tmp[k++]=a[p1++];
            else
                tmp[k++]=a[p2++];
        }

        while(p1<=mid) tmp[k++]=a[p1++];//如果第一个序列未检测完，直接将后面所有元素加到合并的序列中
        while(p2<=right) tmp[k++]=a[p2++];//同上

        //复制回原树组
        for (int i = left; i <=right; i++)
            a[i]=tmp[i];
    }

    private static void out(int[] array) {
        // TODO Auto-generated method stub
        for(int i = 0;i < array.length;i++){
            System.out.print(array[i]+" ");
        }
    }

    private static void swap(int[] nums, int index, int j) {
        // TODO Auto-generated method stub
        int temp=nums[index];
        nums[index]=nums[j];
        nums[j]=temp;
    }
}
