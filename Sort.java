package sort;

import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.Date;

public class Sort {
	public static void main(String args[]){
		int [] Array = new int[] {5,2,8,1,4,9,7,6,3};
		//quickSort(Array,0,Array.length-1);
		heapSort(Array,Array.length-1);
		for(int i = 0;i < Array.length;i++){
			System.out.print(Array[i]+" ");
		}
	}
	
	private static void InsertSort(int[] array) {
		// TODO Auto-generated method stub
		int i,j,temp;
		for(i=1;i<array.length;i++) {
			out(array);
			System.out.println();
			temp=array[i];
			for(j=i-1;j>=0;j--) {
				if(temp>array[j]) {
					break;
				}else {
					array[j+1]=array[j];
				}
			}
			array[j+1]=temp;
		}
	}

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
	
	private static void heapSort(int arr[],int len){//堆排序
		BuildMaxHeap(arr,len);
		for(int i=len;i>0;i--){
			swap(arr,i,1);
			AdjustDown(arr,1,i-1);
		}
	}
	private static void swap(int[] nums, int index, int j) {
		// TODO Auto-generated method stub
		int temp=nums[index];
		nums[index]=nums[j];
		nums[j]=temp;
	}

	private static void AdjustDown(int[] arr, int k, int len) {
		// TODO Auto-generated method stub
		arr[0]=arr[k];
		for(int i=2*k;i<=len;i=i*2){
			if(i<len&&arr[i]<arr[i+1])
				i++;
			if(arr[0]>=arr[i])break;
			else{
				arr[k]=arr[i];
				k=i;
			}
		}
		arr[k]=arr[0];
	}
	private static void BuildMaxHeap(int[] arr, int len) {
		// TODO Auto-generated method stub
		for(int i=len/2;i>0;i--){
			AdjustDown(arr,i,len);
		}
	}
	
	private static void out(int[] array) {
		// TODO Auto-generated method stub
		for(int i = 0;i < array.length;i++){
			System.out.print(array[i]+" ");
		}
	}
}
