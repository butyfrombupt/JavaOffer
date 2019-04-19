package leetcode.Fourty;

/**
 * Created by wsbty on 2019/4/19.
 */
import java.util.*;
public class Main {

    public void nextPermutation(int[] nums) {//31 下一个排列
        //从后往前找一个不符合递增的，再找一个比这个稍大的交换，然后反转整个后半部分
        int index=-1;
        for(int i=nums.length-1;i>0;i--){
            if(nums[i]>nums[i-1]){
                index=i-1;
                break;
            }
        }
        if(index>-1){
            for(int j=nums.length-1;j>index;j--){
                if(nums[j]>nums[index]){
                    swap(nums,index,j);
                    break;
                }
            }
        }
        reverseArray(nums, index + 1);
    }
    private static void reverseArray(int[] nums, int index) {
        // TODO Auto-generated method stub
        int i=index;int j=nums.length-1;
        while(i<j){
            swap(nums, i, j);
            i++;j--;
        }

    }
    private static void swap(int[] nums, int index, int j) {
        // TODO Auto-generated method stub
        int temp=nums[index];
        nums[index]=nums[j];
        nums[j]=temp;
    }

    public int search(int[] nums, int target) {//33 搜索旋转排序数组
        //把排序数组分为两部分，看看会落入哪一部分
        int index=-1;
        int left=0;
        int right=nums.length-1;
        while(left<=right){
            int mid=(left+right)/2;
            if(nums[mid]==target){
                index=mid;
                break;
            }
            if(nums[mid]>=nums[left]){
                if(nums[mid]>target && target>=nums[left]){
                    right = mid - 1;
                }else{
                    left = mid + 1;
                }
            }else{
                if(nums[mid]<target && target <=nums[right]){
                    left = mid + 1;
                }else{
                    right = mid - 1;
                }
            }
        }
        return index;
    }

    public int[] searchRange(int[] nums, int target) {//34 在排序数组中查找元素的第一个和最后一个位置
        int a[]=new int[]{-1,-1};
        if(nums.length==0)return a;
        int left=0;
        int right=nums.length-1;
        int index=0;
        while(left<=right){
            int mid=(left+right)/2;
            if(target == nums[mid])
            {
                index = mid;
                break;
            }
            if(target > nums[mid])
            {
                left = mid + 1;
            }
            if(target < nums[mid])
            {
                right = mid - 1;
            }
        }
        for(int i=index;i>=0;i--){
            if(nums[i]==target){
                a[0]=i;
            }
        }
        for(int j=index;j<nums.length;j++){
            if(nums[j]==target){
                a[1]=j;
            }
        }
        return a;
    }

    public int searchInsert(int[] nums, int target) {//35. 搜索插入位置
        int index= Arrays.binarySearch(nums,target);
        if(index<0)return -index-1;
        else return index;
    }

    public boolean isValidSudoku(char[][] board) {//36 有效的数独
        for(int i=0;i<9;i++){
            int [] bitRow = new int[9];
            int [] bitCol = new int[9];
            int [] bitRange =new int[9];
            for(int j=0;j<9;j++){
                if(board[i][j]!='.'){
                    if(bitRow[board[i][j] - '1']==1){
                        return false;
                    }
                    else{
                        bitRow[board[i][j] -'1']=1;
                    }
                }
                if(board[j][i]!='.'){
                    if(bitCol[board[j][i] - '1']==1){
                        return false;
                    }
                    else{
                        bitCol[board[j][i] -'1']=1;
                    }
                }
                int rowIndex = 3 * (i/3) + j/3;
                int colIndex = 3 * (i%3) + j%3;
                if(board[rowIndex][colIndex]!='.'){
                    if(bitRange[board[rowIndex][colIndex] -'1']==1){
                        return false;
                    }
                    else{
                        bitRange[board[rowIndex][colIndex] -'1']=1;
                    }
                }
            }
        }
        return true;
    }

    public String countAndSay(int n) {//38. 报数
        String array []=new String [31];
        array[0]="1";
        for(int w=1;w<30;w++){
            String s=array[w-1];
            StringBuffer str=new StringBuffer();
            int index=0;
            int count=1;
            for(int i=0;i<s.length();i=i+count){
                while(true){
                    if(index+i<s.length()){
                        if(s.charAt(i)==s.charAt(index+i)){
                            index++;
                        }
                        else{
                            str.append(String.valueOf(index));
                            str.append(s.charAt(i));
                            count=index;
                            index=0;
                            break;
                        }
                    }
                    else{
                        str.append(String.valueOf(index));
                        str.append(s.charAt(i));
                        count=index;
                        index=0;
                        break;
                    }

                }
            }
            array[w]=str.toString();
        }
        return array[n-1];
    }

    public List<List<Integer>> combinationSum(int[] candidates, int target) {//39. 组合总和
        List<Integer> result=new ArrayList<Integer>();
        List<List<Integer>> resultList=new ArrayList<List<Integer>>();
        Arrays.sort(candidates);
        dfsCombinationSum(candidates,result,resultList,0,target);
        return resultList;
    }
    private void dfsCombinationSum(int[] candidates, List<Integer> result, List<List<Integer>> resultList, int index,
                                   int target) {
        // TODO Auto-generated method stub
        if(target<0)return;
        if(target==0){
            resultList.add(new ArrayList<>(result));
        }
        else{
            for(int i=index;i<candidates.length;i++){
                result.add(candidates[i]);
                dfsCombinationSum(candidates, result, resultList, i, target-candidates[i]);
                result.remove(result.size()-1);
            }
        }

    }

    public List<List<Integer>> combinationSum2(int[] candidates, int target) {//40. 组合总和II
        List<Integer> result=new ArrayList<>();
        List<List<Integer>> resultList=new ArrayList<>();
        Arrays.sort(candidates);
        dfsCombinationSum2(candidates,result,resultList,0,target);
        HashSet h = new HashSet(resultList);
        resultList.clear();
        resultList.addAll(h);
        return resultList;
    }
    private void dfsCombinationSum2(int[] candidates, List<Integer> result, List<List<Integer>> resultList, int index,
                                    int target) {
        // TODO Auto-generated method stub
        if(target<0)return;
        if(target==0){
            //指向一块新的不变的地址否则resultList会随着result改变而改变
            resultList.add(new ArrayList<>(result));
        }
        else{
            for(int i=index;i<candidates.length;i++){
                if (target < candidates[i]) {
                    return;
                }
                result.add(candidates[i]);
                dfsCombinationSum2(candidates, result, resultList, i+1, target-candidates[i]);
                result.remove(result.size()-1);
            }
        }

    }

}
