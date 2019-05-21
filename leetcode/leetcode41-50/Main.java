package leetcode.Fifty;

import java.util.*;

/**
 * Created by wsbty on 2019/4/20.
 */

public class Main {
    /*
    模拟 1234 * 25 的过程

    value   0    1    2    3    4    5
    num1              1    2    3    4
    num2                        2    5
                      5    10   15   20
                 2    4    6    8
    处理进位前   2    9    16   23   20
    处理进位后   3    0    8    5    0

     */
    public String multiply(String num1, String num2) {//43 字符串相乘
        int len1 = num1.length() - 1;
        int len2 = num2.length() - 1;
        if (len1 < 0 || len2 < 0) return "";
        int value[] = new int[len1 + len2 + 2];
        for (int i = len1; i >= 0; i--) {
            for (int j = len2; j >= 0; j--) {
                value[i + j + 1] += (num1.charAt(i) - '0') * (num2.charAt(j) - '0');
            }
        }
        //处理进位
        int add = 0;
        for (int i = value.length - 1; i >= 0; i--) {
            value[i] = value[i] + add;
            add = value[i] / 10;
            value[i] = value[i] % 10;
        }
        //处理前端的0
        StringBuilder sb = new StringBuilder();
        int index = 0;
        while (index < value.length - 1 && value[index] == 0) {
            index++;
        }
        for (int i = index; i < value.length; i++) {
            sb.append(value[i]);
        }
        return sb.toString();
    }

        /*
        给定一个非负整数数组，你最初位于数组的第一个位置。

        数组中的每个元素代表你在该位置可以跳跃的最大长度。

        你的目标是使用最少的跳跃次数到达数组的最后一个位置。

        示例:

        输入: [2,3,1,1,4]
        输出: 2
         */
        public int jump(int[] nums) {//45. 跳跃游戏 II
            if(nums.length == 1) return 0;
            int maxreach=0;//之前能到达的最远位置
            int cur=0;
            int count=0;
            while(cur<nums.length-1){
                count++;
                maxreach=cur;
                for(int i=0;i<=maxreach;i++){
                    cur=Math.max(cur, i+nums[i]);
                }
            }
            return count;
    }
    public List<List<Integer>> permute(int[] nums) {//46全排列
        List<List<Integer>> resultList=new ArrayList<>();
        int len = nums.length;
        permuteOne(nums,0,len,resultList);
        return resultList;
    }
    private static void permuteOne(int[] nums, int start, int len, List<List<Integer>> resultList) {
        if(start==len-1){
            List<Integer> result=new ArrayList<>();
            for(int j=0;j<len;j++){
                result.add(nums[j]);
            }
            resultList.add(result);
            return;
        }
        for(int i=start;i<len;i++){
            swap(nums,start,i);
            permuteOne(nums, start+1,len, resultList);
            swap(nums,start,i);
        }
    }
    public static void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    public List<List<Integer>> permuteUnique(int[] nums) {//47全排列II
        List<List<Integer>> resultList=new ArrayList<>();
        int len = nums.length;
        Arrays.sort(nums);

        permuteOne(nums,0,len,resultList);

        return resultList;
    }
    private static void permuteOne1(int[] nums, int start, int len, List<List<Integer>> resultList) {
        if(start==len-1){
            List<Integer> result=new ArrayList<>();
            for(int j=0;j<len;j++){
                result.add(nums[j]);
            }
            resultList.add(result);
            return;
        }
        for(int i=start;i<len;i++){
            if (isSame(nums, start, i)) {
                swap(nums,start,i);
                permuteOne(nums, start+1,len, resultList);
                swap(nums,start,i);
            }
        }
    }
    public static boolean isSame(int[] nums, int begin, int end){
        for (int i = begin; i < end; i++)
            if (nums[i] == nums[end])
                return false;
        return true;
    }

    /*
    先对角线交换，然后左右互换
     */
    public void rotate(int[][] matrix) {//48. 旋转图像
        int len=matrix.length;
        for (int i = 0; i < len; ++i) {
            for (int j = i + 1; j < len; ++j) {
                swap(matrix,i, j);
            }
        }
        int left=0;
        int right=len-1;
        while(left<right){
            for(int index=0;index<len;index++){
                int temp=matrix[index][left];
                matrix[index][left]=matrix[index][right];
                matrix[index][right]=temp;
            }
            left++;
            right--;
        }
    }
    private static void swap(int[][] matrix, int i, int j) {
        // TODO Auto-generated method stub
        int temp=matrix[i][j];
        matrix[i][j]=matrix[j][i];
        matrix[j][i]=temp;
    }

    /*
    输入: ["eat", "tea", "tan", "ate", "nat", "bat"],
    输出:
    [
    ["ate","eat","tea"],
    ["nat","tan"],
    ["bat"]
    ]
     */
    public List<List<String>> groupAnagrams(String[] strs) {//49. 字母异位词分组
        if (strs.length == 0) return new ArrayList();
        Map<String, List> ans = new HashMap<String, List>();
        for(String s : strs){
            char a[]=s.toCharArray();
            Arrays.sort(a);
            String key=String.valueOf(a);
            if(!ans.containsKey(key)){
                ans.put(key, new ArrayList());
            }
            ans.get(key).add(s);
        }
        return new ArrayList(ans.values());
    }

    public double myPow(double x, int n) {//50. Pow(x, n)
        if(n<0)
            return 1/pow(x,-n);
        else
            return	pow(x,n);
    }
    private static double pow(double x, int n) {
        // TODO Auto-generated method stub
        if(n == 0)
            return 1;
        double v = pow(x, n / 2);
        if(n % 2 == 0) {
            return v * v;
        } else {
            return v * v * x;
        }

    }

}
