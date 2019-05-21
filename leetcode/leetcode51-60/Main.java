package leetcode.sixty;
import java.util.*;
/**
 * Created by wsbty on 2019/5/21.
 */
public class Main {

    public List<List<String>> solveNQueens(int n) {//51.N皇后
        List<List<String>> result = new ArrayList<List<String>>();
        queens(result, n, 1, new String[n],
                new int[n + 1], new int[2 * n + 1], new int[2 * n + 1]);
        return result;
    }

    /**
     * 从1行开始，逐行扫描
     * 同行判断：单调递增，不作处理
     * 同列判断：j列未被占用，colums[i]=1，表示j列已被占用
     * 左下判断：i+j未被占用，位置坐标特点：x + y恒定
     * 右下判断：i-j+n未被占用，位置坐标特点：x - y恒定
     */
    void queens(List<List<String>> result, int n, int i, String[] lines,
                int[] columns, int[] lDown, int[] rDown) {
        if (i > n) {
            // 数据复制，防止篡改
            result.add(new ArrayList<String>(Arrays.asList(lines)) );
            return;
        }
        for (int j = 1; j <= n; j++) {
            if (columns[j] == 0 && lDown[i + j] == 0 && rDown[i - j + n] == 0) {
                // 设置占用，i行j列
                char[] line = new char[n];
                Arrays.fill(line, '.');
                line[j - 1] = 'Q';
                lines[i - 1] = new String(line);
                // 同列设置、左下设置、右下设置
                columns[j] = lDown[i + j] = rDown[i - j + n] = 1;
                // 继续i+1行，逐行扫描
                queens(result, n, i + 1, lines, columns, lDown, rDown);
                // 数据清理，结构复用
                columns[j] = lDown[i + j] = rDown[i - j + n] = 0;
            }
        }
    }
/*
输入: [-2,1,-3,4,-1,2,1,-5,4],
输出: 6
解释: 连续子数组 [4,-1,2,1] 的和最大，为 6。
 */
    public int maxSubArray(int[] nums) {//53. 最大子序和
        int maxSum=nums[0];
        int dp[] =new int [100000];
        dp[0]=nums[0];
        for(int i=1;i<nums.length;i++){
            dp[i]=Math.max(dp[i-1]+nums[i], nums[i]);
        }
        for(int k=0;k<nums.length;k++){
            if(dp[k]>maxSum){
                maxSum=dp[k];
            }
        }
        return maxSum;
    }
    /*
   输入:
[
[ 1, 2, 3 ],
[ 4, 5, 6 ],
[ 7, 8, 9 ]
]
输出: [1,2,3,6,9,8,7,4,5]
 */
    public List<Integer> spiralOrder(int[][] matrix) {//54. 螺旋矩阵
        List<Integer> result =new ArrayList<>();
        if(matrix.length==0)
            return result;
        int row=matrix.length-1;
        int col=matrix[0].length-1;
        int i=0;int j=0;
        while(i<row&&j<col){
            pushCircle(matrix,i,j,result,row,col);
            i++;
            j++;
            row--;
            col--;
        }
        if (i == row) {
            for (int n = i; n<= col; n++)
                result.add(matrix[i][n]);
        } else if (j == col) {
            for (int n = j; n<= row; n++)
                result.add(matrix[n][j]);
        }
        return result;
    }

    private void pushCircle(int[][] matrix, int iStart, int jStart, List<Integer> result,int row,int col) {
        // TODO Auto-generated method stub
        for(int i=iStart;i<=col;i++){
            result.add(matrix[iStart][i]);
        }
        for(int j=jStart+1;j<=row;j++){
            result.add(matrix[j][col]);
        }
        for(int i=col-1;i>=iStart;i--){
            result.add(matrix[row][i]);
        }
        for(int j=row-1;j>jStart;j--){
            result.add(matrix[j][iStart]);
        }

    }
/*
输入: [2,3,1,1,4]
输出: true
解释: 从位置 0 到 1 跳 1 步, 然后跳 3 步到达最后一个位置。
示例 2:

输入: [3,2,1,0,4]
输出: false
解释: 无论怎样，你总会到达索引为 3 的位置。但该位置的最大跳跃长度是 0 ， 所以你永远不可能到达最后一个位置。
 */
    public boolean canJump(int[] nums) {//55. 跳跃游戏
        int len=nums.length-1;
        if(len==0)return true;
        List<Integer> zero = new ArrayList<Integer>();
        for(int i=len;i>=0;i--){
            if(nums[i]==0)zero.add(i);
        }
        if(zero.size()==0)return true;
        boolean flag = true;
        for(int zeroPosition: zero){
            flag=false;
            int index=zeroPosition;
            while(index>=0){
                if((nums[index]>(zeroPosition-index))||nums[index]>=(zeroPosition-index)&&zeroPosition==len){
                    flag=true;
                    break;
                }
                index--;
            }
            if(flag==false)break;
        }
        return flag;
    }
/*
输入: [[1,3],[2,6],[8,10],[15,18]]
输出: [[1,6],[8,10],[15,18]]
解释: 区间 [1,3] 和 [2,6] 重叠, 将它们合并为 [1,6].
 */
    public List<Interval> merge(List<Interval> intervals) {//56.合并区间
        List<Interval>res=new ArrayList<Interval>();
        Collections.sort(intervals, (a, b) -> (a.start - b.start));
        for(Interval list : intervals){
            if(res.isEmpty())res.add(list);
            else if(list.start<=res.get(res.size()-1).end){
                res.get(res.size()-1).end=Math.max(list.end, res.get(res.size()-1).end);
            }
            else{
                res.add(list);
            }
        }
        return res;
    }
/*
输入: "Hello World"
输出: 5
*/
    public int lengthOfLastWord(String s) {//58. 最后一个单词的长度
        String [] arr = s.split("\\s+");
        if(arr.length==0)
            return 0;
        else
            return arr[arr.length-1].length();
    }

    public static int[][] generateMatrix(int n) {//59. 螺旋矩阵 II
        int[][] result=new int[n][n];
        for(int i=0;i<n;i++){
            int[] row=new int[n];
            for(int j=0;j<n;j++){
                row[j]=0;
            }
            result[i]=row;
        }
        int i=0;
        int di=0;
        int j=0;
        int dj=1;
        for(int k=1;k<n*n+1;k++){
            result[i][j]=k;
            if(result[(i+n+di)%n][(j+n+dj)%n]!=0){
                int temp=di;
                di=dj;
                dj=-temp;
            }
            i+=di;
            j+=dj;
        }
        return result;
    }
    public String getPermutation(int n, int k) {//60. 第k个排列
        List<String> resultList=new ArrayList<>();
        int nums[]=new int[n];
        for(int i=0;i<n;i++){
            nums[i]=i+1;
        }
        int len = nums.length;
        //Permutation(nums,0,len,resultList);
        Collections.sort(resultList,new Comparator<String>() {
            @Override
            public int compare(String o1, String o2) {
                if(o1 == null || o2 == null){
                    return -1;
                }
                if(o1.length() > o2.length()){
                    return 1;
                }
                if(o1.length() < o2.length()){
                    return -1;
                }
                if(o1.compareTo(o2) > 0){
                    return 1;
                }
                if(o1.compareTo(o2) < 0){
                    return -1;
                }
                if(o1.compareTo(o2) == 0){
                    return 0;
                }
                return 0;
            }
        });
        String res=resultList.get(k-1);
        return res;
    }
}
