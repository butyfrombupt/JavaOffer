package leetcode;

import java.util.*;

public class Main {

    public static void main(String[] args) {

    }
    public int[] twoSum(int[] nums, int target) {// 1 两数之和
        for (int i=0;i<nums.length;i++){
            for (int j=i+1;j<nums.length;j++){
                if(nums[j]+nums[i]==target){
                    return new int[]{i,j};
                }
            }
        }
        throw new IllegalArgumentException("No two sum solution");
    }
}
