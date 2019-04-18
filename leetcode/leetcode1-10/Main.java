package leetcode;

import java.util.*;

public class Main {
    public class ListNode {
        int val;
        ListNode next;
        ListNode(int x) { val = x; }
    }
    public static void main(String[] args) {
        String s="abccddaabcde";

        System.out.println(lengthOfLongestSubstring(s));
    }
    public static int[] twoSum(int[] nums, int target) {// 1 两数之和
        for (int i=0;i<nums.length;i++){
            for (int j=i+1;j<nums.length;j++){
                if(nums[j]+nums[i]==target){
                    return new int[]{i,j};
                }
            }
        }
        throw new IllegalArgumentException("No two sum solution");
    }
    public static ListNode addTwoNumbers(ListNode l1, ListNode l2) {// 2 两数相加
        ListNode head=new ListNode(0);
        ListNode middle = head;

        int addFlag=0;
        int sum=0;
        while(l1!=null||l2!=null){
            int x=(l1!=null)?l1.val:0;
            int y=(l2!=null)?l2.val:0;
            sum=x+y+addFlag;
            addFlag=sum/10;
            middle.next=new ListNode(sum%10);
            middle=middle.next;
            if(l1!=null)l1=l1.next;
            if(l2!=null)l2=l2.next;
        }
        if(addFlag>0){
            middle.next=new ListNode(1);
        }
        return head.next;
    }
    public static int lengthOfLongestSubstring(String s) { //3 无重复字符的最长子串
        StringBuffer sBuffer = new StringBuffer("");
        int max=0;
        for(int i=0;i<s.length();i++){
            String str=s.substring(i,i+1);
            if(sBuffer.indexOf(str)==-1){
                sBuffer.append(str);
            }
            else{
                sBuffer=new StringBuffer(sBuffer.substring(sBuffer.indexOf(str)+1));
                sBuffer.append(str);
            }
            if(sBuffer.length()>max){
                max=sBuffer.length();
                //System.out.println("此时最长子串是"+sBuffer);
            }
        }
        return max;
    }
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {//4 寻找两个有序数组的中位数
        int len1=nums1.length;
        int len2=nums2.length;
        int totalLen=len1+len2;
        int a[]=new int [totalLen];
        len1--;len2--;totalLen--;
        while(len1>=0||len2>=0){
            if(len1<0){
                a[totalLen]=nums2[len2];
                len2--;
            }
            else if(len2<0){
                a[totalLen]=nums1[len1];
                len1--;
            }
            else{
                if(nums1[len1]>=nums2[len2]){
                    a[totalLen]=nums1[len1];
                    len1--;
                }
                else{
                    a[totalLen]=nums2[len2];
                    len2--;
                }
            }
            totalLen--;
        }
        double middleSum;
        if(a.length%2==0){
            middleSum=(a[a.length/2]+a[a.length/2-1])/2.00000;
            return middleSum;
        }
        else{
            return a[(a.length-1)/2];
        }
    }

    public String longestPalindrome(String s) {//5 最长回文子串
        int max=0;
        String ansStr="";
        for(int i=0;i<s.length();i++){
            for(int j=i;j<s.length();j++){
                String tempStr=s.substring(i,j+1);
                if(isHuiwenString(tempStr)&&tempStr.length()>max){
                    max=tempStr.length();
                    ansStr=tempStr;
                }
            }
        }
        return ansStr;
    }
    public static boolean isHuiwenString(String s){
        String s1=new StringBuffer(s).reverse().toString();
        return s.equals(s1);
    }

    public String convert(String s, int numRows) {//6 Z字形变换
        if(numRows <= 1) return s;
        StringBuilder res = new StringBuilder();
        int mul = (numRows - 1) * 2;
        for(int i = 0;i < numRows;i++){
            int m = 2 * i,j = i;
            while(j < s.length()){
                res.append(s.charAt(j));
                m = mul - m;
                if(m == 0) m = mul;
                j += m;
            }
        }
        return res.toString();
    }
    public int reverse(int x) {//7 整数反转
        try {
            if(x>=0){
                String s=String.valueOf(x);
                String test="";
                for(int i=0;i<s.length();i++){
                    char temp=s.charAt(i);
                    if(i==s.length()-1&&temp=='0'){
                        test=""+test;
                    }
                    else
                        test=temp+test;
                }
                return Integer.parseInt(test);
            }
            else{
                String s=String.valueOf(x);
                String test="";
                for(int i=1;i<s.length();i++){
                    char temp=s.charAt(i);
                    if(i==s.length()-1&&temp=='0'){
                        test=""+test;
                    }
                    else
                        test=temp+test;
                }
                return -Integer.parseInt(test);
            }
        }catch(Exception e){
            return 0;
        }
    }

    public int myAtoi(String str) {//8 字符串转换整数 (atoi)
        if (str == null)
            return 0;
        str = str.trim();
        char[] chars = str.toCharArray();
        if (chars.length == 0)
            return 0;
        char c0 = chars[0];
        if (c0 != '-' && c0 != '+' && !Character.isDigit(c0))
            return 0;
        if (c0 == '-' || c0 == '+') {
            if (chars.length == 1)
                return 0;
            else if (!Character.isDigit(chars[1]))
                return 0;
        }
        StringBuilder sb = new StringBuilder();
        sb.append(c0);
        int i = 1;
        while (i < chars.length){
            if (!Character.isDigit(chars[i]))
                break;
            sb.append(chars[i]);
            i++;
        }
        int res = 0;
        try {
            res = Integer.parseInt(sb.toString());
        } catch (Exception e) {
            if (c0 == '-')
                return Integer.MIN_VALUE;
            else
                return Integer.MAX_VALUE;
        }
        return res;
    }

    public boolean isPalindrome(int x) {//9 回文数
        try{
            String s=new StringBuffer(String.valueOf(x)).reverse().toString();
            return s.equals(String.valueOf(x));
        }catch(Exception e){
            return false;
        }
    }
}
