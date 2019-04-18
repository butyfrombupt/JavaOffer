package leetcode.Twenty;

import java.util.*;

public class Main {
    public class ListNode {
        int val;
        ListNode next;
        ListNode(int x) { val = x; }
    }
    public static void main(String[] args) {


    }

    public int maxArea(int[] height) {//11 盛最多水的容器
        int max=0;
        for(int i=0;i<height.length;i++){
            for(int j=i;j<height.length;j++){
                if((j-i)*Math.min(height[i], height[j])>max){
                    max=(j-i)*Math.min(height[i], height[j]);
                }
            }
        }
        return max;
    }
    public String intToRoman(int num) {//12 整数转罗马数字
        String ans="";
        String oneArray[]=new String []{"","I","II","III","IV","V","VI","VII","VIII","IX"};
        String twoArray[]=new String []{"","X","XX","XXX","XL","L","LX","LXX","LXXX","XC"};
        String threeArray[]=new String []{"","C","CC","CCC","CD","D","DC","DCC","DCCC","CM"};
        String fourArray[]=new String []{"","M","MM","MMM"};
        int four=num/1000%10;
        int three=num/100%10;
        int two=num/10%10;
        int one=num%10;
        ans=fourArray[four]+threeArray[three]+twoArray[two]+oneArray[one];
        return ans;
    }
    public int romanToInt(String s) {//13 罗马数字转整数
        int temp=0;
        Map<String, Integer> RomanMap=new HashMap();
        RomanMap.put("I", 1);
        RomanMap.put("V", 5);
        RomanMap.put("X", 10);
        RomanMap.put("L", 50);
        RomanMap.put("C", 100);
        RomanMap.put("D", 500);
        RomanMap.put("M", 1000);
        for(int i=0;i<s.length();i++){
            if (i<s.length()-1 && RomanMap.get(String.valueOf(s.charAt(i)))<RomanMap.get(String.valueOf(s.charAt(i+1))))
                temp-=RomanMap.get(String.valueOf(s.charAt(i)));
            else
                temp+=RomanMap.get(String.valueOf(s.charAt(i)));
        }
        return temp;
    }
    public String longestCommonPrefix(String[] strs) {//14 最长公共前缀
        char ch;
        String prefix = "";
        if(strs==null||strs.length==0)
            return prefix;
        String base = strs[0];
        for (int i=0;i<base.length();i++) {
            char c = base.charAt(i);
            for (int j=1;j<strs.length;j++) {
                if(i>strs[j].length()-1)
                    return prefix;
                if(c!=strs[j].charAt(i))
                    return prefix;
            }
            prefix=prefix+c;
        }
        return prefix;
    }
    public List<List<Integer>> threeSum(int[] nums) {//15 三数之和
        List<List<Integer>> resultAns=new ArrayList<List<Integer>>();
        Arrays.sort(nums);
        for(int i=0;i<nums.length-2;i++){
            if(i!=0 && nums[i]==nums[i-1])
                continue;
            int sum=nums[i];
            int left=i+1;
            int right=nums.length-1;
            while(left<right){
                int sum1=nums[left]+nums[right];
                if(sum1==-sum){
                    List<Integer> result=new ArrayList<Integer>();
                    result.add(sum);
                    result.add(nums[left]);
                    result.add(nums[right]);
                    resultAns.add(result);
                    left++;
                }
                else if(sum1<-sum){
                    left++;
                }
                else{
                    right--;
                }
            }
        }
        HashSet h = new HashSet(resultAns);
        resultAns.clear();
        resultAns.addAll(h);
        return resultAns;
    }
    public int threeSumClosest(int[] nums, int target) {//16 最接近的三数之和
        Arrays.sort(nums);
        int min=9999999;
        int ans=0;
        for(int i=0;i<nums.length-2;i++){
            //去重
            if(i!=0 && nums[i]==nums[i-1])
                continue;
            int left=i+1;
            int right=nums.length-1;
            while(left<right){
                int sum=nums[left]+nums[right]+nums[i];
                if(Math.abs(sum-target)<min){
                    min=Math.abs(sum-target);
                    ans=sum;
                }
                else if(sum<target){
                    left++;
                }
                else{
                    right--;
                }
            }
        }
        return ans;
    }

    public List<String> letterCombinations(String digits) {//17 电话号码的字母组合
        List<String>list=new ArrayList<String>();
        if(digits.length()==0)return list;
        String[] a = new String[] {"","","abc","def",
                "ghi","jkl","mno","pqrs","tuv","wxyz"};
        StringBuffer sb = new StringBuffer();
        combine(digits,0,list,a,sb);
        return list;
    }
    private static void combine(String digits, int n, List<String> list, String[] a, StringBuffer sb) {
        // TODO Auto-generated method stub
        if(n==digits.length()){
            list.add(sb.toString());
            return;
        }
        for(int i=0;i<a[digits.charAt(n)-'0'].length();i++){
            sb.append(a[digits.charAt(n)-'0'].charAt(i));
            combine(digits, n+1, list, a, sb);
            sb.deleteCharAt(sb.length()-1);
        }
    }

    public List<List<Integer>> fourSum(int[] nums, int target) {//18 四数之和
        List<List<Integer>> resultAns=new ArrayList<List<Integer>>();
        Arrays.sort(nums);
        for(int i=0;i<nums.length;i++){
            //去重
            if(i!=0 && nums[i]==nums[i-1])
                continue;
            int sum=nums[i];
            int newTarget=target-sum;
            for(int q=i+1;q<nums.length;q++){

                int newNewTarget=newTarget-nums[q];
                int left=q+1;
                int right=nums.length-1;
                while(left<right){
                    int sum1=nums[left]+nums[right];
                    if(newNewTarget==nums[left]+nums[right]){
                        List<Integer> result=new ArrayList<Integer>();
                        result.add(sum);
                        result.add(nums[q]);
                        result.add(nums[left]);
                        result.add(nums[right]);
                        resultAns.add(result);
                        left++;
                    }
                    else if(sum1<newNewTarget){
                        left++;
                    }
                    else{
                        right--;
                    }
                }
            }

        }
        HashSet h = new HashSet(resultAns);
        resultAns.clear();
        resultAns.addAll(h);
        return resultAns;
    }

    public ListNode removeNthFromEnd(ListNode head, int n) {//19 删除链表的倒数第N个节点
        ListNode tempHead=new ListNode(0);
        tempHead.next=head;
        ListNode cur=tempHead;
        int length=0;
        while(cur.next!=null){
            cur=cur.next;
            length++;
        }
        cur=tempHead;
        int index=0;
        while(cur.next!=null){
            if(index==length-n){
                cur.next=cur.next.next;
            }
            else{
                cur=cur.next;
            }
            index++;
        }
        return tempHead.next;
    }

    public boolean isValid(String s) {//20 有效的括号
        Stack<Character> stack = new Stack<Character>();
        for(int i=0;i<s.length();i++){
            char haha=s.charAt(i);
            if(stack.size()==0){
                stack.push(haha);
            }
            else if(Matched(stack.peek(),haha)){
                stack.pop();
            }
            else{
                stack.push(haha);
            }
        }
        return stack.empty();
    }
    private static boolean Matched(char c1, char c2) {
        return (c1 == '(' && c2 == ')') || (c1 == '[' && c2 == ']') || (c1 == '{' && c2 == '}');
    }
}
