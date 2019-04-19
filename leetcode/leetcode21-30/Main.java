package leetcode.Thirty;

import java.util.*;
/**
 * Created by wsbty on 2019/4/19.
 */
public class Main {

    public class ListNode {
        int val;
        ListNode next;
        ListNode(int x) { val = x; }
    }

    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {// 合并两个有序链表
        ListNode head=new ListNode(0);
        ListNode cur=head;
        if(l1==null)cur.next=l2;
        if(l2==null)cur.next=l1;
        while(l1!=null&&l2!=null){
            if(l1.val<=l2.val){
                cur.next=l1;
                cur=cur.next;
                l1=l1.next;
            }
            else{
                cur.next=l2;
                cur=cur.next;
                l2=l2.next;
            }
            if (l1 == null) {
                cur.next = l2;
            } else {
                cur.next = l1;
            }
        }
        return head.next;
    }

    public List<String> generateParenthesis(int n) {//22 括号生成
        List<String>list=new ArrayList<String>();
        if(n==0)return list;
        generateAll(new char[2 * n], 0, list);
        return list;
    }
    private static void generateAll(char[] current, int pos, List<String> list) {
        // TODO Auto-generated method stub
        if (pos == current.length) {
            if (isValid1(new String(current)))
                list.add(new String(current));
        } else {
            current[pos] = '(';
            generateAll(current, pos+1, list);
            current[pos] = ')';
            generateAll(current, pos+1, list);
        }
    }
    public static boolean isValid1(String s) {//20有效括号
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

    public ListNode swapPairs(ListNode head) {//24 两两交换链表中的节点
        ListNode dummyHead=new ListNode(0);
        if(head==null)return head;
        if(head.next==null)//如果只有一个节点，直接返回这个节点
            return head;
        dummyHead.next=head;
        ListNode fast=head;
        ListNode cur=dummyHead;
        while(fast!=null&&fast.next!=null){
            ListNode temp=cur.next;
            cur.next=fast.next;
            fast.next=temp;
            fast=fast.next;
            cur=cur.next;
        }
        return dummyHead.next;
    }

    public int removeDuplicates(int[] nums) {//26 删除排序数组中的重复项
        //{0,0,1,1,1,2,2,3,3,4};
        if(nums.length==0)return 0;
        int index=0;
        for(int i=1;i<nums.length;i++){
            if(nums[index]!=nums[i]){
                index++;
                nums[index]=nums[i];
            }
        }
        return index+1;
    }

    public int removeElement(int[] nums, int val) {//27 移除元素
        if(nums.length==0)return 0;
        int index=0;
        for(int i=0;i<nums.length;i++){
            if(nums[i]!=val){
                nums[index]=nums[i];
                index++;
            }
        }
        return index;
    }

}
