package HandCode;

import java.util.*;
public class StackArrayQueue {

    public static void main(String[] args) {
        //Integer.hashCode();
    }


    /*
    栈实现队列
    stack2不为空，stack2的栈顶元素就是最先进入队列的元素
    stack2为空时，把stack1的元素逐个弹出压入stack2
     */
    Stack<Integer> s1;//用于入队
    Stack<Integer> s2;//用于出队
    /** Initialize your data structure here. */
    public StackArrayQueue(int i) {
        s1 = new Stack<Integer>();
        s2 = new Stack<Integer>();
    }

    /** Push element x to the back of queue. */
    public void push(int x) {
        s1.push(x);
    }

    /** Removes the element from in front of queue and returns that element. */
    public int pop() {
        if(!s2.isEmpty()){//出队栈不为空时，直接从出队栈中移除栈顶元素
            return s2.pop();
        }else{//出队栈为空时，从入队栈中依次将元素放入出队栈
            while(!s1.isEmpty()){
                s2.push(s1.pop());
            }
            return s2.pop();//放完后，从出队栈依次将栈顶元素弹出
        }
    }

    /** Get the front element. */
    public int peek() {
        if(!s2.isEmpty()){//出队栈不为空时，直接从出队栈中得到栈顶元素
            return s2.peek();
        }else{//出队栈为空时，从入队栈中依次将栈顶元素放入出队栈
            while(!s1.isEmpty()){
                s2.push(s1.pop());
            }
            return s2.peek();//放完后，从出队栈得到栈顶元素
        }
    }

    /** Returns whether the queue is empty. */
    public boolean empty() {
        return s1.isEmpty() && s2.isEmpty();
    }


    /*
    用队列实现栈
    已 1 2 3 举例 top一直指向最新入队的，作为栈顶，push直接入队列q1
    pop操作保证最后留一个，剩的那个是栈顶，出队记得更新top，然后q1 和 q2 交换
     */
    private Queue<Integer> q1;
    private Queue<Integer> q2;
    private int top;

    public StackArrayQueue(char i) {
        q1 = new LinkedList<>();
        q2 = new LinkedList<>();
    }

    /** Push element x onto stack. */
    public void push1(int x) {
        q1.offer(x);
        top = x;
    }

    /** Removes the element on top of the stack and returns that element. */
    public int pop1() {
        while (q1.size() > 1){//最后留一个，那个就是相当于栈顶元素
            top = q1.poll();
            q2.offer(top);
        }
        Queue<Integer> temp = q1;
        q1 = q2;
        q2 = temp;
        return q2.poll();
    }

    /** Get the top element. */
    public int top1() {
        return top;
    }

    /** Returns whether the stack is empty. */
    public boolean empty1() {
        return q1.isEmpty() && q2.isEmpty();
    }


    /*
    用数组实现栈
     */
    public class MyStackByArray<E> {


        private Object[] butyStack;

        private int size;

        private int Capacity = 10;

        private MyStackByArray() {
            butyStack = new Object[Capacity];
        }

        private Boolean isEmpty() {
            return this.size == 0;
        }

        private E peek() {
            if (isEmpty()) {
                return null;
            }
            return (E) butyStack[size - 1];
        }

        public E pop() {
            E e = peek();
            butyStack[size - 1] = null;
            size--;
            return e;
        }


        private void push2(E item) {
            resize(size);
            butyStack[size++] = item;
        }

        private void resize(int size) {
            int length = butyStack.length;
            if (size > length) {
                int newLen = size + Capacity;
                butyStack = Arrays.copyOf(butyStack, newLen);
            }
        }
    }
}
