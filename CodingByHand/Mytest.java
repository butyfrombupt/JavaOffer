package HandCode;

import java.util.Iterator;

public class Mytest {
    public static void main(String[] args) {
        //进行添加操作
//        MyLinkList<String> myLinkedList = new MyLinkList<>();
//        myLinkedList.add("Java");
//        myLinkedList.add("C++");
//        myLinkedList.add("Python");
//        myLinkedList.add( "PHP");
//        //遍历结果
//
//        printLinkedList(myLinkedList.iterator());

    }

    private static void printLinkedList(Iterator<String> iterator) {
        System.out.print("当前链表为: ");
        while (iterator.hasNext()){
            System.out.print(iterator.next() + " ");
        }
        System.out.println();
    }
}
