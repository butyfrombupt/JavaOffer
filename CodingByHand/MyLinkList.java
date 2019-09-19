package HandCode;

import java.util.Iterator;
import java.util.NoSuchElementException;

public class MyLinkList<E> {

    ListNode<E> first;

    ListNode<E> last;

    MyLinkList(){

    }

    private class ListNode<E> {
        E e;
        ListNode<E> prev;
        ListNode<E> next;

        ListNode(ListNode<E> prev,E e,ListNode<E> next){
            this.prev = prev;
            this.e = e;
            this.next = next;
        }
    }
    public void add(E e) {
        ListNode<E> l = last;
        ListNode<E> newNode = new ListNode<>(l, e, null);
        last = newNode;
        if (l == null)
            first = newNode;
        else
            l.next = newNode;
    }

    public E getLast() {
        ListNode<E> l = last;
        if (l == null)
            throw new NoSuchElementException();
        return l.e;
    }


    public Iterator<E> iterator(){
        return new myIterator();
    }

    public class myIterator implements Iterator<E>{


        private ListNode<E> newNode = first;

        @Override
        public boolean hasNext() {
            return newNode != last;
        }

        @Override
        public E next() {
            if (!hasNext())
                throw new NoSuchElementException();
            E nextItem = newNode.e;
            newNode = newNode.next;
            return nextItem;
        }
    }

    public void printLinkedList(Iterator<Integer> iterator){
        System.out.print("当前链表为: ");
        while (iterator.hasNext()){
            System.out.print(iterator.next() + " ");
        }
        System.out.println();
    }


}
