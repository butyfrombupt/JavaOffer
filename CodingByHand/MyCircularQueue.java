package HandCode;

public class MyCircularQueue {

    public static void main(String args[]){
        MyCircularQueue queue = new MyCircularQueue(3);
        queue.enQueue(1);
        queue.enQueue(2);
        System.out.println(queue.Front());
        System.out.println(queue.Rear());
        queue.deQueue();
        queue.deQueue();
        System.out.println(queue.Front());
        System.out.println(queue.Rear());
    }

    private int data[];
    private int size;
    private int head;
    private int rear;

    public MyCircularQueue(int k) {
        size = 0;
        data = new int[k];
        head = -1;
        rear = -1;
    }

    /** Insert an element into the circular queue. Return true if the operation is successful. */
    public boolean enQueue(int value) {
        if(size == data.length){
            return false;
        }
        rear = (rear + 1) % data.length;
        data[rear] = value;
        if(size == 0){
            head = rear;
        }
        size++;
        return true;
    }

    /** Delete an element from the circular queue. Return true if the operation is successful. */
    public boolean deQueue() {
        if(size == 0){
            return false;
        }
        head = (head + 1) % data.length;
        size--;
        return true;
    }

    /** Get the front item from the queue. */
    public int Front() {
        if(size == 0){
            return -1;
        }
        return data[head];
    }

    /** Get the last item from the queue. */
    public int Rear() {
        if(size == 0){
            return -1;
        }
        return data[rear];
    }

    /** Checks whether the circular queue is empty or not. */
    public boolean isEmpty() {
        return size == 0;
    }

    /** Checks whether the circular queue is full or not. */
    public boolean isFull() {
        return size == data.length;
    }
}
