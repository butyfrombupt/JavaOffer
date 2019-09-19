package HandCode;

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

public class ThreadPrintNum {

    public static void main(String args[]){

        AtomicInteger num = new AtomicInteger(0);

        Lock lock = new ReentrantLock();

        Thread t1 = new Thread(new Runnable() {
            @Override
            public void run() {
                while (num.get() != 100){
                    if(num.get() % 2 == 0){
                        try {
                            lock.lock();
                            System.out.println("t1 print: "+num.incrementAndGet());
                        } catch (Exception e){
                            e.printStackTrace();
                        }
                        finally {
                            lock.unlock();
                        }
                    }

                }
            }
        });

        Thread t2 = new Thread(new Runnable() {
            @Override
            public void run() {
                while (num.get() != 100){
                    if(num.get() % 2 != 0){
                        try {
                            lock.lock();
                            System.out.println("t2 print: "+num.incrementAndGet());
                        } catch (Exception e){
                            e.printStackTrace();
                        }
                        finally {
                            lock.unlock();
                        }
                    }

                }
            }
        });
        t1.start();
        t2.start();
    }
}
