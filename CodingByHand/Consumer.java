package HandCode;

import java.util.Random;
import java.util.concurrent.BlockingQueue;

public class Consumer implements Runnable{

    private BlockingQueue<Integer> queue;

    private static final int SLEEPTIME = 1000;

    private static volatile boolean isRunning = true;

    public Consumer(BlockingQueue<Integer> queue){
        this.queue = queue;
    }


    @Override
    public void run() {

        int data;

        Random r = new Random();

        System.out.println("start consumer id = " + Thread.currentThread().getId());

        try {
            while (isRunning){

                Thread.sleep(r.nextInt(SLEEPTIME));

                if(!queue.isEmpty()){
                    data = queue.take();
                    System.out.println("consumer " + Thread.currentThread().getId() + " consumer data：" + data
                            + ", size：" + queue.size());
                }
                else{
                    System.out.println("Queue is empty, consumer " + Thread.currentThread().getId()
                            + " is waiting, size：" + queue.size());
                }
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
            Thread.currentThread().interrupt();
        }

    }

    public void stop() {
        isRunning = false;
    }
}
