package HandCode;

import java.util.Random;
import java.util.concurrent.BlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

public class Producer implements Runnable {

    private BlockingQueue<Integer> queue;

    private static final int SLEEPTIME = 10000;

    private static volatile boolean isRunning = true;

    private static AtomicInteger count = new AtomicInteger();

    public Producer(BlockingQueue<Integer> queue){
        this.queue = queue;
    }

    @Override
    public void run() {
        int data;

        Random r = new Random();

        System.out.println("start producer id = " + Thread.currentThread().getId());

        try {
            while (isRunning){

                Thread.sleep(r.nextInt(SLEEPTIME));

                data = count.incrementAndGet();
                System.out.println("producer " + Thread.currentThread().getId() + " create data：" + data
                        + ", size：" + queue.size());
                if (!queue.offer(data, 2, TimeUnit.SECONDS)) {
                    System.err.println("failed to put data：" + data);
                }
            }
        }catch (InterruptedException e){
            e.printStackTrace();
            Thread.currentThread().interrupt();
        }
    }

    public void stop() {
        isRunning = false;
    }
}
