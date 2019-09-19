package HandCode;

public class DeadLock {

    public static Object o1 = new Object();
    public static Object o2 = new Object();

    public static void main(String[] args){
        Thread a = new Thread(new Lock1());
        Thread b = new Thread(new Lock2());
        a.start();
        b.start();
    }
    static class Lock1 implements Runnable{

        @Override
        public void run() {
            try {
                System.out.println("lock1 : "+Thread.currentThread());
                while (true){
                    synchronized (o1){
                        System.out.println("lock1 o1 locked!");
                        Thread.sleep(3000);
                        synchronized (o2){
                            System.out.println("lock1 o2 locked!");
                        }
                    }
                }
            } catch (Exception e){
                e.printStackTrace();
            }
        }
    }

    static class Lock2 implements Runnable{

        @Override
        public void run() {
            try {
                System.out.println("lock2 : "+Thread.currentThread());
                while (true){
                    synchronized (o2){
                        System.out.println("lock1 o2 locked!");
                        Thread.sleep(3000);
                        synchronized (o1){
                            System.out.println("lock1 o1 locked!");
                        }
                    }
                }
            } catch (Exception e){
                e.printStackTrace();
            }
        }
    }
}
