package HandCode;

class FooBar {
    private int n;

    private volatile boolean isRunning = true;

    public FooBar(int n) {
        this.n = n;
    }

    public synchronized void foo(Runnable printFoo) throws InterruptedException {


        for (int i = 0; i < n; i++) {
            // printFoo.run() outputs "foo". Do not change or remove this line.
            if(!isRunning ){
                this.wait();
            }
            printFoo.run();
            this.notify();
            isRunning = false;
        }
    }

    public synchronized void bar(Runnable printBar) throws InterruptedException {



        for (int i = 0; i < n; i++) {
            if(isRunning){
                this.wait();
            }
            // printBar.run() outputs "bar". Do not change or remove this line.
            printBar.run();
            this.notify();
            isRunning = true;
        }
    }
}
