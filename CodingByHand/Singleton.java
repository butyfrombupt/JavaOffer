package HandCode;

public class Singleton {


    private static volatile Singleton instance;//防止指令重排，
    // 1 分配对象内存空间 2 初始化对象 3 把对象指向刚才的内存空间（此时 instance != null），
    // 1 3 2 执行的话 第二个线程可能return一个没有初始化的对象

    Singleton(){

    }
    public static Singleton getInstance(){
        if(instance == null){//减少开销，减少lock
            synchronized (Singleton.class){
                if(instance == null){//有可能会有两个线程走到这个代码块，只不过一个加锁另一个等罢了
                    instance = new Singleton();
                }
            }
        }
        return instance;
    }
}
