package HandCode;

public class Test {

    public static void main(String args[]){
        FooBar fooBar = new FooBar(10);
        Thread a = new Thread(()->{
            try {
                fooBar.foo(()->{
                    System.out.print("Foo");
                });
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        });
        a.start();
        Thread b = new Thread(()->{
            try {
                fooBar.bar(()->{
                    System.out.print("Bar");
                });
            }
            catch (InterruptedException e){
                e.printStackTrace();
            }
        });
        b.start();
    }
}
