package HandCode;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantReadWriteLock;

public class Cache {

    static Map<String,Object> map = new HashMap<>();
    static ReentrantReadWriteLock rwl = new ReentrantReadWriteLock();
    static Lock read = rwl.readLock();
    static Lock write = rwl.writeLock();

    public static final Object get(String key){
        read.lock();
        try {
            return map.get(key);
        } finally {
            read.unlock();
        }
    }

    public static final Object put(String key, Object value){
        write.lock();
        try{
            return map.put(key,value);
        } finally {
            write.unlock();
        }
    }

    public static final void clear(){
        write.lock();
        try{
            map.clear();
        } finally {
            write.unlock();
        }
    }
}
