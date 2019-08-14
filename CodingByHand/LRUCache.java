package HandCode;

import java.util.LinkedHashMap;
import java.util.Map;

class LRUCache {

    private LinkedHashMap<Integer, Integer> map;
    private int size;
    private int oldestKey = 0;

    public LRUCache(int capacity) {
        this.size = capacity;
        map = new LinkedHashMap<Integer, Integer>(capacity,1,true) {
            @Override
            protected boolean removeEldestEntry(Map.Entry<Integer, Integer> eldest) {
                boolean isRemove = false;
                if (this.size() > size) {
                    isRemove = true;
                    oldestKey = eldest.getKey();
                }
                return isRemove;
            }
        };
    }

    public int get(int key) {
        if (map.get(key) != null) {
            return map.get(key);
        } else return -1;
    }

    public void put(int key, int value) {
        if (map.size() > size) {
            map.remove(oldestKey);
        }
        map.put(key, value);
    }
}