package HandCode;

public class ButyHashMap<K,V> implements BtyMap<K, V>{

    private Entry<K, V>[] arr = null;

    int size;

    public ButyHashMap() {

        arr = new Entry[16];

    }

    public int size() {

        return this.size;

    }

    class Entry<K,V> implements BtyMap.Entry<K, V>{

        private K key;

        private V value;

        private Entry<K, V> next;

        public Entry(){

        }

        public Entry(K key, V value,Entry<K, V> next) {
            this.key = key;
            this.value = value;
            this.next=next;
        }

        @Override
        public K getKey(){return key;}

        @Override
        public V getValue(){return value;}

    }

    @Override
    public V put(K key, V value) {
        V oldValue= null;
        int index = key.hashCode() % arr.length;
        if (arr[index] == null) {
            arr[index] = new Entry<K,V>(key, value,null);
            size++;
        } else {
            Entry<K, V> entry=arr[index];
            Entry<K, V> e = entry;
            while(e != null){
                if(key == e.getKey()||key.equals(e.getValue())){
                    oldValue = e.value;
                    e.value = value;
                    return oldValue;
                }
                e = e.next;
            }
            arr[index] = new Entry(key, value, entry);
            size++;
        }
        return oldValue;
    }

    @Override
    public V get(K key){
        int index=key.hashCode() % arr.length;

        if(arr[index]==null)
            return null;
        else{
            Entry<K, V> entry = arr[index];
            while(entry!=null){
                if(key == entry.getKey()||key.equals(entry.getKey())){
                    return entry.value;
                }

                entry=entry.next;
            }
        }
        return null;
    }
}
