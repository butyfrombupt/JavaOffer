package HandCode;

public interface BtyMap<K,V> {

    public V put(K k,V v);

    public V get(K k);

    interface Entry<K,V>{
        public K getKey();

        public V getValue();
    }

}
