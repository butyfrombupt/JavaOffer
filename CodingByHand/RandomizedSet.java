package HandCode;

import java.util.HashMap;
import java.util.Random;

public class RandomizedSet {

    HashMap<Integer,Integer> valueMap;
    HashMap<Integer,Integer> indexMap;

    public RandomizedSet() {
        valueMap = new HashMap<>();
        indexMap = new HashMap<>();
    }

    /** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
    public boolean insert(int val) {
        if(valueMap.containsKey(val)){
            return false;
        }
        valueMap.put(val,valueMap.size());
        indexMap.put(indexMap.size(),val);

        return true;
    }

    /** Removes a value from the set. Returns true if the set contained the specified element. */
    public boolean remove(int val) {
        if(!valueMap.containsKey(val)){
            return false;
        }
        int removeIndex = valueMap.get(val);
        valueMap.remove(val);
        indexMap.remove(removeIndex);

        Integer lastElement = indexMap.get(indexMap.size());
        if(lastElement != null){
            indexMap.put(removeIndex,lastElement);
            valueMap.put(lastElement,removeIndex);
        }

        return true;
    }

    /** Get a random element from the set. */
    public int getRandom() {
        if(valueMap.size()==0){
            return -1;
        }

        if(valueMap.size()==1){
            return indexMap.get(0);
        }

        Random r = new Random();
        int index = r.nextInt(valueMap.size());

        return indexMap.get(index);
    }
}
