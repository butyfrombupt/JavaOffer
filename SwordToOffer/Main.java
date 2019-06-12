package SwordToOffer;

import java.util.*;


public class Main {
    public static void main(String[] args) {
//        int [] nums = new int[]{6,7};
//        //System.out.println(reConstructBinaryTree(nums,nums1));
//        TreeNode root = new TreeNode(1);
//        TreeNode root2 = new TreeNode(2);
//        TreeNode root3 = new TreeNode(3);
//        TreeNode root4 = new TreeNode(4);
//        TreeNode root5 = new TreeNode(5);
//        TreeNode root6 = new TreeNode(6);
//        root5.left = root3;
//        root3.left = root2;
//        root3.right = root4;
//        root2.left = root;
//        root5.right = root6;
//        System.out.println(KthNode(root5,3));
        int nums [] = new int[]{2,3,4,2,6,2,5,1};
        //System.out.println(FindNumsAppearOnce(nums,nums,nums));
        String s = "a";
        String s1=".";
        System.out.println(maxInWindows(nums,3));

    }



    public boolean Find(int target, int [][] array) {//二维数组中的查找

//        Method 1:
//        for(int i=0;i<array.length;i++){
//            int low=0;
//            int high=array[i].length-1;
//            while(low<=high){
//                int mid=(low+high)/2;
//                if(target>array[i][mid]){
//                    low=mid+1;
//                }
//                else if(target<array[i][mid]){
//                    high=mid-1;
//                }
//                else{
//                    return true;
//                }
//            }
//        }
//        return false;
//      Method 2
        int len=array.length-1;
        int index=0;
        while((len>=0)&&(index<=array[0].length-1)){
            if(array[len][index]>target){
                len--;
            }
            else if(target>array[len][index]){
                index++;
            }
            else {
                return true;
            }
        }
        return false;
    }
    public static String replaceSpace(StringBuffer str) {//替换空格
        String res="";
        char c[]=str.toString().toCharArray();
        for(int i=0;i<c.length;i++){
            if(c[i]==' '){
                res=res+"%20";
            }
            else{
                res=res+c[i];
            }
        }
        return res;
    }
    public static boolean duplicate(int numbers[], int length, int[] duplication) {//数组中重复的数字
        /*
        if(numbers == null || numbers.length == 0) return false;
        Arrays.sort(numbers);
        boolean flag=false;
        int index=0;
        for(int i=1;i<numbers.length;i++){
            if(numbers[i]==numbers[i-1]){
                flag=true;
                duplication[index++]=numbers[i];
            }
        }
        return flag;
        */
        if(numbers == null || numbers.length == 0) return false;
        for(int i=0;i<length;i++) {
            if (numbers[i] < 0 || numbers[i] > length - 1) {
                return false;
            }
        }
            for(int i=0;i<length;i++){
                while (numbers[i]!=i){
                    if(numbers[i]==numbers[numbers[i]]){
                        duplication[0]=numbers[i];
                        return true;
                    }
                    int temp=numbers[i];
                    numbers[i]=numbers[temp];
                    numbers[temp]=temp;
                }
            }
        return false;
    }

    public ArrayList<Integer> printListFromTailToHead(ListNode listNode) {//从尾到头打印链表
        Stack<Integer> stack = new Stack<>();
        while (listNode!=null){
            stack.push(listNode.val);
            listNode=listNode.next;
        }
        ArrayList<Integer> list = new ArrayList<>();
        while (!stack.isEmpty()) {
            list.add(stack.pop());
        }
        return list;

    }
    public TreeNode reConstructBinaryTree(int [] pre,int [] in) {//重建二叉树
        if(pre.length==0||in.length==0)
            return null;
        return buildTree(pre,in,0,pre.length-1,0,in.length-1);

    }

    private TreeNode buildTree(int[] pre, int[] in, int startPre, int endPre, int startIn, int endIn) {
        if(startPre>endPre||startIn>endIn)
            return null;
        TreeNode root=new TreeNode(pre[startPre]);
        for(int i=startIn;i<=endIn;i++){
            if(in[i]==pre[startPre]){
                int leftLen=i-startIn;//左子树长度
                int leftPreEnd=startPre+leftLen;
                root.left=buildTree(pre,in,startPre+1,startPre+i-startIn,startIn,i-1);
                root.right=buildTree(pre,in,i-startIn+startPre+1,endPre,i+1,endIn);
                break;
            }
        }
        return root;
    }

//    public TreeLinkNode GetNext(TreeLinkNode pNode)//二叉树的下一个节点
//    {
//        if(pNode==null) return null;
//        if(pNode.right!=null){    //如果有右子树，则找右子树的最左节点
//            pNode = pNode.right;
//            while(pNode.left!=null) pNode = pNode.left;
//            return pNode;
//        }
//        while(pNode.next!=null){ //没右子树，则找第一个当前节点是父节点左孩子的节点
//            if(pNode.next.left==pNode) return pNode.next;
//            pNode = pNode.next;
//        }
//        return null;   //退到了根节点仍没找到，则返回null
//    }
    Stack<Integer> stack1 = new Stack<Integer>();
    Stack<Integer> stack2 = new Stack<Integer>();

    public void push(int node) {//用栈实现队列
        stack1.push(node);
    }

    public int pop() {//用两个栈实现队列
        if(!stack2.isEmpty()){//出队栈不为空时，直接从出队栈中移除栈顶元素
            return stack2.pop();
        }else{//出队栈为空时，从入队栈中依次将元素放入出队栈
            while(!stack1.isEmpty()){
                stack2.push(stack1.pop());
            }
            //return stack2.pop();//放完后，从出队栈依次将栈顶元素弹出
            return stack2.peek();//放完后，从出队栈依次将栈顶元素弹出

        }
    }
    public int Fibonacci(int n) {//斐波那契数列
        int result [] =new int []{0,1};
        if(n < 2){
            return result[n];
        }
        int one = 0;
        int two = 1;
        int sum = 0;
        for(int i = 2;i<=n;i++){
            sum = one + two;
            two = one;
            one = sum;
        }
        return sum;
    }
    public int JumpFloor(int target) {//跳台阶
        if(target == 1) return 1;
        if(target == 2) return 2;
        return JumpFloor(target - 1) + JumpFloor(target - 2);
    }

    public int JumpFloorII(int target) {//跳台阶2
        if (target <= 0) return 0;
        return (int) Math.pow(2, target - 1);
    }
    public int minNumberInRotateArray(int [] array) {//旋转数组中的最小数字
        if(array.length == 0) return 0;
        int left = 0;
        int right = array.length-1;
        int mid =left;
        while(array[left] >= array[right]){
            if(right - left ==1){
                mid = right;
            }
            mid = (left + right)/2;
            if(array[left] ==array[right]&&array[mid]==array[left]){
                return MinInOrder(array,left,right);
            }
            if(array[mid]>=array[left])
                left = mid;
            else if(array[mid]<=array[right])
                right = mid;
        }
        return array[mid];

    }
    public int MinInOrder(int [] arr,int left,int right){
        int res = arr[left];
        for(int i =left+1;i<=right;i++){
            if(arr[i] < res){
                res = arr[i];
            }
        }
        return res;
    }
    public static void printOneToN(int n){
        if(n<=0){
            return;
        }
        char number [] = new char[n];
        for(int i = 0;i<n;i++){
            number[i] = '0';
        }
        for(int i = 0;i<10;i++){
            number[0] = (char) (i + '0');
            printToMaxOfNDigitsRecursively(number, n, 0);
        }

    }

    private static void printToMaxOfNDigitsRecursively(char[] number, int len, int index) {
        if(index == len -1){
            printNumber(number);
            return;
        }
        for(int i = 0;i<10;++i){
            number[index+1] = (char) (i + '0');
            printToMaxOfNDigitsRecursively(number, len, index+1);
        }
    }

    private static void printNumber(char[] number) {
        boolean isBeginning0 = true;
        int nLength = number.length;
        for (int i = 0; i < nLength; ++i) {
            if(isBeginning0 && number[i]!='0'){
                isBeginning0 = false;
            }
            if(!isBeginning0){
                System.out.print(number[i]);
            }
        }
        System.out.println();
    }
    public ListNode FindKthToTail(ListNode head,int k) {//链表中倒数第k个结点
        if(head == null||k<=0)return null;
        ListNode pre = head;
        ListNode last = head;
        for(int i = 0;i < k - 1;i++){
            if(pre.next!=null){
                pre = pre.next;
            }
            else{
                return null;
            }
        }
        while(pre.next!=null){
            pre = pre.next;
            last = last.next;
        }
        return last;
    }
    public void reOrderArray(int [] array) {//调整数组顺序使奇数位于偶数前面
//        if(array.length == 0)return;
//        int begin = 0;
//        int end = array.length-1;
//        while (begin < end){
//            while(begin<end&&(array[begin]&1)!=0)
//                begin++;
//            while(begin<end&&(array[begin]&1)==0)
//                end--;
//            int temp = array[begin];
//            array[begin] = array[end];
//            array[end] = temp;
//        }
        List<Integer>ls1=new ArrayList<Integer>();
        List<Integer>ls2=new ArrayList<Integer>();
        for(int i=0;i<array.length;i++){
            if(array[i]%2==1){
                ls1.add(array[i]);
            }
            else {
                ls2.add(array[i]);
            }
        }
        ls1.addAll(ls2);
        for(int i=0;i<array.length;i++){
            array[i]=ls1.get(i);
        }
    }

    public ListNode EntryNodeOfLoop(ListNode pHead)//链表中环的入口节点
    {
        if(pHead==null)return null;
        ListNode fast = pHead;
        ListNode slow = pHead;
        boolean hasCycle =false;
        while(fast.next!=null&&fast.next.next!=null){
            slow = slow.next;
            fast = fast.next.next;
            if(slow == fast){
                hasCycle = true;
                break;
            }
        }
        if(hasCycle){
            ListNode cur = pHead;
            while (cur!=slow){
                cur = cur.next;
                slow = slow.next;
            }
            return cur;
        }
        else{
            return null;
        }
    }
    public ListNode Merge(ListNode list1,ListNode list2) {//合并两个有序链表
        if(list1 ==null)
            return list2;
        else if(list2==null)
            return  list1;
        ListNode dummyHead = null;
        if(list1.val <list2.val){
            dummyHead.val = list1.val;
            dummyHead.next = Merge(list1.next,list2);
        }
        else{
            dummyHead.val = list2.val;
            dummyHead.next = Merge(list1,list2.next);
        }
        return dummyHead;

    }
    public boolean HasSubTree(TreeNode root1, TreeNode root2){//判断一个二叉树是否为另一个二叉树的子树
        boolean result = false;
        if(root1 != null && root2 !=null){
            if(root1.val == root2.val){
                result = doesTree1HasTree2(root1, root2);
            }
            if (!result){  //如果上个节点失败，则继续比较tree1的左右子节点
                result = doesTree1HasTree2(root1.left, root2) || doesTree1HasTree2(root1.right, root2);
            }
        }
        return result;
    }

    public  boolean doesTree1HasTree2(TreeNode root1, TreeNode root2){

        if (root2 == null)
            return true;
        if (root1 == null)
            return false;
        if (root1.val != root2.val)
            return false;
        return doesTree1HasTree2(root1.left, root2.left) && doesTree1HasTree2(root1.right, root2.right);
    }

    public void Mirror(TreeNode root) {//二叉树的镜像
        if(root == null){
            return;
        }
        if(root.left==null &&root.right==null){
            return;
        }
        TreeNode temp = root.left;
        root.left = root.right;
        root.right = temp;
        if(root.left!=null)
            Mirror(root.left);
        if(root.right!=null){
            Mirror(root.right);
        }

    }
    boolean isSymmetrical(TreeNode pRoot)//对称二叉树
    {
        return isMirror(pRoot,pRoot);
    }

    private boolean isMirror(TreeNode pRoot1, TreeNode pRoot2) {
        if(pRoot1==null&&pRoot2==null)
            return true;
        if(pRoot1==null||pRoot2==null)
            return false;
        if(pRoot1.val!=pRoot2.val)
            return false;
        return isMirror(pRoot1.left,pRoot2.right)&&isMirror(pRoot1.right,pRoot2.left);
    }
    public ArrayList<Integer> printMatrix(int [][] matrix) {
        ArrayList<Integer> results = new ArrayList<Integer>() ;
        if(matrix.length==0) {
            return results;
        }
        int rows=matrix.length-1;
        int cols=matrix[0].length-1;
        int start=0;
        while(start<rows&&start<cols){
            printMatrixClockWisely(matrix,start,rows,cols,results);
            start++;
            rows--;
            cols--;
        }
        if (start == rows) {
            for (int k = start; k<= cols; k++){
                results.add(matrix[start][k]);
            }
        } else if (start == cols) {
            for (int n = start; n<= rows; n++)
                results.add(matrix[n][start]);
        }
        return results;
    }

    private void printMatrixClockWisely(int[][] matrix, int start, int rows, int cols, ArrayList<Integer> result) {
            for(int i=start;i<=cols;i++){
                result.add(matrix[start][i]);
            }
            for(int j=start+1;j<=rows;j++){
                result.add(matrix[j][cols]);
            }
            for(int i=cols-1;i>=start;i--){
                result.add(matrix[rows][i]);
            }
            for(int j=rows-1;j>start;j--){
                result.add(matrix[j][start]);
            }

    }
    public ArrayList<Integer> PrintFromTopToBottom(TreeNode root) {//从上到小打印二叉树
        ArrayList<Integer> results = new ArrayList<Integer>() ;
        if(root == null)
            return results;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while(!queue.isEmpty()){
            TreeNode node = queue.poll();
            results.add(node.val);
            if (node.left!=null)queue.add(node.left);
            if(node.right!=null)queue.add(node.right);
        }
        return results;
    }

    public boolean IsPopOrder(int [] pushA,int [] popA) {//栈的压入，弹出序列
        Stack<Integer> stack = new Stack<>();
        boolean flag = false;
        int popIndex = 0;
        for(int i = 0;i<pushA.length;i++){
            stack.push(pushA[i]);
            while(!stack.isEmpty() &&stack.peek() == popA[popIndex]){
                stack.pop();
                popIndex++;
            }
        }
        if(stack.isEmpty())
            flag = true;
        return flag;
    }

    public static boolean VerifySquenceOfBST(int [] sequence) {//二叉搜索树的后序遍历序列
        if(sequence.length == 0)return false;
        int root = sequence[sequence.length-1];
        int index = 0;
        for(;index<sequence.length-1;index++){
            index = index;
            if(sequence[index] > root){
                break;
            }
        }
        int j = index;
        for(;j<sequence.length-1;j++){
            if(sequence[j] < root){
                return false;
            }
        }
        boolean left = true;
        if(index > 0){
            int leftArr [] = Arrays.copyOfRange(sequence,0,index);
            left = VerifySquenceOfBST(leftArr);
        }
        boolean right = true;
        if(index < sequence.length-1) {
            int rightArr [] = Arrays.copyOfRange(sequence,index,sequence.length-1);
            right = VerifySquenceOfBST(rightArr);
        }
        return left&&right;
    }
    public ArrayList<ArrayList<Integer> > Print(TreeNode pRoot) {//按之字形顺序打印二叉树
        int level = 0;
        ArrayList<ArrayList<Integer>> results = new ArrayList<>();
        if(pRoot == null)
            return results;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(pRoot);
        while(!queue.isEmpty()){
            int count = queue.size();
            ArrayList<Integer> list = new ArrayList<>();
            while (count > 0){
                TreeNode node = queue.poll();
                if(level % 2 ==0){
                    list.add(node.val);
                }
                else {
                    list.add(0,node.val);
                }
                if(node.left !=null)
                    queue.add(node.left);
                if(node.right!=null)
                    queue.add(node.right);
                count--;
            }
            results.add(list);
            level++;
        }
        return results;
    }
    public ArrayList<ArrayList<Integer>> FindPath(TreeNode root,int target) {//二叉树中和为某一值的路径
        ArrayList<ArrayList<Integer>> results = new ArrayList<>();
        if(root ==null)return results;
        ArrayList<Integer> list = new ArrayList<>();
        findPathSub(root,target,list,results);
        return results;
    }

    private void findPathSub(TreeNode root, int target, ArrayList<Integer> list, ArrayList<ArrayList<Integer>> results) {
        if(root!=null){
            if(target == root.val &&root.left==null&&root.right==null){
                list.add(root.val);
                results.add(new ArrayList<Integer>(list));
                list.remove(list.size()-1);
                return;
            }
            else{
                list.add(root.val);
                if(root.left!=null){
                    findPathSub(root.left,target-root.val,list,results);
                }
                if(root.right!=null){
                    findPathSub(root.right,target-root.val,list,results);
                }
                list.remove(list.size()-1);
            }
        }
    }

    public TreeNode Convert(TreeNode pRootOfTree) {//二叉搜索树与双向链表
        List<TreeNode> list = new ArrayList<>();
        if(pRootOfTree == null) return null;
        InOrder(pRootOfTree,list);
        TreeNode headNode = null;
        if(list.size()>0){
            headNode = list.get(0);
            for (int i = 1;i<list.size();i++){
                list.get(i-1).right = list.get(i);
            }
            for (int j = list.size()-1;j>0;j--){
                list.get(j).left = list.get(j-1);
            }
            //list.get(0).left = headNode;
        }
        return headNode;

    }

    private void InOrder(TreeNode pRootOfTree, List<TreeNode> list) {
        if (pRootOfTree!=null){
            InOrder(pRootOfTree.left,list);
            list.add(pRootOfTree);
            InOrder(pRootOfTree.right,list);
        }
    }
//    别人的做法
//    TreeNode head = null;
//    TreeNode realHead = null;
//    public TreeNode Convert(TreeNode pRootOfTree) {
//        ConvertSub(pRootOfTree);
//        return realHead;
//    }
//
//    private void ConvertSub(TreeNode pRootOfTree) {
//        if(pRootOfTree==null) return;
//        ConvertSub(pRootOfTree.left);
//        if (head == null) {
//            head = pRootOfTree;
//            realHead = pRootOfTree;
//        } else {
//            head.right = pRootOfTree;
//            pRootOfTree.left = head;
//            head = pRootOfTree;
//        }
//        ConvertSub(pRootOfTree.right);
//    }
      public static ArrayList<String> Permutation(String str) {//字符串的排列
          int len = str.length();
          ArrayList<String> res = new ArrayList<>();
          char [] ch = str.toCharArray();
          PermutationSub(ch,len,0,res);
          HashSet h = new HashSet(res);
          res.clear();
          res.addAll(h);
          Collections.sort(res);
          return res;
      }

    private static void PermutationSub(char [] ch, int len, int start, ArrayList<String> res) {
        if(start == len -1){
            res.add(new String(ch));
            return;
        }
        for(int i = start;i<len;i++){
            swapStr(ch,start,i);
            PermutationSub(ch,len,start+1, res);
            swapStr(ch,start,i);
        }
    }

    private static void swapStr(char [] ch, int i, int j) {
        char chTemp = ch[i];
        ch[i] = ch[j];
        ch[j] = chTemp;
    }

    public ListNode deleteDuplication(ListNode pHead)//删除链表中重复的节点
    {
        if(pHead==null||pHead.next==null)
            return pHead;
        ListNode dummyHead = new ListNode(0);
        dummyHead.next = pHead;
        ListNode pre = dummyHead;
        ListNode cur = pHead;
        while(cur!=null&&cur.next!=null){
            if(cur.val == cur.next.val){
                cur = cur.next;
                while(cur.next!=null&&cur.val == cur.next.val){
                    cur = cur.next;
                }
                pre.next = cur.next;
                cur = pre.next;
            }else{
                pre = pre.next;
                cur = cur.next;
            }
        }
        return dummyHead.next;
    }
    public RandomListNode Clone(RandomListNode pHead)//复杂链表的复制
    {
        CloneNodes(pHead);
        ConnectSiblingNode(pHead);
        return ReConnectNodes(pHead);
    }

    private RandomListNode ReConnectNodes(RandomListNode pHead) {
        RandomListNode pNode = pHead;
        RandomListNode pCloneHead = null;
        RandomListNode pCloneNode = null;
        if(pNode != null){
            pCloneHead =pCloneNode = pNode.next;
            pNode.next = pCloneNode.next;
            pNode = pNode.next;
        }
        while (pNode!=null){
            pCloneNode.next = pNode.next;
            pCloneNode = pCloneNode.next;
            pNode.next = pCloneNode.next;
            pNode = pNode.next;
        }
        return pCloneHead;
    }

    private void ConnectSiblingNode(RandomListNode pHead) {
        RandomListNode cur = pHead;
        while (cur != null){
            RandomListNode pCloned = cur.next;
            if(cur.random!=null){
                pCloned.random = cur.random.next;
            }
            cur = pCloned.next;
        }
    }

    private void CloneNodes(RandomListNode pHead) {
        RandomListNode cur = pHead;
        while(cur!=null){
            RandomListNode newCloneNode = new RandomListNode(cur.val);
            newCloneNode.next = cur.next;
            newCloneNode.random = null;
            cur.next = newCloneNode;
            cur = newCloneNode.next;
        }
    }
    public static String Serialize(TreeNode root) {
        StringBuilder sb = new StringBuilder();
        getSerializeString(root, sb);
        if(sb.length() != 0)
            sb.deleteCharAt(sb.length()-1);
        return sb.toString();
    }
    public static void getSerializeString(TreeNode root, StringBuilder sb){
        if(root == null)
            sb.append("#,");
        else{
            sb.append(root.val + ",");
            getSerializeString(root.left, sb);
            getSerializeString(root.right, sb);
        }
    }
    static int DeserializeIndex;
    public static TreeNode Deserialize(String str) {
        String [] s= str.split(",");
        DeserializeIndex = -1;
        TreeNode node = DeserializeSub(s);
        return node;
    }

    private static TreeNode DeserializeSub(String[] s) {
        DeserializeIndex++;
        TreeNode node = null;
        if(!s[DeserializeIndex].equals("#")){
            node = new TreeNode(Integer.valueOf(s[DeserializeIndex]));
            node.left = DeserializeSub(s);
            node.right = DeserializeSub(s);
        }
        return node;
    }

    ArrayList<ArrayList<Integer> > Print1(TreeNode pRoot) {//二叉树打印多行
        ArrayList<ArrayList<Integer>> results = new ArrayList<>();
        if(pRoot == null) return results;
        int count = 0;
        Queue<TreeNode> queue = new LinkedList<>();
        queue.add(pRoot);
        while (!queue.isEmpty()){
            count = queue.size();
            ArrayList<Integer> list = new ArrayList<>();
            while (count > 0){
                TreeNode node = queue.poll();
                list.add(node.val);
                if(node.left !=null)
                    queue.add(node.left);
                if(node.right!=null)
                    queue.add(node.right);
                count--;
            }
            results.add(list);
        }
        return results;
    }

    public static TreeNode KthNode(TreeNode pRoot, int k)//二叉搜索树的第K个结点
    {
        if(pRoot == null || k <= 0){
            return null;
        }
        List<TreeNode> list = new ArrayList<>();
        Stack<TreeNode> stack = new Stack<>();
        while (pRoot !=null|| !stack.isEmpty()){
            if(pRoot != null){
                stack.push(pRoot);
                pRoot = pRoot.left;
            }
            else{
                pRoot = stack.pop();
                list.add(pRoot);
                pRoot = pRoot.right;
            }
        }
        if(k > list.size())
            return null;
        else
            return list.get(k-1);
    }
    public int MoreThanHalfNum_Solution(int [] array) {//数组中出现次数超过一半的数字
        if(array.length < 1) return 0;
        Arrays.sort(array);
        int mid = array[array.length/2];
        int count = 0;
        for(int i = 0;i<array.length;i++){
            if(array[i] == mid)
                count++;
        }
        if(count > array.length/2){
            return mid;
        }
        else{
            return 0;
        }
    }
    private static int partition(int[] arr, int low, int high) {
        int pivot = arr[low];               // 枢轴记录
        while (low < high) {
            while (low < high && arr[high] >= pivot) --high;
            arr[low] = arr[high];           // 交换比枢轴小的记录到左端
            while (low < high && arr[low] <= pivot) ++low;
            arr[high] = arr[low];           // 交换比枢轴小的记录到右端
        }
        // 扫描完成，枢轴到位
        arr[low] = pivot;
        // 返回的是枢轴的位置
        return low;
    }
    public static ArrayList<Integer> GetLeastNumbers_Solution(int [] input, int k) {//最小的K个数
        ArrayList<Integer> res = new ArrayList<Integer>();

        if(input.length < k || k == 0)
            return res;
        int start = 0;
        int end = input.length - 1;
        int index = partition(input,start,end);
        while(index != k - 1 ){
            if(index > k-1){
                end = index -1;
                index = partition(input,start,end);
            }
            else{
                start = index +1;
                index = partition(input,start,end);
            }
        }
        for(int i = 0;i<k;i++){
            res.add(input[i]);
        }
        return res;
    }
    public static int FindGreatestSumOfSubArray(int[] array) {//连续子数组的最大和
        if(array.length == 0) return 0;
        int max = array[0];
        int dp [] = new int[array.length];
        dp[0] = array[0];
        for(int i = 1; i < array.length;i++){
            dp[i] = Math.max(dp[i-1] + array[i] ,array[i]);
        }
        for(int j = 0;j < dp.length;j++){
            System.out.print(dp[j]+" ");
            if(dp[j] > max){
                max = dp[j];
            }
        }
        return max;
    }
    //数据流中的中位数
    private static PriorityQueue<Integer> minHeap = new PriorityQueue<>();
    private static PriorityQueue<Integer> maxHeap = new PriorityQueue<>(new Comparator<Integer>() {
        @Override
        public int compare(Integer i, Integer j) {
            return j.compareTo(i);
        }
    });

    public static void Insert(Integer num) {
        int size = minHeap.size() + maxHeap.size();
        if((size & 1) ==0){//偶数
            if(!maxHeap.isEmpty() && num < maxHeap.peek()){
                maxHeap.offer(num);
                num = maxHeap.poll();
            }
            minHeap.offer(num);
        }
        else{
            if(!minHeap.isEmpty() && num > minHeap.peek()){
                minHeap.offer(num);
                num = minHeap.poll();
            }
            maxHeap.offer(num);
        }

    }

    public static Double GetMedian() {
        int size = minHeap.size() + maxHeap.size();
        double ans;
        if((size & 1) == 1){
            ans = minHeap.peek();
        }
        else{
            ans = (maxHeap.peek() + minHeap.peek()) / 2.0;
        }
        return ans;
    }

    public static int NumberOf1Between1AndN_Solution(int n) {//整数中1出现的次数（从1到n整数中1出现的次数）
        if(n < 0){
            return 0;
        }
        String str= Integer.toString(n);
        int result = getNumberOf1(str, 0);
        return result;
    }
    public static int getNumberOf1(String str,int index){
        int length = str.length()-index;
        if(length==1 && str.charAt(index)=='0'){
            return 0;
        }
        if(length==1){
            return 1;
        }
        //以21345为例 10000 + 8000 + 821
        //计算最高位的1
        int first = str.charAt(index)-'0';
        int result = 0;
        if(first>1){
            result += exp(length-1);
        }else if(first==1){
            result += 1 + Integer.parseInt(str.substring(index+1));
        }
        //计算除了最高位的其他位
        result += first *(length-1)*exp(length-2);
        //计算比如2345中0---345中1的个数进行递归
        result += getNumberOf1(str, index+1);
        return result;
    }

    public static int exp(int n){
        int result =1;
        while(n>=1){
            result*=10;
            n--;
        }
        return result;
    }

    public static int digitAtIndex(int index){
        if(index < 0)
            return -1;
        int digits = 1;
        while (true){
            long numbers = countOfIntegers(digits);
            if(index < numbers * digits)
                return digitAtIndexSub(index,digits);
            index -= digits * numbers;
            digits++;
        }
    }

    private static int digitAtIndexSub(int index, int digits) {
        //1000000000
        long number = beginNumber(digits) + index/digits;
        long indexFromRight = digits -index%digits;
        for(int i = 1;i<indexFromRight;i++){
            number /=10;
        }
        return (int)number%10;
    }

    private static long beginNumber(int digits) {
        if(digits == 1)return 0;
        return (long)Math.pow(10,digits-1);
    }

    private static long countOfIntegers(int digits) {
        if(digits == 1)return 10;
        long count = (int)Math.pow(10,digits-1);
        return 9 * count;
    }

    public String PrintMinNumber(int [] numbers) {//把数组排成最小的数
        if(numbers == null ||numbers.length == 0) return "";
        String [] str = new String[numbers.length];
        StringBuffer sb = new StringBuffer();
        for(int i = 0;i<numbers.length;i++){
            str[i] = String.valueOf(numbers[i]);
        }
        Arrays.sort(str, new Comparator<String>() {
            @Override
            public int compare(String o1, String o2) {
                String s1 = o1 + o2;
                String s2 = o2 + o1;
                return s1.compareTo(s2);
            }
        });
        for(int i = 0;i<str.length;i++){
            sb.append(str[i]);
        }
        return sb.toString();
    }

    public static int GetUglyNumber_Solution(int index) {//丑数
        //时间效率太低
//        if(index <= 0) return 0;
//        int count = 0;
//        int NuglyNumber = 0;
//        while (NuglyNumber < index){
//            count++;
//            if(isUgly(count)){
//                NuglyNumber++;
//            }
//        }
//        return count;
        if(index <= 0) return 0;
        int nums[] = new int[index];
        nums[0] = 1;
        int number2 = 0;
        int number3 = 0;
        int number5 = 0;
        int indexCur = 1;
        while(indexCur < index){
            int min = Math.min(nums[number2] * 2,Math.min(nums[number3] * 3,nums[number5] * 5));
            nums[indexCur] = min;
            if(min == nums[number2] * 2) number2++;
            if(min == nums[number3] * 3) number3++;
            if(min == nums[number5] * 5) number5++;
            indexCur++;
        }
        return nums[indexCur-1];
    }
    public static boolean isUgly(int num){
        while(num % 2 ==0)
            num = num / 2;
        while(num % 3 ==0)
            num = num / 3;
        while(num % 5 ==0)
            num = num / 5;
        return (num == 1) ? true : false;
    }
    public static int FirstNotRepeatingChar(String str) {//第一个只出现一次的字符
        Map<Character,Integer> map = new LinkedHashMap<>();
        for(int i=0;i<str.length();i++){
            char c=str.charAt(i);
            if(map.containsKey(c)){
                int count=map.get(c);
                count+=1;
                map.put(c, count);
            }
            else{
                map.put(c, 1);
            }
        }
        char res = 'a';
        for(Map.Entry<Character,Integer> entry: map.entrySet())
        {
            if(entry.getValue() == 1){
                res = entry.getKey();
                break;
            }
        }
        return str.indexOf(res);
    }
    public static int InversePairs(int [] array) {//数组中的逆序对
        if(array.length <= 0)
            return 0;
        int count = InversePairsCore(array,0,array.length-1);
        return count;
    }

    static int InversePairsCoreCount = 0;
    private static int InversePairsCore(int[] array, int start, int end) {
        if(start<end){
            int mid=(start+end)/2;
            InversePairsCore(array, start, mid);
            InversePairsCore(array, mid+1, end);
            mergeCount(array, start, mid, end);
        }
        return InversePairsCoreCount;
    }

    private static void mergeCount(int[] array, int start, int mid, int end) {
        int [] copy = new int[array.length];
        int p1 = start;
        int p2 = mid+1;
        int k = start;

        while(p1<=mid && p2<=end){
            if(array[p1]<=array[p2])
                copy[k++]=array[p1++];
            else {
                copy[k++] = array[p2++];
                InversePairsCoreCount = (InversePairsCoreCount+mid-p1+1)%1000000007;
            }
        }

        while(p1<=mid)
            copy[k++]=array[p1++];
        while(p2<=end)
            copy[k++]=array[p2++];

        for (int i = start; i <=end; i++)
            array[i]=copy[i];
    }

//    public static void merge(int []a,int left,int mid,int right){
//        int []tmp=new int[a.length];//辅助数组
//        int p1=left,p2=mid+1,k=left;//p1、p2是检测指针，k是存放指针
//
//        while(p1<=mid && p2<=right){
//            if(a[p1]<=a[p2])
//                tmp[k++]=a[p1++];
//            else
//                tmp[k++]=a[p2++];
//        }
//
//        while(p1<=mid) tmp[k++]=a[p1++];//如果第一个序列未检测完，直接将后面所有元素加到合并的序列中
//        while(p2<=right) tmp[k++]=a[p2++];//同上
//
//        //复制回原素组
//        for (int i = left; i <=right; i++)
//            a[i]=tmp[i];
//    }
//
//    public static void mergeSort(int[] a, int start, int end){
//        if(start<end){//当子序列中只有一个元素时结束递归
//            int mid=(start+end)/2;//划分子序列
//            mergeSort(a, start, mid);//对左侧子序列进行递归排序
//            mergeSort(a, mid+1, end);//对右侧子序列进行递归排序
//            merge(a, start, mid, end);//合并
//        }
//    }

    public ListNode FindFirstCommonNode(ListNode pHead1, ListNode pHead2) {//两个链表的第一个公共结点
        long len1 = findListLength(pHead1);
        long len2 = findListLength(pHead2);
        long lenDiff = len1 - len2;
        ListNode longHead = pHead1;
        ListNode shortHead = pHead2;
        if(len2 > len1){
            lenDiff = len2 - len1;
            longHead = pHead2;
            shortHead = pHead1;
        }
        for(int i = 0;i<lenDiff;i++){
            longHead = longHead.next;
        }
        while(longHead != null && shortHead != null && longHead != shortHead){
            longHead = longHead.next;
            shortHead = shortHead.next;
        }
        ListNode firstCommonNode = longHead;
        return firstCommonNode;
    }

    private long findListLength(ListNode pHead) {
        long len = 0;
        ListNode tempHead = pHead;
        while (tempHead != null){
            ++len;
            tempHead = tempHead.next;
        }
        return len;
    }

    public int GetNumberOfK(int [] array , int k) {//数字在排序数组中出现的次数
        int count = 0;
        for(int i = 0;i<array.length;i++){
            if(array[i] == k)
                count++;
        }
        return count;
    }
    public boolean IsBalanced_Solution(TreeNode root) {//平衡二叉树
        if(root == null) return true;
        return Math.abs(TreeDepth(root.right) - TreeDepth(root.left)) <= 1 && IsBalanced_Solution(root.left) && IsBalanced_Solution(root.right);
    }

    public int TreeDepth(TreeNode root) {//二叉树的深度
        if(root==null){
            return 0;
        }
        int leftMax=TreeDepth(root.left);
        int rightMax=TreeDepth(root.right);
        return 1+Math.max(leftMax,rightMax);
    }

    public static void FindNumsAppearOnce(int [] array,int num1[] , int num2[]) {//数组中只出现一次的数字
        if(array.length < 2) return;
        int res = 0;
        for(int i=0;i< array.length;i++){
            res = res^array[i];
        }
        int index = findFirstBitIs(res);
        int res1 = 0, res2 = 0;
        for(int j = 0;j<array.length;j++){
            if(IsBit(array[j],index))
                res1 = res1 ^ array[j];
            else
                res2 = res2 ^ array[j];
        }
        num1[0] = res1;
        num2[0] = res2;
    }

    private static boolean IsBit(int num, int index) {//判断该位是不是为1
        num = num >> index;
        return (num & 1) == 1;
    }

    public static int findFirstBitIs(int num){//获取从低位开始的第一个1
        int indexBit = 0;
        while(((num & 1)==0) && (indexBit)<8*4){
            num = num >> 1;
            ++indexBit;
        }
        return indexBit;
    }
    public static ArrayList<ArrayList<Integer> > FindContinuousSequence(int sum) {//和为S的连续正数序列
        ArrayList<ArrayList<Integer>> res = new ArrayList<>();
        if(sum < 3)
            return res;
        int small = 1;
        int big = 2;
        int middle = (1 + sum)/2;
        int curSum = small + big;//因为最起码两个数，small大于middle就不可能有这么一个序列了
        while(small < middle){
            if(curSum == sum){
                ArrayList<Integer> temp = new ArrayList<>();
                for (int i = small;i<=big;i++){
                    temp.add(i);
                }
                res.add(temp);
                big++;
                curSum = curSum + big;
            }
            while(curSum > sum && small < middle){//大了
                curSum = curSum - small;
                small++;
            }
            while(curSum < sum && small < middle){//小了
                big++;
                curSum = curSum + big;
            }
        }
        return res;
    }

    public ArrayList<Integer> FindNumbersWithSum(int [] array,int sum) {//和为S的两数字
        ArrayList<Integer> res = new ArrayList<>();
        if(array.length < 1)
            return res;
        int left = 0;
        int right = array.length - 1;
        while (left < right){
            int tempSum = array[left] + array[right];
            if(tempSum == sum){
                res.add(array[left]);
                res.add(array[right]);
                break;
            }
            else if(tempSum > sum){
                right--;
            }
            else{
                left++;
            }
        }
        return res;
    }
    public String LeftRotateString(String str,int n) {//左旋转字符串
        if(str.equals(""))return "";
        StringBuffer sb = new StringBuffer();
        for(int i = n;i<str.length();i++){
            sb.append(str.charAt(i));
        }
        for(int i =0;i<n;i++){
            sb.append(str.charAt(i));
        }
        return sb.toString();
    }
    public static String ReverseSentence(String str) {//翻转单词顺序列
        if(str==null || str.length()<=0)
            return "";
        if(str.trim().equals("")){
            return str;
        }
        String [] strArr = str.split(" ");
        StringBuffer sb = new StringBuffer();
        for(int i = 0;i<strArr.length;i++){
            StringBuffer temp = new StringBuffer(strArr[i]);
            temp.reverse();
            sb.append(temp);
            sb.append(" ");
        }
        sb.reverse();
        return sb.toString().substring(1);
    }

    public boolean isContinuous(int [] numbers) {//扑克牌顺子
        if(numbers.length!=5)
            return false;
        Arrays.sort(numbers);
        int countZero = 0;
        int countGap = 0;
        for(int i =0;i<numbers.length;i++){
            if(numbers[i] == 0)
                countZero++;
        }
        int left = countZero;
        int right = left + 1;
        while(right < numbers.length){
            if(numbers[left] == numbers[right]){
                return false;
            }
            countGap = countGap + numbers[right] - numbers[left] - 1;
            left = right;
            right++;
        }
        return countGap > countZero ? false:true;
    }
    public int LastRemaining_Solution(int n, int m) {//孩子们的游戏
        if(n<1||m<1)
            return -1;
        LinkedList<Integer> list = new LinkedList<>();
        for(int i =0;i<n;i++){
            list.add(i);
        }
        int start = 0;
        while (list.size() > 1){
            start = (start + m - 1) % list.size();
            list.remove(start);
        }
        return list.get(0);
    }

    public int Sum_Solution(int n) {//1+2+3+。。。。。+n
        int sum = n;
        boolean ans = (n>0)&&((sum+=Sum_Solution(n-1))>0);
        return sum;
    }
    public static int Add(int num1,int num2) {//不用加减乘除做加法
        int sum = 0;
        int add = 0;
        int count = 0;
        while(num2 !=0){
            count++;
            System.out.println(count+" "+sum+" "+add);
            sum = num1 ^num2;
            add = (num1 & num2) << 1;
            System.out.println(count+" "+sum+" "+add);
            num1 = sum;
            num2 = add;
        }
        return num1;
    }
    public static int StrToInt(String str) {//把字符串转换成整数
        if (str.equals("") || str.length() == 0)
            return 0;
        char [] c = str.toCharArray();
        int symbol = 1;
        int start = 0;
        int sum = 0;
        if(c[0] == '-'){
            symbol = -1;
            start = 1;
        }
        else if(c[0] == '+'){
            start = 1;
        }
        for(int i = start;i<c.length;i++){
            if(c[i]>'9'||c[i]<'0')
                return 0;
            sum = sum * 10 + (int) (c[i] - '0');
            if(sum>Integer.MAX_VALUE)
                return 0;
        }
        return symbol*sum;
    }
    public static int[] multiply(int[] A) {//构建乘机数组
        int length = A.length;
        int[] B = new int[length];
        if(length != 0 ){
            B[0] = 1;
            //计算下三角连乘
            for(int i = 1; i < length; i++){
                B[i] = B[i-1] * A[i-1];
            }
            int temp = 1;
            //计算上三角
            for(int j = length-2; j >= 0; j--){
                temp *= A[j+1];
                System.out.println(temp);
                B[j] *= temp;
                System.out.println(B[j]+" "+j);
            }
        }
        return B;
    }
    public static boolean match(char[] str, char[] pattern)//正则表达式
    {
        if(str == null||pattern == null){
            return false;
        }
        return matchCore(str,0,str.length,0,pattern.length,pattern);
    }

    private static boolean matchCore(char[] str,int strIndex,int strLen,int patternIndex,int patternLen, char[] pattern) {
        if(strIndex == strLen && patternIndex == patternLen)
            return true;
        if(strIndex!=strLen && patternIndex ==patternLen)
            return false;
        if(patternIndex + 1 < patternLen && pattern[patternIndex+1] =='*'){
            if((pattern[patternIndex]=='.' && strIndex!=strLen)||(strIndex!=strLen && pattern[patternIndex] ==str[strIndex])){
                return matchCore(str,strIndex+1,strLen,patternIndex+2,patternLen,pattern)||//  判定完毕，继续pattern
                        matchCore(str,strIndex+1,strLen,patternIndex,patternLen,pattern)||//继续判定
                        matchCore(str,strIndex,strLen,patternIndex+2,patternLen,pattern);//判定完毕 *认定为判了个空
            }
            else {
                return matchCore(str, strIndex, strLen, patternIndex + 2, patternLen, pattern);//直接跳过*的判定
            }
        }
        if((pattern[patternIndex]=='.' && strIndex!=strLen)||(strIndex!=strLen && pattern[patternIndex] ==str[strIndex]))
            return matchCore(str,strIndex+1,strLen,patternIndex+1,patternLen,pattern);
        return false;
    }

    private int indexIsNumeric = 0;
    public boolean isNumeric(char[] str) {
        if (str.length < 1)
            return false;

        boolean flag = scanInteger(str);

        if (indexIsNumeric < str.length && str[indexIsNumeric] == '.') {
            indexIsNumeric++;
            flag = scanUnsignedInteger(str) || flag;
        }

        if (indexIsNumeric < str.length && (str[indexIsNumeric] == 'E' || str[indexIsNumeric] == 'e')) {
            indexIsNumeric++;
            flag = flag && scanInteger(str);
        }

        return flag && indexIsNumeric == str.length;

    }

    private boolean scanInteger(char[] str) {
        if (indexIsNumeric < str.length && (str[indexIsNumeric] == '+' || str[indexIsNumeric] == '-') )
            indexIsNumeric++;
        return scanUnsignedInteger(str);

    }

    private boolean scanUnsignedInteger(char[] str) {
        int start = indexIsNumeric;
        while (indexIsNumeric < str.length && str[indexIsNumeric] >= '0' && str[indexIsNumeric] <= '9')
            indexIsNumeric++;
        return start < indexIsNumeric; //是否存在整数
    }
    static LinkedHashMap<Character,Integer> listStr = new LinkedHashMap();
    public static void Insert(char ch)
    {
        if(listStr.containsKey(ch)){
            listStr.put(ch,listStr.get(ch)+1);
        }
        else
            listStr.put(ch,1);
    }
    //return the first appearence once char in current stringstream
    public static char FirstAppearingOnce()
    {
        for(Map.Entry<Character,Integer> entry: listStr.entrySet()){
            if(entry.getValue() == 1)
                return entry.getKey();
        }
        return '#';
    }
    public static ArrayList<Integer> maxInWindows(int [] num, int size)//滑动窗口的最大值
    {
        ArrayList<Integer> result = new ArrayList<>();
        if (num == null || num.length == 0 || size <= 0 || num.length < size) {
            return result;
        }
        //双端队列，用来记录每个窗口的最大值下标
        //[2,3,4,2,6,2,5,1],3
        LinkedList<Integer> list = new LinkedList<>();
        for(int i = 0;i<num.length;i++){
            while(!list.isEmpty() && num[list.peekLast()] < num[i]){
                list.pollLast();
            }
            list.add(i);
            if(list.peekFirst() == i - size){
                list.pollFirst();
            }
            if(i >= size - 1)
                result.add(num[list.peekFirst()]);
        }
        return result;
    }
}
