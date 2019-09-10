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
        int nums [] = new int[]{1,2,4,7,11,15};
        //sout 是快捷键
        printOneToN(4);

    }

    /*
    面试题 3
    题目 长度为n数组中，所有数字在0 - n-1 范围内，某些数字是重复的，找出一个重复的数组
    输入：{2,3,1,0,2,5,3} 长度 7
    输出：2或3
    第一种方法：
    数组排序，找出相邻相等的就是重复数字 时间复杂度nlogn 哈希表也可以 空间换时间了
    第二种方法：
    首先无重复数字的话，下标即是数字本身，从头到尾扫描，举例说明
    2，3，1，0，2
    下标0与2不相等，与在2的数字交换
    1，3，2，0，2
    下标0仍然与1不相等，与在1的数字交换
    3，1，2，0，2
    继续
    0，1，2，3，2
    此时扫描 0 1 2 3 下标和数字都相等，扫到第二个2的时候，发现 下标 2的位置也是2了，说明重复
     */
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

    /*
    给定一个包含 n + 1 个整数的数组 nums，其数字都在 1 到 n 之间（包括 1 和 n），可知至少存在一个重复的整数。假设只有一个重复的整数，找出这个重复的数。
    不能更改原数组（假设数组是只读的）。只能使用额外的 O(1) 的空间。时间复杂度小于 O(n2) 。
    数组中只有一个重复的数字，但它可能不止重复出现一次。思想还是一个萝卜一个坑，举例说明
    2,3,5,4,3,2,6,7   1~7分两段1~4,5~7 而1~4这范围内的数字出现了5次 重复数字肯定在此了
     */
    public static int findDuplicate(int[] nums) {//不修改数组找出数组中重复的数字
        int start = 1;
        int end = nums.length - 1;
        while(start <= end){
            int mid = (start + end) / 2;
            int count = countRange(nums,start,mid);//统计数字1到mid中间 数字的个数，大于则说明重复元素在此区间
            if(start == end){
                if(count > 1)
                    return start;
            }
            if(count > mid - start + 1) {//在左边
                end = mid;
            }
            else{
                start = mid + 1;
            }
        }
        return -1;
    }

    private static int countRange(int[] nums, int start, int end) {
        int count = 0;
        for(int i = 0;i<nums.length;i++){
            if(nums[i] >=start && nums[i] <= end)
                ++count;
        }
        return count;
    }

    /*
    面试题 4
    在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序
    1 2 8 9
    2 4 9 12
    4 7 10 13
    6 8 11 15  target 7
    第一种方法按行或者列扫描，再二分查找
    第二种 每次找右上角或者左下角
    本例是找右上角 第一个9 7<9 则9这一列不用看了，8这列同理，2<7 按行继续找
     */
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
        boolean flag = false;
        int rows = array.length;//行数
        if(rows == 0)
            return flag;
        int cols = array[0].length;//列数
        if(array != null){
            int row = 0;
            int col = cols - 1;//右上角那个数
            while (row < rows && col >=0){
                if(array[row][col] > target){
                    col--;
                }
                else if(array[row][col] < target){
                    row++;
                }
                else{
                    flag = true;
                    break;
                }
            }
        }
        return flag;
    }
    /*
    面试题 5
    把空格换成%20
     */
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
    /*
    面试题 6 从尾到头打印链表
    借助栈
     */
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
    /*
    面试题 7
    重建二叉树，给一个前序和一个中序，建立二叉树
    前序：1 2 4 7 3 5 6 8  中序 4 7 2 1 5 3 8 6
    树：                       1
                           2       3
                         4       5   6
                          7         8
     */
    public TreeNode reConstructBinaryTree(int [] pre,int [] in) {//重建二叉树
        if(pre.length==0||in.length==0)
            return null;
        return buildTree(pre,in,0,pre.length-1,0,in.length-1);

    }

    private TreeNode buildTree(int[] preOrder, int[] inOrder, int startPreOrder, int endPreOrder, int startInOrder, int endInOrder) {
        if(startPreOrder>endPreOrder||startInOrder>endInOrder)
            return null;
        TreeNode root=new TreeNode(preOrder[startPreOrder]);
        for(int i=startInOrder;i<=endInOrder;i++){
            if(inOrder[i]==preOrder[startPreOrder]){
                int leftLen=i-startInOrder;//左子树长度
                root.left=buildTree(preOrder,inOrder,startPreOrder+1,startPreOrder+i-startInOrder,startInOrder,i-1);
                root.right=buildTree(preOrder,inOrder,i-startInOrder+startPreOrder+1,endPreOrder,i+1,endInOrder);
                break;
            }
        }
        return root;
    }
    /*
    面试题 8
    找出中序遍历中的下一个节点，next指向的是父节点
     1
   2   3
  4 5 6    425163

       1
   2       3
  4 5    6   7
          8
         9          425169837   比如找8的过程 需向上两层 找父节点有左子树的
     */
    public TreeLinkNode GetNext(TreeLinkNode pNode)//二叉树的下一个节点
    {
        if(pNode==null) return null;
        if(pNode.right!=null){    //如果有右子树，则找右子树的最左节点
            pNode = pNode.right;
            while(pNode.left!=null) pNode = pNode.left;
            return pNode;
        }
        while(pNode.next!=null){ //没右子树，则找第一个当前节点是父节点左孩子的节点
            if(pNode.next.left==pNode) return pNode.next;//节点不是根节点。如果该节点是其父节点的左孩子，则返回父节点,直到当前结点是其父节点的左孩子位置
            pNode = pNode.next;
        }
        return null;   //退到了根节点仍没找到，则返回null
    }
    /*
    面试题 9 用两个栈实现队列
    stack2不为空，stack2的栈顶元素就是最先进入队列的元素
    stack2为空时，把stack1的元素逐个弹出压入stack2
     */
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
            return stack2.peek();//放完后，从出队栈依次将栈顶元素弹出

        }
    }
    /*
    面试题 10 斐波那契数列
     */
    public int Fibonacci(int n) {//斐波那契数列
        int result [] =new int []{0,1};
        if(n < 2){
            return result[n];
        }
        int Nminusone = 0;//n-1的意思
        int Nminustwo = 1;//n-2的意思
        int sum = 0;
        for(int i = 2;i<=n;i++){
            sum = Nminusone + Nminustwo;
            Nminustwo = Nminusone;//n-2
            Nminusone = sum;//n-1
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

    /*
    面试题 11 旋转数组的最小数字
    3 4 5 1 2 是1 2 3 4 5 的一个旋转数组
    二分查找，如果mid大left说明前半部分还是大的那个递增区间，最小元素就要去后半部分找
    特殊情况1 0 1 1 1 和 1 1 1 0 1 都可以是 0 1 1 1 1 的旋转，当三数都一样的话 则顺序查找
     */
    public int minNumberInRotateArray(int [] array) {//旋转数组中的最小数字
        if(array.length == 0) return 0;
        int left = 0;
        int right = array.length-1;
        int mid = left;
        while(array[left] >= array[right]){
            if(right - left ==1){
                mid = right;
            }
            mid = (left + right)/2;
            if(array[left] == array[right] && array[mid] == array[left]){
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
    /*
    面试题12
    路径可以从矩阵中的任意一个格子开始，每一步可以在矩阵中向左，向右，向上，向下移动一个格子。如果一条路径经过了矩阵中的某一个格子，则之后不能再次进入这个格子。
    例如 a b c e  这样的3 X 4 矩阵中包含一条字符串"bcced"的路径，但是矩阵中不包含"abcb"路径，因为字符串的第一个字符b占据了矩阵中的第一行第二个格子之后，路径不能再次进入该格子。
         s f c s
         a d e e
     注意matrix是一维数组
     */
    public boolean hasPath(char[] matrix, int rows, int cols, char[] str)
    {
        if(matrix.length==0||rows<1||cols<1||str.length==0){
            return false;
        }

        boolean [] isVisited = new boolean[matrix.length];

        int path =0;
        for(int i = 0;i<rows;i++){
            for(int j =0;j<cols;j++){
                if(hasPathCore(matrix,rows,cols,i,j,str,path,isVisited)){
                    return true;
                }
            }
        }
        return false;
    }
    private boolean hasPathCore(char[] matrix, int rows, int cols, int i, int j, char[] str, int path,
                                boolean[] isVisited) {
        // TODO Auto-generated method stub
        int index = i*cols+j;
        if(i<0||j<0||i>=rows||j>=cols||matrix[index]!=str[path]||isVisited[index]){
            return false;
        }
        if(path == str.length-1) return true;
        isVisited[index] = true;
        if(hasPathCore(matrix, rows, cols, i+1, j, str, path+1, isVisited)||hasPathCore(matrix, rows, cols, i-1, j, str, path+1, isVisited)||
                hasPathCore(matrix, rows, cols, i, j+1, str, path+1, isVisited)||hasPathCore(matrix, rows, cols, i, j-1, str, path+1, isVisited)){
            return true;
        }
        isVisited[index] = false;
        return false;
    }

    /*
    面试题13 机器人的运动范围
    地上有一个m行和n列的方格。一个机器人从坐标0,0的格子开始移动，每一次只能向左，右，上，下四个方向移动一格，但是不能进入行坐标和列坐标的数位之和大于k的格子。
    例如，当k为18时，机器人能够进入方格（35,37），因为3+5+3+7 = 18。
    但是，它不能进入方格（35,38），因为3+5+3+8 = 19。请问该机器人能够达到多少个格子？
     */
    public static int movingCount(int threshold, int rows, int cols)//机器人的运动范围
    {
        boolean [] flag = new boolean[rows*cols];
        for(int i=0;i<rows*cols;i++)
            flag[i]=false;
        int count = movingCountCore(threshold,0,0,rows,cols,flag);

        return count;
    }
    private static int movingCountCore(int threshold, int i, int j, int rows, int cols, boolean[] flag) {
        // TODO Auto-generated method stub
        int count = 0;
        if(movingCountValid(threshold,i,j,rows,cols,flag)){
            int index = i*cols+j;
            flag[index] = true;
            count = 1+movingCountCore(threshold,i+1,j,rows,cols,flag)+movingCountCore(threshold,i-1,j,rows,cols,flag)
                    +movingCountCore(threshold,i,j+1,rows,cols,flag)+movingCountCore(threshold,i,j-1,rows,cols,flag);
        }
        return count;
    }
    private static boolean movingCountValid(int threshold, int i, int j, int rows, int cols, boolean[] flag) {
        // TODO Auto-generated method stub
        if(i>=0&&j>=0&&i<rows&&j<cols&&flag[i*cols+j]==false&&getDigitSum(i)+getDigitSum(j)<=threshold){
            return true;
        }
        return false;
    }
    private static int getDigitSum(int num) {
        // TODO Auto-generated method stub
        int sum = 0;
        while(num>0){
            sum +=num%10;
            num=num/10;
        }
        return sum;
    }
    /*
    面试题14 剪绳子
    一根长度为n的绳子，剪成m段 求每段最大乘积
    n=2 1*1 n=3 1*2 dp[3]=3 是 后面需要，绳子长度为3
     */
    public static int integerBreak(int n) {
        if(n < 2)
            return 0;
        if(n == 2)
            return 1;
        if(n == 3)
            return 2;
        int dp[] = new int[n+1];
        dp[0] = 0;
        dp[1] = 1;dp[2] = 2;dp[3] = 3;
        int max = 0;
        for(int i = 4;i <= n;i++){
            for(int j = 1;j <= i/2;j++){
                int temp = dp[j] * dp[i-j];
                if(max < temp)
                    max = temp;
                dp[i] = max;
            }
            max = dp[n];
        }
        for(int i =0;i<=n;i++)
            System.out.print(dp[i]+" ");
        return max;
    }
    /*
    面试题15 二进制中1的个数
    把一个数减去1时，如果该整数的二进制最右边一位是1，则最后一位变为0，其余位不变；如果最右边一位不是1则必定为0，一个整数只要不是0，则至少有一位为1，减去1后
    最靠右边的1变为0，再靠右边的0全变为1。
    假设n=1100（2）时，减去1的结果为1011，再和1100做位与运算，得到1000，把最右边的1变为了0，如此反复的做下去，即可统计出1的个数。

     */
    public int NumberOf1(int n) {
        int count = 0;
        while(n!=0){
            ++count;
            n = (n-1)&n;
        }
        return count;
    }
    /*
    面试题17 打印从1到最大的n位数
    0~9全排列
    然后去除最前面的那个0
     */
    public static void printOneToN(int n){
        if(n<=0){
            return;
        }
        char number [] = new char[n];
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

    /*
    面试题18 删除链表的中的节点,看注释
     */
    public void deleteNodeEasy(ListNode node) {//leetcode上的链表删除，不是头节 不是尾节点 节点数>2
        node.val = node.next.val;
        node.next = node.next.next;
    }
    public void deleteNode(ListNode head, ListNode toBeDeleted){
        //参数校验
        if(head == null || toBeDeleted == null){
            return ;
        }
        //链表中只有一个节点，那么待删除的节点既是头结点，又是尾结点
        if(head == toBeDeleted && head.next == null){
            head = null;
        }else{
            //待删除的节点是尾节点
            if(toBeDeleted.next == null){
                ListNode temp = head;
                while(temp.next != toBeDeleted){
                    temp = temp.next;
                }
                temp.next = null;
            }else{          //待删除的节点不是尾节点
                toBeDeleted.val = toBeDeleted.next.val;
                toBeDeleted.next = toBeDeleted.next.next;
            }
        }
    }
    /*
    面试题18 删除链表的中的重复节点
    pre前置指针 和一个当前指针 移动
     */
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
    /*
    面试题19 正则表达式匹配

     */
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
    /*
    面试题20 表示数值的字符串
    如果出现 . 就去判定小数
     */
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
    //扫描可以是正负开头的0~9的数位
    private boolean scanInteger(char[] str) {
        if (indexIsNumeric < str.length && (str[indexIsNumeric] == '+' || str[indexIsNumeric] == '-') )
            indexIsNumeric++;
        return scanUnsignedInteger(str);

    }
    //扫描0~9的数位
    private boolean scanUnsignedInteger(char[] str) {
        int start = indexIsNumeric;
        while (indexIsNumeric < str.length && str[indexIsNumeric] >= '0' && str[indexIsNumeric] <= '9')
            indexIsNumeric++;
        return start < indexIsNumeric; //是否存在整数
    }
    /*
    面试题21 调整数组顺序使奇数位与偶数前面
    第一种方法类似于快排，双指针不停互换
    第二种直接就开辅助数组
     */
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

    /*
    面试题22 链表中倒数第k个结点
    快慢指针 快指针先走 k-1 步
    然后快慢指针一起走 走到头 慢指针所指就是 倒数第k个结点
     */
    public ListNode FindKthToTail(ListNode head,int k) {//链表中倒数第k个结点
        if(head == null||k<=0)return null;//注意让程序崩溃的地方
        ListNode pre = head;
        ListNode last = head;
        for(int i = 0;i < k - 1;i++){
            if(pre.next!=null){
                pre = pre.next;
            }
            else{
                return null;//注意让程序崩溃的地方
            }
        }
        while(pre.next!=null){
            pre = pre.next;
            last = last.next;
        }
        return last;
    }

    /*
    面试题23 链表中环的入口结点
    快慢指针，如果相遇证明有环
    此时慢指针和一个新指针（从头开始）再同时走，再相遇的位置为环的人口，可以画图来看 l = n l是头到入口，n是快慢指针相遇位置绕圈到入口
     */
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

    /*
    面试题24 反转链表
    为了避免断链，需要先把即将改变指向的那个结点保存一下，知道它所指向的那个结点
     */
    public ListNode ReverseList(ListNode head) {
        if (null == head || null == head.next) {
            return head;
        }
        ListNode next = null;
        ListNode pre = null;
        while(head != null){
            next = head.next;
            head.next = pre;
            pre = head;
            head = next;
        }
        return pre;
    }
    /*
    面试题25 合并两个有序链表
    有一个没了就直接续上，递归调用
    */
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
    /*
         面试题26 树的子结构
         第一步先判断两树根节点是否相等，相等则进行第二步判定，不相等 或者 相等但并非子树 都继续比较tree1的左右节点
         第二步 一直递归调用查看是否节点值都相等，知道子树结束（到叶子节点） 则成功
     */
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

    /*
     * 面试题27：二叉树的镜像
     * 输入一个二叉树，该函数输出它的镜像
     * 先先序遍历这颗树，如果遍历到的节点都有子节点 就交换他的两个子节点
     */
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
    /*
     * 面试题28 对称的二叉树
     * 判断一颗二叉树是不是和它的镜像一样
     */
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

    /*
     * 面试题29 顺时针打印矩阵
     * 解法与书上略有不同
     * 模拟顺时针打印，就是边界值一定要注意
     */
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
        //行数为奇数，还剩一行最后
        if (start == rows) {
            for (int k = start; k<= cols; k++){
                results.add(matrix[start][k]);
            }
        } else if (start == cols) {//列数为奇数，还剩一行最后
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

    /* 面试题30 包含min函数的栈 最小栈
     * 实现 min push pop函数
     * 每次把最小的数push到辅助栈中-》s2， push操作 s1正常入栈，s2为空或者s2栈顶元素大于要插入的值则s2入栈，pop操作s1正常操作，如果出栈的是最小，s2也出
     */
    private Stack<Integer> s1 = new Stack<>();
    private Stack<Integer> s2 = new Stack<>();

    public void pushMinStack(int x) {
        s1.push(x);
        if(s2.isEmpty()||s2.peek()>=x) s2.push(x);
    }

    public void popMinStack() {
        int x=s1.pop();
        if(s2.peek()==x)s2.pop();
    }

    public int top() {
        return s1.peek();
    }

    public int min() {
        return s2.peek();
    }

    /* 面试题31 栈的压入、弹出序列
     * 输入两个序列 一个是栈的压入顺序 一个是弹出序列，判断 这个弹出序列是否能由压入顺序得到
     * 使用一个辅助栈， 如果下一个弹出的数字刚好是栈顶数字，则直接弹出，如果弹出数字不在栈顶，说明还没压入辅助栈，则压入，如果所有数字都押入了还没找到下一个弹出数字，false
     */
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

    /*
     * 面试题32 从上打印二叉树（层序遍历）
     */
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
    /* 面试题 32 变型2
     * 上述的变型
     */
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

    /* 面试题 32 变型2
     * 上述的变型
     */
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
                    list.add(0,node.val);//从头部插
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

    /* 面试题33 二叉搜索树的后续遍历序列
     * 输入一个数组 判断是不是某二叉搜索树的后续遍历结果 5 7 6 9 11 10 8
     * 首先最后一个数肯定是根节点， 小于根节点的都是左树 大于是右，再递归调用
     */
    public static boolean VerifySquenceOfBST(int [] sequence) {//二叉搜索树的后序遍历序列
        if(sequence.length == 0)return false;
        int root = sequence[sequence.length-1];
        int index = 0;
        for(;index<sequence.length-1;index++){
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
        return left && right;
    }

    /*
     * 面试题34 二叉树中 和为某一值的路径
     * 输入一颗树和target 输出根节点到叶子节点所经过的结点之和为target
     * 前序遍历某一节点 把该节点添加到路径上 并累加值，当路径累加值等于target且 该节点是叶子节点则找了，然后回溯 继续递归
     */
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

    /*
     * 面试题35 复杂链表的复制
     * 一个链表除了有指向next的指针，还有random指针，不一定指向哪个节点，复制一条这种链表
     * 困难在于，复制的时候，每次找random节点的指向都需要从头开始扫描
     * 解决办法：先把每个节点都复制出来一份，A->B-> 改为A->A'->B->B'>C->C'
     * 第二步 把A' B'这种的random指向对应节点，第三步拆分，奇偶就可以拆开
     */
    public RandomListNode Clone(RandomListNode pHead)//复杂链表的复制
    {
        CloneNodes(pHead);
        ConnectSiblingNode(pHead);
        return ReConnectNodes(pHead);
    }
    private void CloneNodes(RandomListNode pHead) {//第一步
        RandomListNode cur = pHead;
        while(cur!=null){
            RandomListNode newCloneNode = new RandomListNode(cur.val);
            newCloneNode.next = cur.next;
            newCloneNode.random = null;
            cur.next = newCloneNode;
            cur = newCloneNode.next;
        }
    }
    private void ConnectSiblingNode(RandomListNode pHead) {//第二步
        RandomListNode cur = pHead;
        while (cur != null){
            RandomListNode pCloned = cur.next;
            if(cur.random!=null){
                pCloned.random = cur.random.next;//这样才能让所有Clone节点串一起
            }
            cur = pCloned.next;
        }
    }
    private RandomListNode ReConnectNodes(RandomListNode pHead) {//第三步
        RandomListNode pNode = pHead;
        RandomListNode pCloneHead = null;
        RandomListNode pCloneNode = null;
        if(pNode != null){
            pCloneHead = pCloneNode = pNode.next;
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

    /* 面试题36 二叉搜索树与双向链表
     * 输入一颗二叉搜索树，将该二叉搜索树转换成一个排序的双向链表，不准new 只能调整指向
     * 由于要求链表是排好序的，直接中序遍历
     */
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
    //    别人的做法 总体思想在于中序的时候就把链表生成了，只不过用了两个头节点
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

    /*
     * 面试题37 序列化二叉树
     * 1 2 4 $ $ $ 3 5 $ $ 6 $ $
     *               1
     *             2   3
     *            4   5 6
     * 前序遍历形成序列化的字符串，反序列化 是#的就是null 否则按index去数组找到对应值
     */
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

    /* 面试题38 字符串的排列
     * 先求出能出现在第一个位置的所有字符，固定第一个位置，求后面所有字符的排列
     */
    public static ArrayList<String> Permutation(String str) {//字符串的排列
        int len = str.length();
        ArrayList<String> res = new ArrayList<>();
        char [] ch = str.toCharArray();
        PermutationSub(ch,len,0,res);
        HashSet h = new HashSet(res); //去重
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
            swapStr(ch,start,i);//回溯
        }
    }
    private static void swapStr(char [] ch, int i, int j) {
        char chTemp = ch[i];
        ch[i] = ch[j];
        ch[j] = chTemp;
    }
    /*
     * 面试题39 数组中出现次数超过一半的数字
     * 数组排序 超一半存在的话 必然是中位数，然后统计这个中位数的count
     */
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

    /* 面试题40 最小的K个数
     * 快排思想 基于数组的第k个数字来调整，比k小去左边 比k大去右边，当 轴值正好落在k-1处 划分完毕
     * 这题也可以用堆排序
     */
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
    /* 面试题41 数据流中的中位数
     * 数据的数目是不断变化的
     * 保证数据容器左边数据都小于右边的数据， 最大堆实现左边数据容器，最小堆实现右边数据容器
     * 为实现平均分配 偶数时插入最小堆 奇数最大堆
     * 插入时 这个树小于最大堆的根节点，则入堆，去除最大堆最大放入最小堆
     * 因为偶数插入最小堆，取值时候 size是奇数，中位数一定落在最小堆中，且是根节点
     */
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
        if((size & 1) == 1){//偶数
            ans = minHeap.peek();
        }
        else{
            ans = (maxHeap.peek() + minHeap.peek()) / 2.0;
        }
        return ans;
    }

    /*
     * 面试题42  连续子数组的最大和
     * 一个数组有正 有负 1 -2 3 10 -4 7 2 -5  最大连续子数组 3 10 -4 7 2 -5  和为18
     * 动态规划
     */
    public static int FindGreatestSumOfSubArray(int[] array) {//连续子数组的最大和
        if(array.length == 0) return 0;
        int max = array[0];
        int dp [] = new int[array.length];
        dp[0] = array[0];
        for(int i = 1; i < array.length;i++){
            dp[i] = Math.max(dp[i-1] + array[i] ,array[i]);
        }
        for(int j = 0;j < dp.length;j++){
            if(dp[j] > max){
                max = dp[j];
            }
        }
        return max;
    }

    /*
     * 面试题43 从1到n整数中1出现的次数
     *  1~n 这n个整数的十进制表示中1出现的次数
     *  以21345为例 10000 + 8000 + 821
     */
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

    /* 面试题44 数字序列中某一位的数字
     * 数字以0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 。。。。。序列化到一个字符串中，求任意第n位对应的数字
     */
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

    /* 面试题 45 把数组排成最小的数
     * 输入一个正整数数组 打印出所有数字拼接一起最小的数 如[3.32.32] 321323
     * 数组中的每个数排序 从小到大之后再拼接
     */
    public String PrintMinNumber(int [] numbers) {//把数组排成最小的数
        if(numbers == null ||numbers.length == 0) return "";
        String [] str = new String[numbers.length];
        StringBuffer sb = new StringBuffer();
        for(int i = 0;i<numbers.length;i++){
            str[i] = String.valueOf(numbers[i]);
        }
//        Arrays.sort(str, new Comparator<String>() {
//            @Override
//            public int compare(String o1, String o2) {
//                String s1 = o1 + o2;
//                String s2 = o2 + o1;
//                return s1.compareTo(s2);
//            }
//        });
        //Java 8 新特性 Lambda表达式 排序写法
        Comparator<String> com = (String x, String y) -> (x + y).compareTo(y + x);
        Arrays.sort(str,com);
        for(int i = 0;i<str.length;i++){
            sb.append(str[i]);
        }
        return sb.toString();
    }

    /* 面试题47 礼物得最大价值
     * 动态规划 处理好边界
     */
    public static int getMaxValue(int [][] numbers){
        int rows = numbers.length;
        int cols = numbers[0].length;
        int dp[][] = new int[rows][cols];
        for(int i = 0;i < rows;i++){
            for(int j = 0;j < cols;j++){
                int left = 0;
                int up = 0;
                if(i > 0){
                    up = dp[i-1][j];
                }
                if(j > 0){
                    left = dp[i][j-1];
                }
                dp[i][j] = Math.max(left,up) + numbers[i][j];
            }
        }
        return dp[rows-1][cols-1];
    }

    /* 面试题48
     * 最长不含重复字符得子串
     * 输入: "abcabcbb" 输出: 3
     * 解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
     */
    public static int lengthOfLongestSubstring(String s) {
        StringBuffer sb = new StringBuffer("");
        int max=0;
        for(int i = 0;i < s.length();i++){
            String str = s.substring(i,i+1);
            if(sb.indexOf(str) == -1){
                sb.append(str);
            }
            else{
                sb=new StringBuffer(sb.substring(sb.indexOf(str)+1));//如果重复了 就从这个重复得字符串开始截取 如 abc又遇到了a 就截取bc
                sb.append(str);
            }
            if(sb.length() > max){
                max=sb.length();
            }
        }
        return max;
    }
    /* 面试题49 丑数
     * 输入 index 返回第index个丑数
     * 只包含因子 2 3 5 得数 如：6 8 14就不是 包含了因子7
     * 第二种思路 不在非丑数得整数上浪费时间 把每个丑数得基础上乘以2 或 3 或 5 排好序
     */
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

    /* 面试题50 第一个只出现一次的字符
     * 不多说了 直接上HashMap
     */
    public static int FirstNotRepeatingChar(String str) {//第一个只出现一次的字符
        Map<Character,Integer> map = new LinkedHashMap<>();
        for(int i = 0;i<str.length();i++){
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

    /* 面试题50变型 字符流中第一个只出现一次的字符
     * 差不多
     */
    static Map<Character,Integer> charMap = new LinkedHashMap<>();
    public static void insertChar(char c){
        if(charMap.containsKey(c)){
            int count = charMap.get(c);
            count = count + 1;
            charMap.put(c, count);
        }
        else{
            charMap.put(c, 1);
        }
    }
    public static char getFirstChar(){
        char c = '\0';
        for(Map.Entry<Character,Integer> entry: charMap.entrySet())
        {
            if(entry.getValue() == 1){
                c = entry.getKey();
                break;
            }
        }
        return c;
    }

    /* 面试题51 数组中的逆序对
     * 7 5 6 4  五个逆序对 76 75 74 64 54
     * 把长度为4的数组分解成两个长度为2的子数组；把长度为2的数组分解成两个成都为1的子数组；把长度为1的子数组 合并、排序并统计逆序对 ；把长度为2的子数组合并、排序，并统计逆序对
     * 每个子数组统计完，排好序再重放如数组中，之所以排序是因为这一部分已经统计完毕。
     */
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

    /* 面试题52 两个链表的第一个公共结点
     * 暴力法 第二种办法：如果两个链表有公共节点 肯定是Y形 先获取每个链表得长度，得到长度差值，然后再一起遍历
     */
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

    /* 面试题53 在排序数组中查找数字
     * 暴力遍历 方法二 二分查找第一个K得出现位置和最后一次得出现位置
     */
    public int GetNumberOfK(int [] array , int k) {//数字在排序数组中出现的次数
        int count = 0;
        for(int i = 0;i<array.length;i++){
            if(array[i] == k)
                count++;
        }
        return count;
    }
    public static int GetNumberOfK1(int [] array , int k) {//数字在排序数组中出现的次数
        int count = 0;
        int first = getFirstK(array,k,0,array.length-1);
        int last = getLastK(array,k,0,array.length-1);
        if(first > -1 && last > -1){//find it ！
            count = last - first + 1;
        }
        return count;
    }

    private static int getFirstK(int[] array, int k, int start, int end) {
        while (start <= end){
            int mid = (start + end) / 2;
            if(array[mid] < k){
                start = mid + 1;
            }
            else if(array[mid] > k){
                end = mid - 1;
            }
            else {
                if((mid > 0 && array[mid - 1] !=k) || mid == 0){
                    return mid;
                }
                else{
                    end = end - 1;
                }
            }
        }
        return -1;
    }
    private static int getLastK(int[] array, int k, int start, int end) {
        while (start <= end){
            int mid = (start + end) / 2;
            if(array[mid] < k){
                start = mid + 1;
            }
            else if(array[mid] > k){
                end = mid - 1;
            }
            else {
                if((mid<array.length-1&&array[mid+1]!=k)||mid==array.length-1){
                    return mid;
                }
                else{
                    start = mid + 1;
                }
            }
        }
        return -1;
    }
    /* 面试题53 的变型
     * 给定一个包含 0, 1, 2, ..., n 中 n 个数的序列，找出 0 .. n 中没有出现在序列中的那个数。
     * 因为缺失了一个 所以 要多 异或一个 n 输入: [3,0,1] 输出: 2
     */
    public static int missingNumber(int[] nums) {
        int ret = 0;
        for(int i = 0; i <= nums.length; ++i) {
            ret ^= i;
        }
        for (int j = 0; j < nums.length; ++j) {
            ret ^= nums[j];
        }
        return ret;
    }
    /* 面试题54 二叉搜索树的第K大节点
     * 直接中序遍历
     */
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
    /* 面试题 55 二叉树的深度
     * 递归即可
     */
    public int TreeDepth(TreeNode root) {//二叉树的深度
        if(root==null){
            return 0;
        }
        int leftMax=TreeDepth(root.left);
        int rightMax=TreeDepth(root.right);
        return 1+Math.max(leftMax,rightMax);
    }
    /* 面试题 55 变型 平衡二叉树
     * 递归即可 左右子树高度 + 左子树是否也是平衡 + 右子树是否也是平衡
     */
    public boolean IsBalanced_Solution(TreeNode root) {//平衡二叉树
        if(root == null) return true;
        return Math.abs(TreeDepth(root.right) - TreeDepth(root.left)) <= 1 && IsBalanced_Solution(root.left) && IsBalanced_Solution(root.right);
    }

    /* 面试题 55 最小深度
     *
     */
    public int minDepth(TreeNode root) {
        if(root == null)
            return 0;
        if(root.left == null && root.right ==null)//左右皆空，为叶子节点
            return 1;
        int min_depth = Integer.MAX_VALUE;

        if (root.left != null) {
            min_depth = Math.min(minDepth(root.left), min_depth);
        }
        if (root.right != null) {
            min_depth = Math.min(minDepth(root.right), min_depth);
        }
        return min_depth + 1;
    }

    /* 面试题 55 最大深度
     *
     */
    public int maxDepth(TreeNode root) {
        if (root == null) {
            return 0;
        } else {
            int left_height = maxDepth(root.left);
            int right_height = maxDepth(root.right);
            return Math.max(left_height, right_height) + 1;
        }
    }

    /* 面试题56 数组中数字出现的次数（只出现一次的数字 两次）
     * 一个整型数组里除了两个数字之外，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字。
     * 因为是找出两个，找出一个很好找 直接异或运算 最后的值就是那个只出现一次的数字
     * 可以先从头到尾异或，那么这个结果一定是那两个只出现一次数字的异或结果，那么找出第一个 为 1 的位置，已这个来划分数组，
     * 这一位为0 一个子数组 为1 一个子数组。这样这两个数肯定分别在两个数组中了
     * 例子 2 4 3 6 3 2 5 5  4 6 的异或 0010 按倒数第二位是不是1划分 2 3 6 3 2  和  4 5 5
     */
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
        while(((num & 1)==0) && (indexBit)<8*4){//int的位长
            num = num >> 1;
            ++indexBit;
        }
        return indexBit;
    }
    /* 面试题56 数组中数字出现的次数 变型 1个数字出现1次 其他的都出现3次
     * 把数字中的每个数转为 二进制 所有数的每一位参与累加 如果能被3整除 说明这一位 那个只出现一次的数是0 反之是1
     */
    public static int singleNumber(int[] nums) {
        int result = 0;
        int bitSum [] = new int [32];
        for(int i = 0;i < nums.length;i++){
            int bitMask = 1;
            for(int j = 31;j >= 0;--j){
                int bit = nums[i] & bitMask;
                if(bit != 0){
                    bitSum[j] += 1;
                }
                bitMask = bitMask << 1;//依次判断个位 十位 百位。。。
            }
        }

//        for(int i = 0;i < 32;i++){
//            result = result << 1;
//            System.out.println(result);
//            result = result + (bitSum[i] % 3);
//        }
        for(int i = 0;i < 32;i++){
            System.out.println(i+" "+bitSum[i]);
            result = result << 1;
            System.out.println("111    "+result);
            result = result + (bitSum[i] % 3);
            System.out.println("222    "+result);

        }
        return result;
    }

    /* 面试题 57 和为S的数字
     * 递增排序数组 和 数字 S 查找两个数 和 为 S
     * 举例 1 2 4 7 11 15 target 15  4+11 = 15
     */
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
    /* 面试题 57 变型 和为S的连续正数序列
     * 输入一个正数 s 打印所有和为s的连续正数序列
     * 举例 target 15  1+2+3+4+5 = 4+5+6 = 7+8 三组
     * 模拟过程 如果target为9 1 2 小了 big++ 1 2 3 继续big++ 1 2 3 4 大了 small++ 减去small 2 3 4 正好 big++ 2 3 4 5 然后 3 4 5 然后 4 5 停止
     */
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

    /* 面试题 58 翻转字符串
    例如，“I am a student.” -> “student. a am I”
    用好sb就可以了
     */
    public static String ReverseSentence(String str) {//翻转单词顺序列
        if(str==null || str.length()<=0)
            return "";
        if(str.trim().equals("")){
            return str;
        }
        String [] strArr = str.split(" ");
        StringBuffer sb = new StringBuffer();
        for(int i = 0;i < strArr.length;i++){
            StringBuffer temp = new StringBuffer(strArr[i]);
            temp.reverse();
            sb.append(temp);
            sb.append(" ");
        }
        sb.reverse();
        return sb.toString().substring(1);
    }

    /* 面试题 58 翻转字符串
    变型 左旋转字符串 S=”abcdefg”,要求输出循环左移2位后的结果，即“cdefgab”
    先分成两个部分 分别旋转 ba gfedc 再合在一起旋转 得到 cdefgab
     */
    public String LeftRotateString(String str,int n) {//左旋转字符串
//        if(str.equals(""))return "";
//        StringBuffer sb = new StringBuffer();
//        for(int i = n;i < str.length();i++){
//            sb.append(str.charAt(i));
//        }
//        for(int i = 0;i < n;i++){
//            sb.append(str.charAt(i));
//        }
//        return sb.toString();
        if(str == null || str.length() == 0){
            return str;
        }

        int len = str.length();

        if(len > 0 && n > 0 && n < len){
            char firstStart = str.charAt(0);
            char firstEnd = str.charAt(n - 1);
            char secondStart = str.charAt(n);
            char secondEnd = str.charAt(len-1);

            reverse(firstStart, firstEnd);
            reverse(secondStart, secondEnd);
            reverse(firstStart, secondEnd);
        }

        return str;
    }

    public void reverse(char begin, char end){

        while(begin < end){
            char c = begin;
            begin = end;
            end = c;

            begin++;
            end--;
        }
    }

    /* 面试题 59 队列的最大值
    输入 [2,3,4,2,6,2,5,1],3 针对数组{2,3,4,2,6,2,5,1}的滑动窗口有以下6个： {[2,3,4],2,6,2,5,1}， {2,[3,4,2],6,2,5,1}，
    {2,3,[4,2,6],2,5,1}，{2,3,4,[2,6,2],5,1}， {2,3,4,2,[6,2,5],1}， {2,3,4,2,6,[2,5,1]}。
    双端队列，用来记录每个窗口的最大值下标（注意是下标值）队列头存的是最大值，因为头有可能会滑出去，所有需要尾部也存后来的数据，如果当前值
    大于已存在的尾部值，那么这个尾部就没必要继续存在了（不可能成为最大值了）
     */
    public static ArrayList<Integer> maxInWindows(int [] num, int size)//滑动窗口的最大值
    {
        ArrayList<Integer> result = new ArrayList<>();
        if (num == null || num.length == 0 || size <= 0 || num.length < size) {
            return result;
        }
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
    public static int Add(int num1,int num2) {//不用加减除做加法
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


    public static class TreeLinkNode {
        int val;
        TreeLinkNode left = null;
        TreeLinkNode right = null;
        TreeLinkNode next = null;

        TreeLinkNode(int val) {
            this.val = val;
        }
    }
    public static class RandomListNode {
        public int val;
        public RandomListNode next;
        public RandomListNode random;

        public RandomListNode() {}
        RandomListNode(int label) {//牛客网的构造函数
            this.val = label;
        }

        public RandomListNode(int _val,RandomListNode _next,RandomListNode _random) {
            val = _val;
            next = _next;
            random = _random;
        }
    }
    public static class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;
        TreeNode(int x) { val = x; }
    }
    public static class ListNode {
        int val;
        ListNode next=null;
        ListNode(int val) {
            this.val=val;
        }
    }
}
