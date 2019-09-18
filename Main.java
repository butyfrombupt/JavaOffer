
import java.util.*;

public class Main {



    public static void main(String[] args) {

        int [] arr = new int[]{1,3,5,4,7};

        final HashMap<Integer,Integer> map = new HashMap<>();
        map.put(1,2);
        System.out.println(map.size());
        String s = "YZX";
        System.out.println(titleToNumber(s));
        
    }

    // new added
    public int numIslands(char[][] grid) {//岛屿的数量

        if(grid == null || grid.length <=0){
            return 0;
        }
        int count = 0;
        int rows = grid.length;
        int cols = grid[0].length;

        for(int i = 0;i < rows;i++){
            for(int j = 0;j < cols;j++){
                if(grid[i][j] == '1'){
                    count++;
                    dfs(grid,i,j);
                }
            }
        }
        return count;
    }

    private void dfs(char[][] grid, int row, int col) {
        int rows = grid.length;
        int cols = grid[0].length;
        if(row < 0 || col < 0 || row >=rows || col >= cols || grid[row][col] == '0'){
            return;
        }
        grid[row][col] = '0';
        dfs(grid,row + 1,col);
        dfs(grid,row - 1,col);
        dfs(grid,row,col - 1);
        dfs(grid,row,col + 1);
    }


    public ListNode ReverseList(ListNode head) {
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


    //交错字符串
    public static boolean wangwang(String str1,String str2,String str3){

        int index1 = 0;
        int index2 = 0;
        int index3 = 0;
        return wangwangSub(str1,str2,str3,index1,index2,index3);
    }


    public static boolean wangwangSub(String str1,String str2,String str3,int index1,int index2,int index3){
        //System.out.println(index1 + " "+ index2 +" "+index3 );
        if(index1 == str1.length() && index2 == str2.length()  && index3 == str3.length() ){
            return true;
        }
        boolean flag = false;
        if(index1 < str1.length() && index3 < str3.length() && str1.charAt(index1) == str3.charAt(index3)){
            flag = flag | wangwangSub(str1,str2,str3,index1+1,index2,index3+1);
        }
        if(index2 < str2.length() && index3 < str3.length() && str2.charAt(index2) == str3.charAt(index3)){
            flag = flag | wangwangSub(str1,str2,str3,index1,index2+1,index3+1);
        }
        return flag;
    }
        //买卖股票开始！

    /*
    穷举法 框架
    dp[i][k][0 or 1]
    0 <= i <= n-1, 1 <= k <= K
    n 为天数，大 K 为最多交易数
    此问题共 n × K × 2 种状态，全部穷举就能搞定。

    for 0 <= i < n:
    for 1 <= k <= K:
        for s in {0, 1}:
            dp[i][k][s] = max(buy, sell, rest)

    每天都有三种「选择」：买入、卖出、无操作，我们用 buy, sell, rest 表示这三种选择
    今天我没有持有股票，有两种可能：
    dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
              max(   继续选择 rest  ,           昨天是持有的 选择 sell      )
    今天我持有股票，同样有两种可能：
    dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])
              max(   继续选择 rest  ,           昨天是未持有的 选择 buy         )
    base case：
    dp[-1][k][0] = dp[i][0][0] = 0 // i = -1 意味着还没有开始 和 k = 0 意味着根本不允许交易 这时候的利润当然是 0
    dp[-1][k][1] = dp[i][0][1] = -infinity // 还没开始的时候 和 不允许交易的情况下 是不可能持有股票的

    状态转移方程：
    dp[i][k][0] = max(dp[i-1][k][0], dp[i-1][k][1] + prices[i])
    dp[i][k][1] = max(dp[i-1][k][1], dp[i-1][k-1][0] - prices[i])


    这段一定要参考
    if (i - 1 == -1) {
        dp[i][0] = 0;
        // 解释：
        //   dp[i][0]
        // = max(dp[-1][0], dp[-1][1] + prices[i])
        // = max(0, -infinity + prices[i]) = 0
        dp[i][1] = -prices[i];
        //解释：
        //   dp[i][1]
        // = max(dp[-1][1], dp[-1][0] - prices[i])
        // = max(-infinity, 0 - prices[i])
        // = -prices[i]
        continue;
    }

     */

    //I 买卖一次
    public int maxProfit(int[] prices) {
        int len = prices.length;
        if(len == 0) return 0;
        int [][]dp = new int[len][2];
        for(int i = 0;i < prices.length;i++){
            if(i == 0){
                dp[i][0] = 0;
                dp[i][1] = -prices[i];
                continue;
            }
            dp[i][0] = Math.max(dp[i-1][0], dp[i-1][1] + prices[i]);
            dp[i][1] = Math.max(dp[i-1][1], - prices[i]);
        }
        return dp[len-1][0];
    }

    //II 买卖无限次数
    public int maxProfit2(int[] prices) {
        int len = prices.length;
        if(len == 0) return 0;
        int [][]dp = new int[len][2];
        for(int i = 0;i < prices.length;i++){
            if(i == 0){
                dp[i][0] = 0;
                dp[i][1] = -prices[i];
                continue;
            }
            dp[i][0] = Math.max(dp[i-1][0], dp[i-1][1] + prices[i]);
            dp[i][1] = Math.max(dp[i-1][1], dp[i-1][0] - prices[i]);
        }
        return dp[len-1][0];
    }

    //III 买卖两次
    public int maxProfit3(int[] prices) {
        int len = prices.length;
        if(len == 0) return 0;
        int dp[][][] = new int[len][3][2];
        for(int i = 0;i < len;i++){
            for(int j = 2;j >= 1;j--){
                if(i == 0){
                    dp[i][j][0] = 0;
                    dp[i][j][1] = -prices[i];
                    continue;
                }
                dp[i][j][0] = Math.max(dp[i-1][j][0], dp[i-1][j][1] + prices[i]);
                dp[i][j][1] = Math.max(dp[i-1][j][1], dp[i-1][j-1][0] - prices[i]);
            }
        }
        return dp[len-1][2][0];
    }

    // IV 范性 可以买卖k次
    public int maxProfit4(int k, int[] prices) {
        int len = prices.length;
        if(len == 0) return 0;
        if(k > len / 2)
            return maxProfit2(prices);//可交易次数大于 len/2 则肯定是无限次了，因为一次交易肯定是一买一卖

        int dp[][][] = new int[len][k+1][2];
        for(int i = 0; i < len;i++){
            for(int j = k; j >=1;j--){
                if(i == 0){
                    dp[i][j][0] = 0;
                    dp[i][j][1] = -prices[i];
                    continue;
                }
                dp[i][j][0] = Math.max(dp[i-1][j][0], dp[i-1][j][1] + prices[i]);
                dp[i][j][1] = Math.max(dp[i-1][j][1], dp[i-1][j-1][0] - prices[i]);
            }
        }
        return dp[len-1][k][0];
    }


    //309. 最佳买卖股票时机含冷冻期
    public int maxProfit5(int[] prices) {
        int len = prices.length;
        if(len == 0) return 0;
        int [][]dp = new int[len][2];
        for(int i = 0;i < prices.length;i++){
            if(i == 0){
                dp[i][0] = 0;
                dp[i][1] = -prices[i];
                continue;
            }
            dp[i][0] = Math.max(dp[i-1][0], dp[i-1][1] + prices[i]);
            if(i >=2){
                dp[i][1] = Math.max(dp[i-1][1], dp[i-2][0] - prices[i]);
            }
            else{
                dp[i][1] = Math.max(dp[i-1][1], - prices[i]);
            }
        }
        return dp[len-1][0];
    }

    //714. 买卖股票的最佳时机含手续费
    public int maxProfit(int[] prices, int fee) {
        int len = prices.length;
        if(len == 0) return 0;
        int [][]dp = new int[len][2];
        for(int i = 0;i < prices.length;i++){
            if(i == 0){
                dp[i][0] = 0;
                dp[i][1] = -prices[i];
                continue;
            }
            dp[i][0] = Math.max(dp[i-1][0], dp[i-1][1] + prices[i]);
            dp[i][1] = Math.max(dp[i-1][1], dp[i-1][0] - prices[i] - fee);
        }
        return dp[len-1][0];
    }


    //后序遍历二叉树，非递归
    public List<Integer> postorderTraversal(TreeNode root) {
        List<Integer> list = new ArrayList<Integer>();
        if(root == null)
            return list;
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        while(!stack.isEmpty()){
            TreeNode node = stack.pop();
            if(node.left != null){
                stack.push(node.left);
            }
            if(node.right != null){
                stack.push(node.right);
            }
            list.add(0,node.val);
        }
        return list;
    }

    //前序遍历二叉树，非递归
    public List<Integer> preorderTraversal(TreeNode root) {
        List<Integer> list = new ArrayList<Integer>();
        if(root == null)
            return list;
        Stack<TreeNode> stack = new Stack<>();
        stack.push(root);
        while(!stack.isEmpty()){
            TreeNode node = stack.pop();
            list.add(node.val);
            if(node.right != null){
                stack.push(node.right);
            }
            if(node.left != null){
                stack.push(node.left);
            }

        }
        return list;
    }


    //中序遍历二叉树，非递归
    public List<Integer> inorderTraversal(TreeNode root) {
        List<Integer> list = new ArrayList<>();
        if(root == null)
            return list;
        Stack<TreeNode> stack = new Stack<>();
        while(root != null || !stack.isEmpty()){
            while (root != null){
                stack.push(root);
                root = root.left;
            }
            TreeNode node = stack.pop();
            list.add(node.val);
            root = root.right;
        }
        return list;
    }


    //222 完全二叉树的节点个数
    public int countNodes(TreeNode root) {
        if(root == null){
            return 0;
        }
        int leftDepth = 0;
        int rightDepth = 0;
        TreeNode tempLeftRoot = root;
        TreeNode tempRightRoot = root;

        while (tempLeftRoot.left != null){
            leftDepth++;
            tempLeftRoot = tempLeftRoot.left;
        }

        while (tempRightRoot.right != null){
            rightDepth++;
            tempRightRoot = tempRightRoot.right;
        }

        if(leftDepth == rightDepth){
            return (int)Math.pow(2,leftDepth + 1) - 1;
        }
        else{
            return countNodes(root.left) + countNodes(root.right) + 1;
        }
    }

    /*
    236 二叉树的最近公共祖先 后序遍历
     */

    private TreeNode res = null;
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root == null){
            return res;
        }
        lowestCommonAncestorSub(root,p,q);
        return res;
    }

    private boolean lowestCommonAncestorSub(TreeNode root, TreeNode p, TreeNode q) {
        if(root == null){
            return false;
        }

        int left = lowestCommonAncestorSub(root.left,p,q) ? 1 : 0;//有一个在左边

        int right = lowestCommonAncestorSub(root.right,p,q) ? 1 : 0;//有一个在右边

        int pRoot  = (root == p || root == q) ? 1 : 0;//当前有一个就是啦

        if(left + right + pRoot >= 2){
            res = root;
        }
        return (left + right + pRoot) > 0 ? true : false;
    }

    /*
    450 删除二叉搜索树中的节点
     */
    public TreeNode deleteNode(TreeNode root, int key) {

        return root;
    }

    //5,7,3,3,7
    public static int countArray(int []nums,int target){
        int count = 0;
        HashSet<Integer> set = new HashSet<>();
        for(int i = 0;i < nums.length;i++){
            if(set.contains(nums[i]) && isUnique(nums[i],target - nums[i])){
                count++;
            }
            else{
                set.add(target - nums[i]);
            }
//            for(int j = i + 1;j < nums.length;j++){
//                if(nums[i] + nums[j] == target && isUnique(nums[i],nums[j])){
//                    count++;
//                }
//            }
        }
        return count;
    }

    static HashMap<String,Integer> map1 = new HashMap<>();

    private static boolean isUnique(int num1, int num2) {
        String key = num1 + "," + num2;
        if(map1.containsKey(key)){
            return false;
        }
        else{
            map1.put(key,1);
            return true;
        }
    }

    public static int robotRes = 0;

    public static void robotDfs(int x,int y,int len,char []ch,int deep,int robotCount){
        if(deep == len){
            if(x == 0 && y == 0 && robotCount > robotRes){
                robotRes = robotCount;
                return;
            }
            else{
                return;
            }
        }
        if(ch[deep] == 'L'){
            robotDfs(x-1,y,len,ch,deep+1,robotCount+1);
        }
        if(ch[deep] == 'R'){
            robotDfs(x+1,y,len,ch,deep+1,robotCount+1);
        }
        if(ch[deep] == 'U'){
            robotDfs(x,y+1,len,ch,deep+1,robotCount+1);
        }
        if(ch[deep] == 'D'){
            robotDfs(x,y-1,len,ch,deep+1,robotCount+1);
        }
    }

    public static List<String> restoreIpAddresses(String s) {//93复原ip地址
        int len = s.length();
        List<String> results = new ArrayList<>();
        if(len > 12){
            return results;
        }
        StringBuilder sb = new StringBuilder();
        restoreIpAddressesSub(s,0,0,sb,results);
        return results;
    }

    //93复原ip地址
    private static void restoreIpAddressesSub(String s, int count, int index, StringBuilder sb, List<String> results) {
        if(index == s.length()){
            if(count == 4){
                results.add(sb.toString().substring(0,sb.toString().length() - 1));
            }
            return;
        }
        if(s.length() - index > (4 - count) * 3){//剩下的没处理的位数比3大
            return;
        }
        if(count == 4 || index > s.length()){
            return;
        }

        StringBuilder before = new StringBuilder(sb);//保存一下之前的值，回溯用
        sb.append(s.charAt(index));
        sb.append(".");
        restoreIpAddressesSub(s,count + 1,index + 1,sb,results);

        if(s.charAt(index) == '0'){ // 多位数的情况 01 不可能存在
            return;
        }

        if(index + 2 <= s.length()){
            sb = new StringBuilder(before);//恢复为之前的解
            sb.append(s.substring(index, index + 2));
            sb.append(".");
            restoreIpAddressesSub(s,count + 1,index + 2,sb,results);
        }
        if(index + 3 <= s.length()){
            sb = new StringBuilder(before);//恢复为之前的解
            int num = Integer.parseInt(s.substring(index, index + 3));
            if (num >= 0 && num <= 255) {
                sb.append(s.substring(index, index + 3));
                sb.append(".");
                restoreIpAddressesSub(s,count + 1,index + 3,sb,results);
            }
        }
    }

    /*
    92. 反转链表 II
     */
    public ListNode reverseBetween(ListNode head, int m, int n) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;

        ListNode last = new ListNode(0);//直接走到n处，获得n后面不反转的链表
        last = head;
        for(int i = 0; i < n; i++){
            last = last.next;
        }

        ListNode pre = dummy;
        for(int i = 1; i < m; i++){
            pre = pre.next;
        }
        head = pre.next;
        ListNode temp = head;
        pre.next = ReverseList(temp,n-m+1);//反转一部分链表
        head = pre.next;


        for(int i = 1; i< n-m+1;i++){//走完反转的这部分链表
            head = head.next;
        }

        head.next = last;//把后半部分给续上
        return dummy.next;
    }
    public ListNode ReverseList(ListNode head,int count) {
        ListNode next = null;
        ListNode pre = null;
        while(head != null && count > 0){
            count--;
            next = head.next;
            head.next = pre;
            pre = head;
            head = next;
        }
        return pre;
    }

    //奇偶链表
    public ListNode oddEvenList(ListNode head) {
        if(head == null){
            return null;
        }
        ListNode evenHead = null;
        ListNode odd = head;
        ListNode even = head.next;
        evenHead = even;
        while (even != null){
            odd.next = even.next;
            odd = odd.next;
            even.next = odd.next;
            even = even.next;
        }
        odd.next = evenHead;
        return head;
    }

    //填充每个节点的下一个右侧节点指针
    public Node connect(Node root) {
        if(root == null)
            return null;
        ArrayList<ArrayList<Node>> results = new ArrayList<>();
        int count = 0;
        Queue<Node> queue = new LinkedList<>();
        queue.add(root);
        while (!queue.isEmpty()){
            count = queue.size();
            ArrayList<Node> list = new ArrayList<>();
            while (count > 0){
                Node node = queue.poll();
                list.add(node);
                if(node.left !=null)
                    queue.add(node.left);
                if(node.right!=null)
                    queue.add(node.right);
                count--;
            }
            results.add(list);
        }
        for(ArrayList<Node> nodes : results){
            for(int i = 0;i < nodes.size();i++){
                if(i < nodes.size() - 1){
                    nodes.get(i).next = nodes.get(i+1);
                }
                else{
                    nodes.get(i).next = null;
                }
            }
        }
        return root;
    }

    //78 子集
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> results = new ArrayList<>();
        if(nums == null || nums.length ==0){
            return results;
        }
        List<Integer> list = new ArrayList<>();
        subsetsSub(0,nums,list,results);
        return results;
    }

    private void subsetsSub(int index, int[] nums, List<Integer> list, List<List<Integer>> results) {
        results.add(new ArrayList<>(list));
        for(int i = index;i < nums.length;i++){
            list.add(nums[i]);
            subsetsSub(i+1,nums,list,results);
            list.remove(list.size() - 1);
        }
    }

    public boolean exist(char[][] board, String word) {
        int rows = board.length;
        int cols = board[0].length;
        if(board.length == 0 || rows < 1 || cols < 1 || word.length() == 0){
            return false;
        }
        boolean isVisited[][] = new boolean[rows][cols];
        int index = 0;
        for(int i = 0;i < rows;i++){
            for(int j = 0;j < cols;j++){
                if(existSub(i,j,rows,cols,board,word,index,isVisited)){
                    return true;
                }
            }
        }
        return false;
    }

    private boolean existSub(int i, int j, int rows, int cols, char[][] board, String word, int index, boolean[][] isVisited) {
        if(i < 0 || i >= rows || j < 0 || j >= cols || board[i][j] != word.charAt(index) || isVisited[i][j] == true){
            return false;
        }
        if(index == word.length() - 1){
            return true;
        }
        isVisited[i][j] = true;
        if(existSub(i - 1, j, rows, cols, board, word, index + 1, isVisited) ||
                existSub(i + 1, j, rows, cols, board, word, index + 1, isVisited) ||
                existSub(i, j - 1, rows, cols, board, word, index + 1, isVisited) ||
                existSub(i, j + 1, rows, cols, board, word, index + 1, isVisited)){
            return true;
        }
        isVisited[i][j] = false;
        return false;
    }

    //347. 前 K 个高频元素   著名的top k 问题，堆和快排
    public List<Integer> topKFrequent(int[] nums, int k) {
        HashMap<Integer,Integer> map = new HashMap<>();
        List<Integer> results = new LinkedList<>();

        for(int i = 0;i < nums.length;i++){
            map.put(nums[i], map.getOrDefault(nums[i], 0) + 1);
        }
        PriorityQueue<Integer> heap = new PriorityQueue<>(
                (num1,num2) ->map.get(num1) - map.get(num2)
        );
        for(int key : map.keySet()){
            heap.add(key);
            if(heap.size() > k){
                heap.poll();
            }
        }
        for(int i = 0;i < k;i++){
            results.add(heap.poll());
        }
        Collections.reverse(results);
        return results;
    }

    //215. 数组中的第K个最大元素
    public int findKthLargest(int[] nums, int k) {
        int start = 0;
        int end = nums.length - 1;
        int index = partition(nums,start,end);
        while(index != k - 1 ){
            if(index > k-1){
                end = index -1;
                index = partition(nums,start,end);
            }
            else{
                start = index +1;
                index = partition(nums,start,end);
            }
        }
        return nums[k];
    }

    private static int partition(int[] arr, int low, int high) {
        int pivot = arr[low];               // 枢轴记录
        while (low < high) {
            while (low < high && arr[high] <= pivot) --high;
            arr[low] = arr[high];           // 交换比枢轴大的记录到左端
            while (low < high && arr[low] >= pivot) ++low;
            arr[high] = arr[low];           // 交换比枢轴小的记录到右端
        }
        // 扫描完成，枢轴到位
        arr[low] = pivot;
        // 返回的是枢轴的位置
        return low;
    }

    //162. 寻找峰值
    public int findPeakElement(int[] nums) {
        int left = 0;
        int right = nums.length - 1;
        while (left < right){
            while (left < right && nums[left] < nums[right])
                left++;
            while (left < right && nums[left] > nums[right])
                right--;
        }
        return nums[nums[left] > nums[right] ? left : right];
    }

    //334. 递增的三元子序列
    public static boolean increasingTriplet(int[] nums) {
        int min = Integer.MAX_VALUE;
        int secondMax = Integer.MAX_VALUE;
        for(int i = 0;i < nums.length;i++){
            if(nums[i] < min){
                min = nums[i];
            }
            else if(nums[i] > min && nums[i] <= secondMax){
                secondMax = nums[i];
            }
            else{
                return true;
            }
        }
        return false;
    }

    //    ((ur)oi)
    public static String wangwang(String s){
        Stack<Integer> stack = new Stack<>();
        ArrayList<Integer> list = new ArrayList<>();
        for(int i = 0;i < s.length();i++){
            if(s.charAt(i) == '(' || s.charAt(i) == ')'){
                if(!stack.isEmpty() && isDouble(s.charAt(stack.peek()),s.charAt(i))){
                    list.add(stack.pop());
                    list.add(i);
                }
                else{
                    stack.push(i);
                }
            }
        }
        if(!stack.isEmpty()){
            return "";
        }
        for(int i = 0;i < list.size();i = i + 2){
            StringBuilder sb = new StringBuilder();
            sb.append(s.substring(list.get(i)+1, list.get(i+1)));
            sb.reverse();
            s = s.replace(s.substring(list.get(i)+1, list.get(i+1)) ,sb.toString());
        }
        StringBuilder res = new StringBuilder();
        for(int i = 0;i < s.length();i++){
            if(s.charAt(i) != '(' && s.charAt(i) != ')'){
                res.append(s.charAt(i));
            }
        }
        return res.toString();
    }



    public static boolean isDouble(char left, char right) {
        if ((left == '(' && right == ')')) {
            return true;
        }
        return false;
    }

    //322. 零钱兑换
    // 以[1，2，5] 11
    // f(11) = min (f(11-1),f(11-2),f(11-5)) + 1
    public static int coinChange(int[] coins, int amount) {
        int dp[] = new int[amount+1];
        Arrays.fill(dp,amount + 1);
        dp[0] = 0;
        for(int i = 1; i <= amount;i++){
            for(int j = 0;j < coins.length;j++){
                if(i - coins[j] >= 0)
                dp[i] = Math.min(dp[i], dp[i - coins[j]] + 1);
            }
        }
        return dp[amount] > amount ? -1 : dp[amount];
    }

    //718. 最长重复子数组 LCS
    //int [] arr = new int[]{1,2,3,2,1};
    //        int [] arr1 = new int[]{3,2,1,5,1};
    public static int findLength(int[] A, int[] B) {
        int len1 = A.length;
        int len2 = B.length;
        int dp[][] = new int[len1][len2];
        for(int i = 0;i < len1;i++){
            if(A[i] == B[0]){
                dp[i][0] = 1;
            }
        }
        for(int j = 0;j < len2;j++){
            if(B[j] == A[0]){
                dp[0][j] = 1;
            }
        }
        int max = 0;
        for(int i = 1;i < len1;i++){
            for(int j = 1;j < len2;j++){
                if(A[i] == B[j]){
                    dp[i][j] = dp[i-1][j-1] + 1;
                    max = Math.max(dp[i][j],max);
                }
            }
        }
        return max;
    }

    //300. 最长上升子序列 [10,9,2,5,3,7,101,18] LIS
    // 当i到达7的时候  dp数组 1 1 1 2 2 3   每次i更新位置 都从前往后再更新一次dp数组
    public static int lengthOfLIS(int[] nums) {
        int len = nums.length;
        if(len <= 1)
            return len;
        int dp[] = new int[len];
        for(int i = 0;i < len;i++){
            dp[i] = 1;
        }
        int max = 0;
        for(int i = 1;i < len;i++){
            for(int j = 0;j < i;j++){
                if(nums[j] < nums[i]){
                    dp[i] = Math.max(dp[i],dp[j]+1);
                }
            }
        }
        for(int i = 0;i < len;i++){
            if(dp[i] > max){
                max = dp[i];
            }
        }
        return max;
    }

    //674. 最长连续递增序列
    //输入: [1,3,5,4,7]
    //输出: 3
    public static int findLengthOfLCIS(int[] nums) {
        int len = nums.length;
        if(len <= 1)
            return len;
        int dp[] = new int[len];
        for(int i = 0;i < len;i++){
            dp[i] = 1;
        }
        int max = 1;
        for(int i = 1;i < len;i++){
            if(nums[i] > nums[i-1]){
                dp[i] = dp[i-1] + 1;
                max = Math.max(dp[i], max);
            }
            else{
                dp[i] = 1;
            }
        }
        return max;
    }

    //150. 逆波兰表达式求值
    public static int evalRPN(String[] tokens) {
        Stack<Integer> stack = new Stack<>();
        for(int i = 0;i < tokens.length;i++){
            if(tokens[i].length() == 1){
                char ch = tokens[i].charAt(0);
                if (ch != '+' && ch != '-' && ch != '*' && ch != '/'){
                    stack.add(Integer.valueOf(tokens[i]));
                }
                else{
                    if (stack.size() < 2)
                        return 0;
                    int num2 = stack.pop();
                    int num1 = stack.pop();
                    switch(ch){
                        case '+':
                            stack.push(num1 + num2);
                            break;
                        case '-':
                            stack.push(num1 - num2);
                            break;
                        case '*':
                            stack.push(num1 * num2);
                            break;
                        case '/':
                            stack.push(num1 / num2);
                            break;
                    }
                }
            }
            else {
                stack.push(Integer.valueOf(tokens[i]));
            }
        }
        return stack.pop();
    }

    //371. 两整数之和（不用加减法）
    public int getSum(int a, int b) {
        while (b != 0){
            int temp = a ^ b;
            b = (a & b) << 1;
            a= temp;
        }
        return a;
    }

    //621. 任务调度器 输入: tasks = ["A","A","A","B","B","B"], n = 2
    //输出: 8
    //执行顺序: A -> B -> (待命) -> A -> B -> (待命) -> A -> B.
    public static int leastInterval(char[] tasks, int n) {
        /*
        假设数组 ["A","A","A","B","B","C"]，n = 2，A的频率最高，记为count = 3，所以两个A之间必须间隔2个任务，才能满足题意并且是最短时间
        （两个A的间隔大于2的总时间必然不是最短），因此执行顺序为： A->X->X->A->X->X->A，这里的X表示除了A以外其他字母，或者是待命，不用关心具体是什么，
        反正用来填充两个A的间隔的。上面执行顺序的规律是： 有count - 1个A，其中每个A需要搭配n个X，再加上最后一个A，所以总时间为 (count - 1) * (n + 1) + 1
        要注意可能会出现多个频率相同且都是最高的任务，比如 ["A","A","A","B","B","B","C","C"]，所以最后会剩下一个A和一个B，
        因此最后要加上频率最高的不同任务的个数 maxCount
        公式算出的值可能会比数组的长度小，如["A","A","B","B"]，n = 0，此时要取数组的长度
         */
        int[] count = new int[26];
        for (int i = 0; i < tasks.length; i++) {
            count[tasks[i]-'A']++;
        }//统计词频
        Arrays.sort(count);//词频排序，升序排序，count[25]是频率最高的
        int maxCount = 0;
        //统计有多少个频率最高的字母
        for (int i = 25; i >= 0; i--) {
            if(count[i] != count[25]){
                break;
            }
            maxCount++;
        }
        //公式算出的值可能会比数组的长度小，取两者中最大的那个
        return Math.max((count[25] - 1) * (n + 1) + maxCount , tasks.length);

    }

    //151. 翻转字符串里的单词
    public static String reverseWords(String s) {
        if(s == null || s.length() <= 0)
            return "";
        if(s.trim().equals("")){
            return "";
        }
        String [] strArr = s.split(" ");
        StringBuffer sb = new StringBuffer();
        for(int i = 0;i < strArr.length;i++){
            StringBuffer temp = new StringBuffer(strArr[i].trim());
            if(temp.length() == 0)
                continue;
            temp.reverse();
            sb.append(temp);
            sb.append(" ");
        }
        sb.reverse();
        return sb.toString().substring(1);
    }

    //695. 岛屿的最大面积
    public int maxAreaOfIsland(int[][] grid) {

        if(grid == null || grid.length <=0){
            return 0;
        }
        int rows = grid.length;
        int cols = grid[0].length;

        int max = 0;
        for(int i = 0;i < rows;i++){
            for(int j = 0;j < cols;j++){
                if(grid[i][j] == 1){
                    int area = maxAreaOfIslandSub(grid,i,j,rows,cols);
                    if(area > max){
                        max = area;
                    }
                }
            }
        }
        return max;
    }

    private int maxAreaOfIslandSub(int[][] grid, int row, int col, int rows, int cols) {
        int area = 1;
        if(row < 0 || col < 0 || row >=rows || col >= cols || grid[row][col] == 0){
            return 0;
        }
        grid[row][col] = 0;
        area = area + maxAreaOfIslandSub(grid,row+1,col,rows,cols);
        area = area + maxAreaOfIslandSub(grid,row-1,col,rows,cols);
        area = area + maxAreaOfIslandSub(grid,row,col+1,rows,cols);
        area = area + maxAreaOfIslandSub(grid,row,col-1,rows,cols);
        return area;
    }

    //567. 字符串的排列
    //s1 = "ab" s2 = "eidbaooo"
    public static boolean checkInclusion(String s1, String s2) {
        if(s2.indexOf(s1) > -1){
            return true;
        }
        if(s1.length() > s2.length()){
            return false;
        }

        char[] s1Arr = s1.toCharArray();
        char[] s2Arr = s2.toCharArray();
        // 申请一个散列，用于记录窗口中具体元素的个数情况
        // 这里用数组的形式呈现，也可以考虑其他数据结构
        int[] hash = new int[26];

        // 预处理(可省), 一般情况是改变 hash
        for(int i = 0;i < s1Arr.length;i++){
            hash[s1Arr[i] - 'a']++;
        }

        // l 表示左指针
        // count 记录当前的条件，具体根据题目要求来定义
        // result 用来存放结果
        int left = 0;
        int count = 0;
        for (int right = 0; right < s2Arr.length; ++right) {
            // 更新新元素在散列中的数量
            hash[s2Arr[right] - 'a']--;

            // 根据窗口的变更结果来改变条件值
            if (hash[s2Arr[right] - 'a'] >= 0) {
                count++;
            }

            // 如果当前条件不满足，移动左指针直至条件满足为止
            if (right >= s1Arr.length) {
                hash[s2Arr[left] - 'a']++;
                if (hash[s2Arr[left] - 'a'] >= 1) {//说明s1中有，但是滑动窗口滑出去了
                    count--;
                }
                left++;
            }
            if (count == s1.length()) {
                return true;
            }
         }
         return false;
    }

    //438. 找到字符串中所有字母异位词
    public List<Integer> findAnagrams(String s, String p) {
        List<Integer> results = new ArrayList<>();
        if(s.length() < p.length()){
            return results;
        }
        char[] sArr = s.toCharArray();
        char[] pArr = p.toCharArray();

        int[] hash = new int[26];
        for(int i = 0;i < pArr.length;i++){
            hash[pArr[i] - 'a']++;
        }
        int left = 0;
        int count = 0;
        for (int right = 0; right < sArr.length; ++right) {
            // 更新新元素在散列中的数量
            hash[sArr[right] - 'a']--;

            // 根据窗口的变更结果来改变条件值
            if (hash[sArr[right] - 'a'] >= 0) {
                count++;
            }

            // 如果当前条件不满足，移动左指针直至条件满足为止
            if (right >= pArr.length) {
                hash[sArr[left] - 'a']++;
                if (hash[sArr[left] - 'a'] >= 1) {//说明s1中有，但是滑动窗口滑出去了
                    count--;
                }
                left++;
            }
            if (count == pArr.length) {
                results.add(left);
            }
        }
        return results;
    }

    //76. 最小覆盖子串
    /*
    输入: S = "ADOBECODEBANC", T = "ABC" 输出: "BANC"
     */
    public String minWindow(String s, String t) {
        String resutls = "";
        if(s.length() < t.length()){
            return resutls;
        }
        char[] sArr = s.toCharArray();
        char[] tArr = t.toCharArray();

        int[] hash = new int[256];
        for (int i = 0; i < tArr.length; ++i) {
            hash[tArr[i]]++;
        }

        int left = 0;
        int min = s.length() + 1;//随便找个大点的数
        int count = 0;
        for(int right = 0;right < sArr.length;right++){
            hash[sArr[right]]--;
            if(hash[sArr[right]] >= 0){
                count++;
            }
            while (left < right && hash[sArr[left]] < 0){//示例中 right = BANC中的A的时候 left 才会动起来，直接来到BANC的B的位置
                hash[sArr[left]]++;//之所以不动count，是因为走的是A，进的也是A
                left++;
            }
            if(count == tArr.length && right - left + 1 < min){
                min = right - left + 1;
                resutls = s.substring(left, right + 1);
            }
        }
        return resutls;
    }

    //164.
    /*
    最大间距 [1,3,5,7,9] 划分为5个区间，出去min max 各占2个区间，如果剩下的三个数等划分，每个区间一个数字，那么第一个区间那里，1，3 肯定是大于等于
    桶的长度，如果不能均匀的落在剩下的三个区间内，则必有空区间，空的那个位置长度必然大于桶长度了
    [1,3][3,5][5,7][7,9][9,~]
     */
    public static int maximumGap(int[] nums) {
        if(nums == null || nums.length < 2){
            return 0;
        }

        int min = Integer.MAX_VALUE;
        int max = Integer.MIN_VALUE;
        for(int i : nums){
            min = Math.min(min,i);
            max = Math.max(max,i);
        }

        int BucketLen = Math.max(1,(max - min) / (nums.length - 1));

        Bucket[] buckets = new Bucket[(max - min) / BucketLen + 1];

        for(int i = 0;i < nums.length;i++){
            int index = (nums[i] - min) / BucketLen;

            if(buckets[index] == null){
                buckets[index] = new Bucket();
            }

            buckets[index].min = Math.min(nums[i], buckets[index].min);
            buckets[index].max = Math.max(nums[i], buckets[index].max);
        }

        int previousMax = Integer.MAX_VALUE;
        int maxGap = Integer.MIN_VALUE;
        for(int i = 0;i < buckets.length;i++){
            if (buckets[i] != null && previousMax != Integer.MAX_VALUE) {
                maxGap = Math.max(maxGap, buckets[i].min - previousMax);
            }
            if (buckets[i] != null) {
                previousMax = buckets[i].max;
                maxGap = Math.max(maxGap, buckets[i].max - buckets[i].min);
            }
        }
        return maxGap;
    }

    //171. Excel表列序号
    //其实就是26进制 所以每遍历一位则ans = ans * 26 + num
    //以ZY为例，Z的值为26，Y的值为25，则结果为26 * 26 + 25=701
    //YZX 25 * 26 * 26 + 26 * 26 + 24
    public static int titleToNumber(String s) {
        Character[] nums = {'0','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z' };
        List<Character> list = Arrays.asList(nums);
        int temp = 0;
        for(int i = 0;i < s.length();i++){
            temp += Math.pow(26,s.length()-1-i)  * list.indexOf(s.charAt(i));
            //temp = temp * 26  + s.charAt(i) - 'A' + 1;
        }
        return temp;
    }

    //128 最长连续序列 [100, 4, 200, 1, 3, 2]
    public int longestConsecutive(int[] nums) {
        if(nums == null || nums.length == 0){
            return 0;
        }
        int max = 0;
        Set<Integer> set = new HashSet<>();
        for(Integer num : nums){
            set.add(num);
        }
        for(int i = 0;i < nums.length;i++){
            if(!set.contains(nums[i] - 1)){//如果当前值前面还有值就没判断的必要了，如 2 前面有1，最长绝对不可能从2开始
                int curNum = nums[i];
                int curMax = 1;
                while (set.contains(curNum + 1)){
                    curNum++;
                    curMax++;
                }
                max = Math.max(max,curMax);
            }
        }
        return max;
    }

    //148. 排序链表 归并排序 找中间节点+断链 再按顺序合并
    public ListNode sortList(ListNode head) {
        return head == null ? null : mergeSort(head);
    }

    private ListNode mergeSort(ListNode head) {
        if(head.next == null){
            return head;
        }
        ListNode slow = head;
        ListNode fast = head;
        ListNode pre = null;
        while (fast != null && fast.next != null){
            pre = slow;
            slow = slow.next;
            fast = fast.next.next;
        }

        pre.next = null;//断链
        ListNode left = mergeSort(head);
        ListNode right = mergeSort(slow);
        return Merge(left,right);
    }

    public ListNode Merge(ListNode list1,ListNode list2) {//合并两个有序链表
        if(list1 == null)
            return list2;
        if(list2 == null)
            return  list1;
        ListNode dummyHead = null;
        if(list1.val <= list2.val){
            dummyHead = list1;
            dummyHead.next = Merge(list1.next,list2);
        }
        else{
            dummyHead = list2;
            dummyHead.next = Merge(list1,list2.next);
        }
        return dummyHead;
    }


    public static class Bucket{
        int min = Integer.MAX_VALUE;
        int max = Integer.MIN_VALUE;
    }

    public class TreeNode {
      int val;
      TreeNode left;
      TreeNode right;
      TreeNode(int x) { val = x; }
     }


    public class ListNode {
        int val;
        ListNode next;
        ListNode(int x) { val = x; }
    }
   
    class Node {
        public int val;
        public Node left;
        public Node right;
        public Node next;

        public Node() {}

        public Node(int _val,Node _left,Node _right,Node _next) {
            val = _val;
            left = _left;
            right = _right;
            next = _next;
        }
    };
}
