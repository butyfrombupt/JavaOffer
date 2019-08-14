import java.util.*;

public class Main {

    public static void main(String[] args) {

        int [] arr = new int[]{-1,2,1,-4};

//        int a[] = twoSum(arr,6);
//        for (int i = 0;i < a.length;i++){
//            System.out.print(a[i]+" ");
//        }

        String s = "()(){[]}";

        System.out.println(isValid(s));
        //System.out.println(Integer.parseInt(s));
    }


    /* 题号 1
    两数之和：[3,2,4] target 6 返回 下标 1 2
    1.暴力 2.一次哈希
     */
    public static int[] twoSum(int[] nums, int target) {
        int result[] = new int[2];
        HashMap<Integer,Integer> map = new HashMap<>();
        for(int i = 0;i < nums.length;i++){
            int temp = target - nums[i];
            if(map.containsKey(temp)){
                result[0] = map.get(temp);
                result[1] = i;
                return result;
            }
            else{
                map.put(nums[i],i);
            }
        }
        throw new IllegalArgumentException("No two sum solution");
    }


    /* 题号 2
    两数相加:两链表相加，输入：(2 -> 4 -> 3) + (5 -> 6 -> 4) 输出：7 -> 0 -> 8
    控制好进位和链表的遍历，注意链表判空的地方
     */
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode head = new ListNode(0);
        ListNode cur = head;
        int addOrNot = 0;
        int sum = 0;
        while (l1 != null || l2 != null){
            int x = l1==null ? 0 :l1.val;
            int y = l2==null ? 0 :l2.val;
            sum = x + y + addOrNot;
            addOrNot = sum / 10;
            cur.next = new ListNode(sum % 10);
            cur = cur.next;
            if(l1!=null){
                l1 = l1.next;
            }
            if(l2!=null){
                l2 = l2.next;
            }
        }
        if(addOrNot > 0){
            cur.next = new ListNode(1);
        }
        return head.next;
    }

    /* 题号 3
    无重复字符的最长子串 如题 输入: "abcabcbb" 输出: 3  解释: 无重复字符的最长子串是 "abc"
    abca 当遇到第二个a 就截断，重新开始，每次判长度
     */
    public int lengthOfLongestSubstring(String s) {
//        int max = 0;
//        StringBuffer sb = new StringBuffer();
//        for(int i = 0;i < s.length();i++){
//            String str = s.substring(i,i+1);
//            if(sb.indexOf(str) == -1){
//                sb.append(str);
//            }
//            else{
//                sb = new StringBuffer(sb.substring(sb.indexOf(str)+1));
//                sb.append(str);
//            }
//            if(max < sb.length()){
//                max = sb.length();
//            }
//        }
//        return max;
        if(s == null || s.length() == 0){
            return 0;
        }
        int max = 1;
        int hash[] = new int [256];
        int left = 0;
        char ch[] = s.toCharArray();
        for(int right = 0;right < s.length();right++){
            hash[ch[right]]++;
            while(hash[ch[right]]!=1){
                hash[ch[left]]--;
                left++;
            }
            max = Math.max(max,right - left + 1);
        }
        return max;
    }

    /* 题号 4
    寻找两个有序数组的中位数 时间复杂度要求 O(log(m + n))
     */
    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        //这题hard 原来的做法 时间复杂度达不到
        return 2.0;
    }



    /* 题号 5
    最长回文字串
    暴力法 每个字串判断一下是不是回文串
    中心扩展法，要注意奇偶情况 aba 和cbba bb也是回文
     */
    public static String longestPalindrome(String s) {
//        int max = 0;
//        String res = "";
//        for(int i = 0;i<s.length();i++){
//            for(int j = i;j < s.length();j++){
//                String str = s.substring(i,j+1);
//                if(str.length() > max && isPalindrome(str)){
//                    max = str.length();
//                    res = str;
//                }
//            }
//        }
//        return res;
        int max = 0;
        String res = "";
        for(int i = 0;i<s.length();i++){
            int len1 = expandAroundCenter(s,i,i+1);
            int len2 = expandAroundCenter(s,i,i);
            int len = Math.max(len1,len2);
            if(len > max){
                res = s.substring(i - (len - 1) / 2,i+1+len/2);
                max = res.length();
            }
        }
        return res;
    }

    private static int expandAroundCenter(String s, int left,int right) {
        int L = left, R = right;
        while (L >= 0 && R < s.length() && s.charAt(L) == s.charAt(R)) {
            L--;
            R++;
        }
        return R - L - 1;
    }

    public static boolean isPalindrome(String s){
        String s1=new StringBuffer(s).reverse().toString();
        return s.equals(s1);
    }

    /* 题号 6
    Z字形变换
    用数学公式写代码是最快的方式，思路就是首位行每次寻找下一个输出字符的步长一直是2*（numRows - 1）,其余行每次需要更新步长
     */
    public String convert(String s, int numRows) {
        String result = "";
        if(numRows == 1 || s.length() <= numRows) return s;
        for(int i = 0;i < numRows;i++)
        {
            int start = i;
            int step;
            if((i == numRows-1) || (i == 0))
            {
                step = 2 * (numRows - 1);
                while(start < s.length())
                {
                    result += s.charAt(start);
                    start += step;
                }
            }
            else
            {
                step = 2 * (numRows -1 -i);
                while(start < s.length())
                {
                    result += s.charAt(start);
                    start += step;
                    step = 2* numRows -step-2;
                }
            }

        }

        return result;
    }

    /* 题号 7
    整数反转 java的方法都封装好了
     */
    public int reverse(int x) {
        String s = String.valueOf(x);
        try{
            if(x >= 0){
                String temp = "";
                for(int i = 0;i < s.length();i++){
                    char c = s.charAt(i);
                    temp = c + temp;
                }
                return Integer.parseInt(temp);
            }
            else{
                String temp = "";
                for(int i = 1;i < s.length();i++){
                    char c = s.charAt(i);
                    temp = c + temp;
                }
                return -Integer.parseInt(temp);
            }
        }catch (Exception e){
            return 0;
        }
    }

    /* 题号 8
    字符串转换整数 (atoi) 这种题没啥意思，体现字符串功底
     */
    public int myAtoi(String str) {//8
        if (str == null)
            return 0;
        str = str.trim();
        char[] chars = str.toCharArray();
        if (chars.length == 0)
            return 0;
        char c0 = chars[0];
        if (c0 != '-' && c0 != '+' && !Character.isDigit(c0))
            return 0;
        if (c0 == '-' || c0 == '+') {
            if (chars.length == 1)
                return 0;
            else if (!Character.isDigit(chars[1]))
                return 0;
        }
        StringBuilder sb = new StringBuilder();
        sb.append(c0);
        int i = 1;
        while (i < chars.length){
            if (!Character.isDigit(chars[i]))
                break;
            sb.append(chars[i]);
            i++;
        }
        int res = 0;
        try {
            res = Integer.parseInt(sb.toString());
        } catch (Exception e) {
            if (c0 == '-')
                return Integer.MIN_VALUE;
            else
                return Integer.MAX_VALUE;
        }
        return res;
    }


    /* 题号 9
    回文数 反转判断即可
     */
    public boolean isPalindrome(int x) {//9
        try{
            String s=new StringBuffer(String.valueOf(x)).reverse().toString();
            return s.equals(String.valueOf(x));
        }catch(Exception e){
            return false;
        }
    }

    /* 题号 11
     盛最多水的容器 ，其实就是下标差值和这两个下标最小的那个值的乘积
     输入: [1,8,6,2,5,4,8,3,7] 输出: 49  （8的下标 1 7的下标 9） 49 = 7*7
     暴力和双指针
     */
    public int maxArea(int[] height) {
        int max = 0;
//        for(int i = 0;i < height.length;i++){
//            for(int j = i + 1;j < height.length;j++){
//                if(Math.min(height[i],height[j]) * (j - i) > max){
//                    max = Math.min(height[i],height[j]) * (j - i);
//                }
//            }
//        }
        int left = 0;
        int right = height.length - 1;
        while (left < right){
            if(Math.min(height[left],height[right]) * (right - left) > max){
                max = Math.min(height[left],height[right]) * (right - left);
            }
            if(height[left] < height[right]){
                left++;
            }
            else{
                right--;
            }
        }

        return max;

    }

    /* 题号 12
    整数转为罗马数字
    输入: 58 输出: "LVIII"
    解释: L = 50, V = 5, III = 3.
     */
    public String intToRoman(int num) {
        String ans="";
        String oneArray[]=new String []{"","I","II","III","IV","V","VI","VII","VIII","IX"};
        String twoArray[]=new String []{"","X","XX","XXX","XL","L","LX","LXX","LXXX","XC"};
        String threeArray[]=new String []{"","C","CC","CCC","CD","D","DC","DCC","DCCC","CM"};
        String fourArray[]=new String []{"","M","MM","MMM"};
        int four=num/1000%10;
        int three=num/100%10;
        int two=num/10%10;
        int one=num%10;
        ans=fourArray[four]+threeArray[three]+twoArray[two]+oneArray[one];
        return ans;
    }

    /* 题号 13
    罗马数字转整数
    左边小话就是大减去小 左边大的话就是相加，注意判断当读到最后一位的情况
     */
    public static int romanToInt(String s) {
        int sum=0;
        Map<String, Integer> RomanMap=new HashMap();
        RomanMap.put("I", 1);
        RomanMap.put("V", 5);
        RomanMap.put("X", 10);
        RomanMap.put("L", 50);
        RomanMap.put("C", 100);
        RomanMap.put("D", 500);
        RomanMap.put("M", 1000);

        for(int i = 0; i < s.length();i++){
            if(i < s.length() - 1 && RomanMap.get(String.valueOf(s.charAt(i))) < RomanMap.get(String.valueOf(s.charAt(i + 1)))){
                sum = sum - RomanMap.get(String.valueOf(s.charAt(i)));
            }
            else{
                sum = sum + RomanMap.get(String.valueOf(s.charAt(i)));
            }
        }

        return sum;
    }

    /* 题号 14
    最长公共前缀
    遍历走起,先找个短的，节省下时间，再已第一个串当基准
     */
    public String longestCommonPrefix(String[] strs) {
        if(strs == null || strs.length == 0){
            return "";
        }
        StringBuilder sb = new StringBuilder();
        int minLen = Integer.MAX_VALUE;
        for(String str : strs){
            minLen = Math.min(str.length(),minLen);
        }

        String base = strs[0];
        for(int i = 0;i < minLen;i++){
            char c = base.charAt(i);
            for(int j = 1;j < strs.length;j++){
                if(strs[j].charAt(i) != c){
                    return sb.toString();
                }
            }
            sb.append(c);
        }
        return sb.toString();
    }

    /* 题号 15
    三数之和 [-1,0,1,2,-1,-4]
    先排序后为【-4，-1，-1，0，1，2】，选基准target转化为两数之和，选target的时候重复的直接跳过，再从基准后面的数开始遍历
    想要加和为0，必然在target的后面，二分即可，做好去重和优化
     */
    public static List<List<Integer>> threeSum(int[] nums) {

        List<List<Integer>> results = new ArrayList<>();

        Arrays.sort(nums);
        for(int i = 0;i < nums.length;i++){
            if(i!=0 && nums[i]==nums[i-1])
                continue;
            if(nums[i] > 0){
                break;
            }
            int target = - nums[i];
            int left = i + 1;
            int right = nums.length - 1;
            while(left < right){
                if(nums[left] + nums[right] == target){
                    results.add(Arrays.asList(nums[i],nums[left],nums[right]));
                    while (left < right && nums[left] == nums[left+1]) {
                        left++; // 去重
                    }
                    while (left < right && nums[right] == nums[right-1]){
                        right--; // 去重
                    }
                    left++;
                    right--;
                }
                else if(nums[left] + nums[right] < target){
                    left++;
                }
                else{
                    right--;
                }
            }
        }
        return results;
    }

    /* 题号 16
    最接近的三数之和
    与上题相似，只不过每次比较一下最小差值并更新
     */
    public static int threeSumClosest(int[] nums, int target) {
        if(nums == null || nums.length < 3)
            return 0;
        Arrays.sort(nums);
        int minDiff = Integer.MAX_VALUE;
        int res = 0;
        for(int i = 0;i < nums.length;i++){
            if(i!=0 && nums[i]==nums[i-1])
                continue;
            int left = i + 1;
            int right = nums.length - 1;
            while(left < right){
                int ThreeSum = nums[left] + nums[right] + nums[i];
                if(ThreeSum == target){
                    return target;
                }
                else if(ThreeSum < target){
                    left++;
                    if(Math.abs(ThreeSum - target) < minDiff){
                        minDiff = Math.abs(ThreeSum - target);
                        res = ThreeSum;
                    }
                }
                else{
                    if(Math.abs(ThreeSum - target) < minDiff){
                        minDiff = Math.abs(ThreeSum - target);
                        res = ThreeSum;
                    }
                    right--;
                }
            }
        }
        return res;
    }

    /* 题号 17
    电话号码的字母组合
    输入："23" 输出：["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].
    这题可以看作是全排列，str数组存a-z 为了与数字 2 3 这种对应，前两个都是空串
    值得注意的是传入的23是String类型 要想和数组下标对齐，需要 - '0'
     */
    public List<String> letterCombinations(String digits) {
        List<String> results = new ArrayList<>();
        if(digits == null || digits.length() == 0){
            return results;
        }
        String[] str = new String[] {"","","abc","def","ghi","jkl","mno","pqrs","tuv","wxyz"};

        StringBuilder sb = new StringBuilder();

        letterCombine(digits,0,str,sb,results);

        return results;
    }

    private void letterCombine(String digits, int index, String[] str, StringBuilder sb, List<String> results) {
        if(index == digits.length()){
            results.add(sb.toString());
            return;
        }
        for(int i = 0;i < str[digits.charAt(index) - '0'].length();i++){
            sb.append(str[digits.charAt(index) - '0'].charAt(i));
            letterCombine(digits,index + 1, str, sb, results);
            sb.deleteCharAt(sb.length() - 1);
        }
    }


    /* 题号18
    四数之和
    注意去重的条件，，保证第一个值能判断，[0,0,0,0] 0
    if(i!=0 && nums[i]==nums[i-1]) 和 if(j > i + 1 && nums[j] == nums[j-1])
     */
    public List<List<Integer>> fourSum(int[] nums, int target) {
        List<List<Integer>> results = new ArrayList<>();
        Arrays.sort(nums);
        for(int i = 0;i < nums.length;i++){
            if(i!=0 && nums[i]==nums[i-1])
                continue;
            int newTarget = target - nums[i];
            for(int j = i+1;j < nums.length;j++){
                if(j > i + 1 && nums[j] == nums[j-1])
                    continue;
                int newstTarget = newTarget - nums[j];
                int left = j + 1;
                int right = nums.length - 1;
                while(left < right){
                    if(nums[left] + nums[right] == newstTarget){
                        results.add(Arrays.asList(nums[i],nums[j],nums[left],nums[right]));
                        while (left < right && nums[left] == nums[left+1]) {
                            left++; // 去重
                        }
                        while (left < right && nums[right] == nums[right-1]){
                            right--; // 去重
                        }
                        left++;
                        right--;
                    }
                    else if(nums[left] + nums[right] < newstTarget){
                        left++;
                    }
                    else{
                        right--;
                    }
                }
            }
        }
        return results;
    }

    /* 题号 19
    删除链表的倒数第N个节点
    快慢指针，注意快指针 到第一个节点，删除的是第一个节，这时候需要直接return了
     */
    public ListNode removeNthFromEnd(ListNode head, int n) {
        if(head == null)
            return null;
        ListNode fast = head;
        ListNode slow = head;
        for(int i = 0;i < n;i++){
            if(fast.next!=null){
                fast = fast.next;
            }
            else{
                return head.next;//到第一个节点啦，删除的是第一个节点
            }
        }
        while (fast.next != null){
            fast = fast.next;
            slow = slow.next;
        }
        slow.next = slow.next.next;
        return head;
    }


    /*题号 20 有效括号
    用到了栈，有匹配的就出栈，没有就入栈，最后判栈空
     */
    public static boolean isValid(String s) {
        Stack<Character> stack = new Stack<>();
        for(int i = 0;i < s.length();i++){
            if(!stack.isEmpty() && isDouble(stack.peek(),s.charAt(i))){
                stack.pop();
            }
            else{
                stack.push(s.charAt(i));
            }
        }
        return stack.isEmpty();
    }

    private static boolean isDouble(Character left, char right) {
        if((left == '(' && right == ')') || (left == '[' && right == ']') || (left == '{' && right =='}')){
            return true;
        }
        return false;
    }

    /* 题号 21 合并两个有序链表、
    有一个没了就直接续上，递归调用
     */
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        if(l1 == null){
            return l2;
        }
        else if(l2 == null){
            return l1;
        }
        ListNode newHead = null;
        if(l1.val < l2.val){
            newHead.val = l1.val;
            newHead.next = mergeTwoLists(l1.next,l2);
        }
        else{
            newHead.val = l2.val;
            newHead.next = mergeTwoLists(l1,l2.next);
        }
        return newHead;
    }

    /* 题号 22 括号生成
    方法一：暴力全排列，再用20题过滤
    方法二：通过对左右括号的计数实现生成有效的括号，左括号数 < n 还可以生成左括号， 而右括号数必须小于左括号数才能生成
     */
    public List<String> generateParenthesis(int n) {
        List<String> list = new ArrayList<>();
        generateParenthesisSub(list,"",0,0,n);
        return list;
    }

    private void generateParenthesisSub(List<String> list, String cur, int left, int right, int n) {
        if(cur.length() == n * 2){
            list.add(cur);
            return;
        }
        if(left < n){
            generateParenthesisSub(list,cur+"(",left+1,right,n);
        }
        if(right < left){
            generateParenthesisSub(list,cur+")",left,right+1,n);
        }
    }

    /* 题号 23 合并k个有序链表
    暴力法 和 最小堆
     */
    public ListNode mergeKLists(ListNode[] lists) {
//        ArrayList<ListNode> arr = new ArrayList<>();
//        for(int i = 0;i < lists.length;i++) {
//            ListNode node = lists[i];
//            while(node != null) {
//                arr.add(node);
//                node = node.next;
//            }
//        }
//        arr.sort(new Comparator<ListNode>() {
//            public int compare(ListNode o1, ListNode o2) {
//                return o1.val - o2.val;
//            }
//        });
//
//
//        ListNode head = new ListNode(1);
//        ListNode cur = head;
//        for(int i = 0;i < arr.size();i++){
//            cur.next = arr.get(i);
//            cur = cur.next;
//        }
//        return head.next;
        if (lists == null || lists.length == 0)
            return null;
        ListNode dummy = new ListNode(-1);
        ListNode cur = dummy;
        PriorityQueue<ListNode> queue =
                new PriorityQueue<>(lists.length, new Comparator<ListNode>() {
                    public int compare(ListNode o1, ListNode o2) {
                        return o1.val - o2.val;
                    }
                });

        for (ListNode list : lists) {
            if (list != null)
                queue.add(list);
        }
        while (!queue.isEmpty()) {
            cur.next = queue.poll();
            cur = cur.next;
            if (cur.next != null)
                queue.add(cur.next);
        }
        return dummy.next;
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
        List<Integer> list = new ArrayList<Integer>();
        if(root == null)
            return list;
        Stack<TreeNode> stack = new Stack<>();
        while(root != null && !stack.isEmpty()){
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
}
