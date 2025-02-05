
# 1. Two Sum
class LC1:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        goals = {}
        for (i, num) in enumerate(nums):
            if num in goals:
                return [goals[num], i]
            else:
                goals[target - num] = i






# 20. Valid Parentheses
class LC20:
    def isValid(self, s: str) -> bool:
        reverse = {')' : '(',
                   '}' : '{',
                   ']' : '['}
        stack = []
        for c in s:
            if c in reverse:
                if len(stack) == 0 or stack[-1] != reverse[c]: return False
                del stack[-1]
            else:
                stack.append(c)

        return len(stack) == 0







# 21. Merge Two Sorted Lists
# Definition for  singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class LC21:
    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        head = ListNode()
        tail = head

        while (list1 is not None and list2 is not None):
            a, b = list1.val, list2.val
            if a <= b:
                tail.next = ListNode(a, None)
                list1 = list1.next
            else:
                tail.next = ListNode(b, None)
                list2 = list2.next
            tail = tail.next

        while (list1 is not None):
            tail.next = ListNode(list1.val, None)
            list1 = list1.next
            tail = tail.next

        while (list2 is not None):
            tail.next = ListNode(list2.val, None)
            list2 = list2.next
            tail = tail.next

        return head.next








# 121. Best Time to Buy and Sell Stock
class LC121:
    def maxProfit(self, prices: List[int]) -> int:
        min_price = prices[0]
        max_profit = 0

        for price in prices[1:]:
            max_profit = max(price - min_price, max_profit)
            min_price = min(price, min_price)

        return max_profit






# 125. Valid Palindrome
class LC125:
    def isPalindrome(self, s: str) -> bool:
        # not a readable solution, just a fun one
        return  (t := [c.lower() for c in s if c.isalnum()]) == t[::-1]







# 226. Invert Binary Tree
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class LC226:

    def invertTree(self, root: Optional[TreeNode]) -> Optional[TreeNode]:
        if not root:
            return None

        left = root.left
        self.invertTree(root.left)
        self.invertTree(root.right)
        root.left = root.right
        root.right = left

        return root







# 242. Valid Anagram
class LC242:
    def isAnagram(self, s: str, t: str) -> bool:
        smap = {}
        tmap = {}
        for c in s:
            smap.setdefault(c, 0)
            smap[c] += 1
        for c in t:
            tmap.setdefault(c, 0)
            tmap[c] += 1

        return smap == tmap






# 704. Binary Search
class LC704:
    def recursive_search(self, left, right, nums, target):
        if left >= right:
            return left if nums[left] == target else -1

        mid = (left + right)//2
        if nums[mid] == target: return mid
        if nums[mid] < target: return self.recursive_search(mid + 1, right, nums, target)
        else: return self.recursive_search(left, mid - 1, nums, target)

    def search(self, nums: List[int], target: int) -> int:
        return self.recursive_search(0, len(nums) - 1, nums, target)






# 733. Flood Fill
class LC733:
    def floodFill(self, image: List[List[int]], sr: int, sc: int, color: int) -> List[List[int]]:
        m, n = len(image), len(image[0])
        start_color = image[sr][sc]
        visited = set()

        def dfs(i, j):
            if not (i >= 0 and i < m and j >= 0 and j < n):
                return
            if image[i][j] != start_color:
                return
            if (i, j) in visited:
                return
            visited.add((i, j))
            image[i][j] = color
            dirs = [(0, 1), (1, 0), (0, -1), (-1, 0)]
            for offset_row, offset_col in dirs:
                dfs(i + offset_row, j + offset_col)

        dfs(sr, sc)
        return image






# 235. Lowest Common Ancestor of a Binary Search Tree
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None
class LC235:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        p_ancestors = set()
        q_ancestors = set()

        def recursiveLCA(node, depth = 0):
            if not node: return

            a = recursiveLCA(node.left, depth + 1)
            b = recursiveLCA(node.right, depth + 1)
            if a:
                return a
            if b:
                return b

            ancestor_is_P = node.left in p_ancestors or node.right in p_ancestors or node == p
            ancestor_is_Q = node.left in q_ancestors or node.right in q_ancestors or node == q

            if ancestor_is_P:
                p_ancestors.add(node)
            if ancestor_is_Q:
                q_ancestors.add(node)

            if ancestor_is_P and ancestor_is_Q:
                return node

        return recursiveLCA(root)







# 110. Balanced Binary Tree
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class LC110:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:
        def recIsBalanced(node):
            if not node: return 0, True

            l_depth, isLeftBalanced = recIsBalanced(node.left)
            r_depth, isRightBalanced = recIsBalanced(node.right)

            isBalanced = (l_depth - r_depth)**2 <= 1

            return max(l_depth, r_depth) + 1, isLeftBalanced and isRightBalanced and isBalanced

        return recIsBalanced(root)[1]








# 141. Linked List Cycle
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None
class LC141:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        seen = set()

        while head:
            if head in seen:
                return True
            seen.add(head)
            head = head.next
        return False







# 232. Implement Queue using Stacks
class LC232:
    def __init__(self):
        self.pushing_stack = []
        self.popping_stack = []


    def push(self, x: int) -> None:
        self.pushing_stack.append(x)

    def pop(self) -> int:
        self.peek()
        return self.popping_stack.pop()


    def peek(self) -> int:
        if not self.popping_stack:
            while self.pushing_stack:
                self.popping_stack.append(self.pushing_stack.pop())
        return self.popping_stack[-1]


    def empty(self) -> bool:
        return not self.pushing_stack and not self.popping_stack

# Your MyQueue object will be instantiated and called as such:
# obj = MyQueue()
# obj.push(x)
# param_2 = obj.pop()
# param_3 = obj.peek()
# param_4 = obj.empty()









# 278. First Bad Version
# The isBadVersion API is already defined for you.
# def isBadVersion(version: int) -> bool:
class LC278:
    def firstBadVersion(self, n: int) -> int:
        def bin_search(left = 1, right = n):
            if left == right:
                return left

            mid = (left + right)//2
            if isBadVersion(mid): return bin_search(left, mid)
            else: return bin_search(mid + 1, right)

        return bin_search()








# 383. Ransom Note
class LC383:
    def canConstruct(self, ransomNote: str, magazine: str) -> bool:
        table = {}
        for letter in magazine:
            table.setdefault(letter, 0)
            table[letter] += 1

        for letter in ransomNote:
            if letter not in table: return False
            if table[letter] == 0: return False
            table[letter] -= 1
        return True







# 70. Climbing Stairs
class LC70:
    def climbStairs(self, n: int) -> int:
        if n == 1: return 1
        if n == 2: return 2

        dp = [0]*(n+1)
        dp[0] = 1
        dp[1] = 1
        dp[2] = 2
        for i in range(2, n+1):
            dp[i] = dp[i-1] + dp[i-2]
        return dp[n]






# 409. Longest Palindrome
class LC409:
    def longestPalindrome(self, s: str) -> int:
        table = {}
        for c in s:
            table.setdefault(c, 0)
            table[c] += 1

        res = 0
        odds = [val - 1 for val in table.values() if val % 2 == 1]
        evens = [val for val in table.values() if val % 2 == 0]
        if odds: res += 1

        res += sum(evens)
        res += sum(odds)
        return res









# 206. Reverse Linked List
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class LC206:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:

        def rec_rev(prev, node):

            if not node:
                return prev

            tail = rec_rev(node, node.next)
            node.next = prev
            return tail

        return rec_rev(None, head)







# 169. Majority Element
class LC169:
    def majorityElement(self, nums: List[int]) -> int:
        candidate = None
        count = 0

        for num in nums:
            if count == 0:
                candidate = num
            if num == candidate:
                count += 1
            else:
                count -= 1

        return candidate









# 67. Add Binary
class LC67:
    def addBinary(self, a: str, b: str) -> str:
        a, b = list(a), list(b)
        n = max(len(a), len(b))
        a = (n - len(a))*["0"] + a
        b = (n - len(b))*["0"] + b

        def xor(x, y, c):
            if x == "1" and y == "1":
                return "1" if c else "0", True
            if  x == "0" and y == "0":
                return "1" if c else "0", False
            return "0" if c else "1", c

        carry = False
        res = ""
        while a and b:
            t, nextcarry = xor(a.pop(), b.pop(), carry)
            res = t + res
            carry = nextcarry

        return ("1" if carry else "") + res








# 543. Diameter of Binary Tree
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class LC543:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:

        def depth(node):
            if not node: return 0, 0
            depth_l, diam_l = depth(node.left)
            depth_r, diam_r = depth(node.right)

            return max(depth_l, depth_r) + 1, max(diam_l, diam_r, depth_l + depth_r)

        return depth(root)[1]







# 876. Middle of the Linked List
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class LC876:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        hare = head

        while hare and hare.next:
            hare = hare.next.next
            head = head.next
        return head

# 53. Maximum Subarray
class LC53:
    def maxSubArray(self, nums: List[int]) -> int:
        c_sum = 0
        running_max = 0
        for num in nums:
            c_sum = max(0, c_sum + num)
            running_max = max(running_max, c_sum)

        return running_max or max(nums)



# 57. Insert Interval
class LC57:
    # binary search
    def search(self, nums: List[List[int]], target: int, start: bool) -> int:
        left = 0
        right = len(nums)-1

        while left <= right:
            mid = (left + right) // 2
            # searches through "start" endpoints
            # whenever the new interval endpoint
            # we are looking at is an "end"

            # searches through "end" endpoints
            # whenever the new interval endpoint
            # we are looking at is a "start"
            if nums[mid][start] == target:
                return mid
            elif nums[mid][start] <= target:
                left = mid + 1
            else:
                right = mid - 1

        # if looking for start interval of merge:
        #     least upper bound of end times
        # if looking for end interval of merge:
        #     greatest lower bound for start times
        return left if start else right


    def insert(self, intervals: List[List[int]], newInterval: List[int]) -> List[List[int]]:
        start_of_merge = self.search(intervals, newInterval[0], start=True)
        end_of_merge = self.search(intervals, newInterval[1], start=False)


        # finds new interval if start/end in range
        if not (start_of_merge >= len(intervals) or end_of_merge < 0):
            newInterval[0] = min(intervals[start_of_merge][0], newInterval[0])
            newInterval[1] = max(intervals[end_of_merge][1], newInterval[1])

        # lowk fire solution
        return intervals[:start_of_merge] + [newInterval] + intervals[end_of_merge + 1:]



# 542. 01 Matrix
class LC542:
    def updateMatrix(self, mat: List[List[int]]) -> List[List[int]]:
        m = len(mat)
        n = len(mat[0])
        # deepcopy
        dp = [[c for c in row] for row in mat]

        # normally
        # dp[i][j] = 1 + min(
        #     dp[i+1][j],
        #     dp[i-1][j],
        #     dp[i][j+1],
        #     dp[i][j-1]
        # )
        # but we can't solve DP that way. However, notice that
        # each valid path either uses only up & left moves
        # or only uses down and right moves

        # first pass
        for row in range(m):
            for col in range(n):
                up = dp[row - 1][col] if row > 0 else inf
                left = dp[row][col - 1] if col > 0 else inf
                if dp[row][col]:
                    dp[row][col] = min(up, left) + 1

        for row in range(m - 1, -1, -1):
            for col in range(n - 1, -1, -1):
                down = dp[row + 1][col] if row < m - 1 else inf
                right = dp[row][col + 1] if col < n - 1 else inf

                if dp[row][col]:
                    dp[row][col] = min(dp[row][col], min(down, right) + 1)

        return dp
