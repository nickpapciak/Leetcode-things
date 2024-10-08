
# 1. Valid Parentheses
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
        
        
        
                
        