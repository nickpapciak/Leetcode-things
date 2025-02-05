# 1249. Minimum Remove to Make Valid Parentheses
class LC1249:
    def minRemoveToMakeValid(self, s: str) -> str:
        x = []
        y = []
        for i, c in enumerate(s):
            if c == '(':
                x.append(i)
            elif c == ')':
                if not x:
                    y.append(i)
                else:
                    x.pop()

        bad = set(x + y)
        res = ""
        for i, c in enumerate(s):
            if i not in bad:
                res += c

        return res


# 314. Binary Tree Vertical Order Traversal
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class LC314:
    def verticalOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        queue = [(0, root)]
        neg_cols = []
        pos_cols = []

        while len(queue) > 0:
            col, node = queue.pop()
            if node is None:
                continue

            if col < 0:
                if -(col + 1) == len(neg_cols):
                    neg_cols.insert(0, [])
                neg_cols[col].append(node.val)
            else:
                if col == len(pos_cols):
                    pos_cols.append([])
                pos_cols[col].append(node.val)

            queue.insert(0, (col - 1, node.left))
            queue.insert(0, (col + 1, node.right))

        return neg_cols + pos_cols


# 680. Valid Palindrome II
class LC680:
    def validPalindrome(self, s: str) -> bool:

        def is_palindrome(x):
            return x == x[::-1]

        # find first mismatch
        for i in range(len(s) // 2):
            j = len(s) - i - 1
            if s[i] != s[j]:
                # either delete char at index i or at index j
                return is_palindrome(s[:i] + s[i + 1:]) or \
                    is_palindrome(s[:j] + s[j + 1:])
        return True


# 1216. Valid Palindrome III
class LC1216:
    def isValidPalindrome(self, s: str, k: int) -> bool:

        dp = [[None for _ in s] for _ in s]

        def minCostPalindrome(i, j):

            if (i >= j):
                return 0

            if dp[i][j] is not None:
                return dp[i][j]

            if s[i] == s[j]:
                dp[i][j] = minCostPalindrome(i + 1, j - 1)
                return dp[i][j]

            dp[i][j] = 1 + min(
                minCostPalindrome(i + 1, j),
                minCostPalindrome(i, j - 1)
            )
            return dp[i][j]

        minCostPalindrome(0, len(s) - 1)
        return dp[0][-1] <= k


# 227. Basic Calculator II
class LC227:
    def calculate(self, s: str) -> int:
        s += '+'
        total = 0
        subtotal = 0
        num = ''
        op = '+'

        for c in s:
            if c == ' ':
                continue
            if c.isdigit():
                num += c
                continue

            num = int(num)
            if op == '+':
                subtotal += num
            elif op == '-':
                subtotal -= num
            elif op == '*':
                subtotal *= num
            elif op == '/':
                subtotal = int(subtotal / num)

            if c in '+-':
                total += subtotal
                subtotal = 0

            op = c
            num = ''

        return total
