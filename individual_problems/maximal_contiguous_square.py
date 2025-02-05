# Leetcode #221 

class Solution:
    def maximalSquare(self, matrix: List[List[str]]) -> int:
        m = len(matrix)
        n = len(matrix[0])
        # adds padding
        dp = [[0]*(n+1) for _ in range(m + 1)]

        # running backwards bottom up from right to left 
        for i in reversed(range(m)):
            for j in reversed(range(n)):
                if matrix[i][j] != "0":
                    dp[i][j] = min(dp[i+1][j], dp[i][j+1], dp[i+1][j+1]) + 1

        # maximum of nested array
        return max(max(dp, key=max))**2
