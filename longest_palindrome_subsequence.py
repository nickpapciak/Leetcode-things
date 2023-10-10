# Leetcode #516

class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        n = len(s)
        dp = [n*[0] for i in range(n)]

        # solving by going diagonally 
        # in the upper triangular matrix
        for cols in range(0, n):
            i = 0
            for j in range(cols, n):

                #dp base cases
                if i > j: dp[i][j] = 0
                elif i == j: dp[i][j] = 1

                # dp subproblem 
                elif s[i] == s[j]: dp[i][j] = 2 + dp[i+1][j-1]
                else: dp[i][j] = max(dp[i+1][j], dp[i][j-1])

                i+=1

        return dp[0][n-1]
