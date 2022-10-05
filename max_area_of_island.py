# Leetcode # 695

from typing import List

# recursively find the area of an island given a starting square
def area(grid, i, j):
    if i >= len(grid) or i < 0 or j>=len(grid[0]) or j < 0 or grid[i][j] == 0: 
        return 0
    grid[i][j] = 0 # prevents infinite loops
    return 1 + area(grid, i, j+1) + area(grid, i+1, j) + area(grid, i, j-1) + area(grid, i-1, j)
        
class Solution:
    def maxAreaOfIsland(self, grid: List[List[int]]) -> int:
        # cartesian product of lengths, just goes through both
        areas = [area(grid, i,j) for i in range(len(grid)) for j in range(len(grid[0]))]
        return max(areas)
    
