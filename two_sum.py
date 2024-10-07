
# Leetcode #136 

def twoSum(nums: List[int], target: int) -> List[int]:
    goals = {}
    for (i, num) in enumerate(nums): 
        if num in goals: 
            return [goals[num], i]
        else: 
            goals[target - num] = i
