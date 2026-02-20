##Task 1: List problem
nums = [10,20,33,40,55,60,77,80]
#Print only numbers divisible by 5 using list comprehension.
divisible_by_5 = [num for num in nums if num % 5 ==0]
print(divisible_by_5)

##Task 2: Dictionary problem
marks = {
"A":78,
"B":45,
"C":90,
"D":66
}

for student, mark in marks.items():
    if(mark >=70):
        print(f"{student}")

##Task 3: Function

def analyze_numbers(nums):
    return min(nums), max(nums), sum(nums)/len(nums)

analyze_numbers(nums)


##Task 4: Lambda
data = [("ajith",90),("ram",70),("john",85)]

sorted_data = sorted(data, key=lambda x: x[1],reverse=False)
print(sorted_data)

##MINI AI-TYPE PROBLEM

ages = [12, 45, 67, 23, 89, 34, 22]

new_list = ["Senior" if age>50 else "Adult" for age in ages]
print(new_list)