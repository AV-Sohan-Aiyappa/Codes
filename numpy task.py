import numpy as np 

arr1 = np.array([3,7,1])
arr2 = np.tile(arr1 , (1,3,4))

print("Original array:",arr1)
print("Repeating the array across dimensions (3,4):\n",arr2)



a = np.random.randint(1,100,20)
sorted_array = np.sort(a)
indices = np.argsort(a)
print("Original Array:",a)
print("Array elements in ascending order: \n",sorted_array)
print("Indices of elements in original array when elements are sorted in ascending order:\n ",indices)