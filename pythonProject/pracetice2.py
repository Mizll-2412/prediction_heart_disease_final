# n = 36
# i=2
# while n>1:
#     if n%i == 0:
#         print(i)
#         n=n//i
#     else:
#         i += 1

numbers = [20, 10, -4, 5, 10, 36, -16, 3, 5, 10, -5, 5]
# print(numbers.count(10))
data ={}
for number in numbers:
    if number in data.keys():
        data[number] +=1
    else:
        data[number] =1
for key,value in data.items():
    print(key, "appear", value)

dict1={'a':100, 'b':200, 'c':300}
dict2={'a':300, 'b':200, 'd':400}

result = {}

for k, v in dict1.items():
    if k in dict2.keys():
        dict2[k]+= v
    else:
        dict2[k]=v
print(dict2)


text = "I live in ha noi viet nam, i was born in bac giang in 2003"
for word in text.lower():
    if word in result:
        result[word]+=1
    else:
        result[word]=1
print(result)