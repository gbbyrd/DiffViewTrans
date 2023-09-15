import json
import random

yes = [5, 10, 11]

no = yes.copy()

no[1] = 54

print(no)
print(yes)

def what(l):
    l[1] = 54
    return l

no = what(yes)

print(yes)
print(no)