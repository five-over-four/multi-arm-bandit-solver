# this script generates n probabilities between 0 and 1.

while True:
    try:
        n = int(input("how many machines? "))
        from random import random
        with open("data.txt", "w") as file:
            file.write(",".join([str(random()) for x in range(n)])[:-1])
        print("done. goodbye.")
        break
    except:
        print("faulty input. try again.")