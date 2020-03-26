
# key part of the generator is that it uses the keyword yield
# loops infinitely or semiinfinitely or to a set amount
def my_gen(limit):
  data = 1
  while data < limit:
    yield data
    data += 2

# create the generator object
gen_obj = my_gen(100)
value=next(gen_obj)
    
try:
    while value:
        print(value)
        value=next(gen_obj)
except StopIteration as e:
    print("Done")