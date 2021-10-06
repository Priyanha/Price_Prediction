import pickle

pickle_in = open("backend/assets/files/text.pickle", "rb")
example_dict = pickle.load(pickle_in)

print("hello")
print(example_dict)