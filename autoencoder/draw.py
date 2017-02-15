import pickle

def read_pkl(data_path):
    input = open(data_path,'rb')
    a = pickle.load(input)
    input.close()
    return a

import matplotlib.pyplot as plt    

#original = read_pkl("./compare/1.pkl")
#processed = read_pkl("./compare/2.pkl")
#
#plt.figure()
#plt.plot(original)
#plt.savefig("original_1.png", dpi=150, format='png')
#plt.show()
#
#plt.figure()
#plt.plot(processed)
#plt.savefig("autoencoder_1.png", dpi=150, format='png')
#plt.show()
#
#original_2 = read_pkl("./compare/3.pkl")
#processed_2 = read_pkl("./compare/4.pkl")
#
#plt.figure()
#plt.plot(original_2)
#plt.savefig("original_2.png", dpi=150, format='png')
#plt.show()
#
#plt.figure()
#plt.plot(processed_2)
#plt.savefig("autoencoder_2.png", dpi=150, format='png')
#plt.show()

original = read_pkl("./compare/1.pkl")
processed = read_pkl("./compare/2.pkl")

plt.figure()
plt.plot(original, 'b', label="original")
plt.plot(processed, 'r', label="reconstruct")
plt.legend()
plt.savefig("1.png", dpi=150, format='png')

original_2 = read_pkl("./compare/3.pkl")
processed_2 = read_pkl("./compare/4.pkl")

plt.figure()
plt.plot(original_2, 'b', label="original")
plt.plot(processed_2, 'r', label="reconstruct")
plt.legend()
plt.savefig("2.png", dpi=150, format='png')