import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from pylab import *
import scipy.io

layer0 = np.asarray(np.loadtxt('0.txt'))
layer1 = np.asarray(np.loadtxt('1.txt'))

# print 'layer0', layer0, layer0.shape, layer0[36, 0:4]
# print 'layer1', layer1, layer1.shape

img_dict = {}
img_dict['start'] = imread('sleep.png')
img_dict['poo'] = imread('poo.png')
for number in xrange(1,9):
    img_dict[number] = imread(str(number)+'.png')
fig = figure(frameon=False)
ax = fig.add_subplot(111)
ax.xaxis.set_ticklabels([None])
ax.yaxis.set_ticklabels([None])
ax.xaxis.set_ticks([None])
ax.yaxis.set_ticks([None])
axis('off')
# print type(img_dict['start']), img_dict['start'].shape
imshow(img_dict['start'])
ion()
# show()
print img_dict.keys()

model = Word2Vec.load_word2vec_format('word2vec.bin', binary=True, norm_only=False)
# print model.size
model.save('textDict.txt')
wordlist = model[:]
print len(wordlist)
# print model.most_similar(positive=['woman', 'prince'], negative=['man'])

while True:
    word = raw_input("How happy is: ")
    try:
        raw_vector = np.asarray(model[word])
#         print raw_vector
        scipy.io.savemat('word.mat', {'word': raw_vector})
        layer0_output = np.maximum(0,np.dot(raw_vector, layer0))
        layer1_output = np.dot(layer0_output, layer1)
        print layer1_output
        if word=='shit' or word=='poo':
            imshow(img_dict['poo'])
        else:
            imshow(img_dict[(int(round(layer1_output)))])
            
        draw()
    except KeyError:
        print word, ' is not/rarely mentioned in Wikipedia. Please choose something more common...\n'
        
