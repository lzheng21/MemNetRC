import csv
from nltk.corpus import stopwords
import re
import pickle
import numpy

def doc_array(file,doc_max_len):
    #Get a doc matrix and word2id dictionary
    doc2content = {}
    f = open(file,'rb')
    word2id = {}
    word_id = 1
    reader = csv.reader(f)
    for line in reader:
        doc_id,title,abstract = line[0],line[3],line[4]
        try:
            doc_id = int(doc_id)
            if doc_id != 12479:
                content = title + '. ' + abstract
            else:
                content = title
            content = content.decode('ISO-8859-1')
            text = content
            text = [w for w in re.split('\W', text) if w]
            text = [word.lower() for word in text if word.lower() not in (stopwords.words('english'))]
            words = []
            for j in xrange(len(text)):
                if text[j] not in word2id:
                    word2id[text[j]] = word_id
                    words.append(word_id)
                    word_id += 1
                else:
                    words.append(word2id[text[j]])

            #padding or truncate
            if len(words) >= doc_max_len:
                words = words[:doc_max_len]
            else:
                words = (doc_max_len-len(words))*[0] + words

            doc2content[doc_id] = words
        except ValueError:
            pass

    f.close()
    max_doc_id = max(doc2content.keys())
    doc_array = numpy.zeros([max_doc_id+1,doc_max_len],dtype=numpy.int32)
    for doc in doc2content.keys():
        doc_array[doc] = doc2content[doc]

    return doc_array
    #f = open('doc_array', 'wb')
    #pickle.dump(doc_array, f)
    #f.close()