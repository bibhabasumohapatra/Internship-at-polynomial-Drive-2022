def load_vectors(fname):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    
    n, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = list(map(float, torkens[1:]))
    return data

def create_embedding_matrix(word_index, embedding_dict):
    
    embedding_matrix = np.zeros((len(word_index)+1, 300))
    
    for word, i in word_index.items():
        if word in embedding_dict:
            embedding_matrix[i] = embedding_dict[word]
    
    return embedding_matrix
