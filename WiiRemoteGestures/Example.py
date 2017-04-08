import numpy as np



class Example:
    def __init__(self, data, label = 0, padded_width = 1):
        self.label = label

        current_width = data.shape[0]
        self.length = current_width
        
        pad_amount = padded_width - current_width
        self.data = np.pad(data, ((0, pad_amount), (0, 0)), 'edge')

        self.label = label
        
    def to_numpy(data):
        count = len(data)
        shape = np.append(count, data[0].data.shape)

        examples = np.ndarray(shape, 'float32')
        for i in range(count):
            examples[i] = data[i].data
        
        #shape = shape[:-1]
        labels = np.ndarray(count, 'int32')
        for i in range(count):
            labels[i] = data[i].label
        
        lengths = np.ndarray(count, 'int32')
        for i in range(count):
            lengths[i] = data[i].length
        
        return examples, labels, lengths
        