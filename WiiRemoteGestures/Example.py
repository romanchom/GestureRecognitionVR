import numpy as np

class Example:
    def __init__(self, data, label = 0, padded_width = 1):
        self.label = label

        current_width = data.shape[0]
        self.length = current_width
        
        pad_amount = padded_width - current_width
        self.data = np.pad(data, ((0, pad_amount), (0, 0)), 'edge')

        self.label = label
        
    def to_numpy(examples, augumenter = None):
        count = len(examples)
        shape = np.append(count, examples[0].data.shape)

        data = np.ndarray(shape, 'float32')
        if augumenter == None:
            for i in range(count):
                data[i] = examples[i].data
        else:
            for i in range(count):
                data[i] = augumenter(examples[i].data)
        
        
        labels = np.ndarray(count, 'int32')
        for i in range(count):
            labels[i] = examples[i].label
        
        lengths = np.ndarray(count, 'int32')
        for i in range(count):
            lengths[i] = examples[i].length
        
        return data, labels, lengths
        