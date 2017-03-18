from SQLBase import SQLBase
from Example import Example
import numpy as np
from Recognizer import Recognizer
import random

from threading import Thread

from kivy.app import App
from GraphViewerWidget import GraphViewerWidget
import numpy as np
from kivy.clock import Clock

class MyApp(App):
    def __init__(self, **kwargs):
        super(MyApp, self).__init__(**kwargs)
        self.train_thread = Thread(target=self.train_nn)
        self.should_exit = False
        self.should_load = True
        self.should_save = True

    def build(self):
        self.graph = GraphViewerWidget()
        self.graph.set_graph("truth", [0, 0, 1, 1], (0, 1, 0))
        self.graph.set_graph("prediction", [0, 1, 1, 0], (1, 0, 0))
        self.train_thread.start()
        return self.graph

    def on_stop(self):
        self.should_exit = True
        self.train_thread.join()

    def train_nn(self):
        no_improvement_limit = 200

        random.seed()
        base = SQLBase('MotionGesture.db')
        recognizer = Recognizer(base.feature_count, base.class_count, base.max_length)
        
        train_set, test_set = base.get_large_sets()

        test_examples, test_labels, test_lengths = Example.to_numpy(test_set)
            
        for i in range(10000):
            batch_size = 200
            random.shuffle(train_set)
            for i in range(0, len(train_set) - batch_size + 1 , batch_size):
                train_examples, train_labels, train_lengths = Example.to_numpy(train_set[i:i+batch_size])
                recognizer.train(train_examples, train_labels, train_lengths)
                if(self.should_exit): break
            
            if(self.should_exit): break
            #cross_entropy, percentage, _ = recognizer.test(test_examples, test_labels, test_lengths)
            cross_entropy, percentage, _, total_pred = recognizer.test(test_examples, test_labels, test_lengths)
            print("Cross entropy {}, percentage correct {}".format(cross_entropy, percentage))
            recognizer.save()
            index = random.randrange(len(test_examples))
            result = total_pred[index]
            labels = test_labels[index]
            
            lab_index = test_set[index].label

            ground_truth = [lab_index == labels[i] for i in range(base.max_length)]
            prediction = result[:, lab_index]
            xs = np.linspace(0.0, 1.0, base.max_length)
            
            interleaved = np.zeros(base.max_length * 2)
            interleaved[0::2] = xs
            interleaved[1::2] = ground_truth
            self.gui_truth = interleaved.tolist()

            interleaved[1::2] = prediction
            self.gui_prediction = interleaved.tolist()

            Clock.schedule_once(self.update_gui)

        #self.nn.export_to_protobuffer("./export")
        

    def update_gui(self, dt):
        self.graph.set_graph("truth", self.gui_truth, (0, 1, 0))
        self.graph.set_graph("prediction", self.gui_prediction, (1, 0, 0))
        #self.graph.set_graph("prediction_new", self.gui_prediction_new, (0, 0, 1))


if __name__ == '__main__':
    app = MyApp()
    app.should_load = False
    app.should_save = True
    app.run()