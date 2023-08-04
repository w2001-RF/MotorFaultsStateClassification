import numpy as np
import pandas as pd
import tensorflow as tf
#from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mealpy.swarm_based.GWO import GWO_WOA
from keras.models import Sequential
from keras.layers import InputLayer, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.activations import relu, sigmoid, softmax, tanh, elu, gelu, selu, linear, hard_sigmoid, softplus, softsign

class HybridMlp:

    def __init__(self, dataset, n_hidden_nodes, n_func_act, epoch, pop_size):
        self.X_train, self.y_train, self.X_test, self.y_test = dataset[0], dataset[1], dataset[2], dataset[3]
        self.n_hidden_nodes = n_hidden_nodes
        self.n_func_act = n_func_act
        self.epoch = epoch
        self.pop_size = pop_size

        self.n_inputs = self.X_train.shape[1]
        self.model, self.problem_size, self.problem = None, None, None
        self.optimizer, self.solution, self.best_fit = None, None, None

    def create_network(self, solution):
        # Create model
        print("current solution ", solution)
        model = Sequential([
                InputLayer(input_shape=self.X_train[0].shape),
                Conv2D(solution[0], kernel_size=(3, 3), activation=act_fun[solution[4]]),
                MaxPooling2D(pool_size=(2, 2)),
                Conv2D(solution[1], kernel_size=(3, 3), activation=act_fun[solution[5]]),
                MaxPooling2D(pool_size=(2, 2)),
                Flatten(),
                Dense(solution[2], activation=act_fun[solution[6]]),
                Dropout(0.5),
                Dense(solution[3], activation=act_fun[solution[7]]),
                Dropout(0.5),
                Dense(self.y_train[0].size, activation=act_fun[solution[8]])
        ])

        # Compile model
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'],
                      jit_compile=True)
        model.fit(X_train, y_train,
                  shuffle=True,
                  steps_per_epoch=880, #X_train.shape[0],
                  #steps_per_epoch=110,
                  batch_size=8,
                  max_queue_size= 50,
                  epochs=1
                  ,
                  verbose=1)
        self.model = model
        self.problem_size = len(self.n_hidden_nodes)*2 + 1

    def create_problem(self):
        self.problem = {
            "fit_func": self.fitness_function,
            "lb": [200, 200, 300, 600, 0, 0, 0, 0, 4],
            "ub": [600, 600, 600, 800, 5, 5, 5, 5, 10],
            "minmax": "max",
            "log_to": "console",
            "save_population": False
        }

    def decode_solution(self, solution):
        # solution: is a vector.
        # solution = [w11, w21, w31, w12, w22, w32, b1, b2, wh11, wh21, wh12, wh22, wh13, wh23, bo1, bo2, bo3]
        # number of weights = n_inputs * n_hidden_nodes + n_hidden_nodes + n_hidden_nodes * n_outputs + n_outputs
        # we decode the solution into the neural network weights
        # we return the model with the new weight (weight from solution)
        self.create_network(list(map(int, solution)))

    def prediction(self, solution, x_data):
        self.decode_solution(solution)
        print("inside prediction method ", self.model.summary())
        return self.model.predict(x_data)

    def training(self):
        solution = self.n_hidden_nodes.copy()
        solution.extend(self.n_func_act)
        self.create_network(solution)
        self.create_problem()
        self.optimizer = GWO_WOA(self.epoch, self.pop_size)

        term_dict = {
            "max_epoch": 10000, #Maximum Generations / Epochs
            "max_fe": 80000, #Maximum Number of Function Evaluation
            #"max_time": 3600, #(Time Bound): If you want your algorithm to run for a fixed amount of time (e.g., K seconds)
            "max_early_stop": 20 #(Early Stopping): Similar to the idea in training neural networks (stop the program if the global best solution has not improved by epsilon after K epochs).
        }
        self.solution, self.best_fit = self.optimizer.solve(self.problem, termination=term_dict,)

    def fitness_function(self, solution):  # Used in training process
        self.decode_solution(solution)
        yhat = self.model.predict(self.X_train)
        yhat = np.argmax(yhat, axis=-1).astype('int')
        y    = np.argmax(self.y_train, axis=1).astype('int')
        acc  = accuracy_score(y, yhat)
        return acc

    def save(self, file_name):
        if file_name == None:
            self.model.save("model_metaheuristique_v0.h5")
        f = file_name if file_name.find(".h5") != -1 else f +".h5"
        self.model.save(f)

# Hide GPU from visible devices
tf.config.set_visible_devices([], 'GPU')


#activation functions
act_fun = {
    0: relu,
    1: elu,
    2: gelu,
    3: selu,
    4: linear,
    5: tanh,
    6: sigmoid,
    7: hard_sigmoid,
    8: softmax,
    9: softplus,
    10: softsign,
}
# Load the dataset and preprocess it
df = pd.read_csv('./data.csv', delimiter=";")
df['Pixels'] = [np.fromstring(x, dtype=np.uint8, sep=' ').reshape(-1, 240, 320, 3).astype("float32") / 255 for x in df["Pixels"]]
img_array = np.concatenate(df.Pixels)

le = LabelEncoder()
img_labels = le.fit_transform(df['short_circuit_faults'])
img_labels = np_utils.to_categorical(img_labels)

X_train, X_test, y_train, y_test = train_test_split(
    img_array, img_labels, shuffle=True, stratify=img_labels, test_size=0.4, random_state=42
)

dataset = [X_train, y_train, X_test, y_test]

n_hidden_nodes = [270, 211, 376, 600]
n_func_act = [2, 3, 1, 4, 8]
epoch = 1
pop_size = 20

# define model
model = HybridMlp(dataset, n_hidden_nodes, n_func_act, epoch, pop_size)
with tf.device("/GPU:0"):
    # fit model
    model.training()

model.save("model_metaheuristique_v1.h5")
# access to the best model - best set of weights - the final weights of neural network
print(f'best_solution: {list(map(int, model.solution))}')

# evaluate on test set
yhat = model.prediction(solution=model.solution, x_data=X_test)
yhat = np.argmax(yhat, axis=-1).astype('int')
y = np.argmax(y_test, axis=1).astype('int')
acc = accuracy_score(y, yhat)
print('Accuracy: %.3f' % acc)
