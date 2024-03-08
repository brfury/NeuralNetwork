import numpy as np 
import os
import json

class NeuralNetwork:
    
    def __init__(self, learning_rate = 10e-2):
        self.weights = np.array([np.random.randn(), np.random.randn()])
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
        self.count = []
        self.cumulative_errors = []
        for _ in range(100):
            self.count.append('--')
       

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def _sigmoid_deriv(self, x):
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def predict(self, input_vector):
        neuron = np.dot(input_vector, self.weights) + self.bias
        neuron = self._sigmoid(neuron)
        neuron_result = neuron
        return neuron_result

    def _compute_gradients(self, input_vector, target):
        neuron = np.dot(input_vector, self.weights) + self.bias
        neuron = self._sigmoid(neuron)
        prediction = neuron

        derror_dprediction = 2 * (prediction - target)
        dprediction_dlayer1 = self._sigmoid_deriv(neuron)
        dlayer1_dbias = 1
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)

        derror_dbias = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
        )
        derror_dweights = (
            derror_dprediction * dprediction_dlayer1 * dlayer1_dweights
        )

        return derror_dbias, derror_dweights

    def _update_parameters(self, derror_dbias, derror_dweights):
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights = self.weights - (
            derror_dweights * self.learning_rate
        )
    
    def cal_percent_train(self, current_iteration, iterations):
        data = round(current_iteration / iterations * 100)
        if data > 80:
            self.learning_rate *= 0.99

        self.count[int(data) - 1] = '*'
        count = ''.join(self.count)
        self.count[int(data) - 1] = '--'
     
        os.system('cls')
        print(f'{count}')
        print(f'train {data}%')
           

    def train(self, input_vectors, targets, iterations):
        
        for current_iteration in range(iterations):
            # Pick a data instance at random
            random_data_index = np.random.randint(len(input_vectors))

            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]

            # Compute the gradients and update the weights
            derror_dbias, derror_dweights = self._compute_gradients(
                input_vector, target
            )

            self._update_parameters(derror_dbias, derror_dweights)

            # Measure the cumulative error for all the instances
            if current_iteration % 100 == 0:
                
                
                cumulative_error = 0
                self.cal_percent_train(current_iteration, iterations)
                
                # Loop through all the instances to measure the error
                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]

                    prediction = self.predict(data_point)
                    error = np.square(prediction - target)

                    cumulative_error = cumulative_error + error
                self.cumulative_errors.append(cumulative_error)
                
                
        #self.save_model()
        return self.cumulative_errors

    def save_model(self, erro):
        model = {
            'wheights': self.weights.tolist(),
            'bias': self.bias,
            'erro': erro
        }

        with open('wheights.json', 'w') as wheights:
            json.dump(model, wheights)

    def load_model(self, filename="wheights.json"):
        """
        This function loads the weights and bias from a JSON file.

        Args:
            filename (str, optional): The filename of the JSON file containing the weights. Defaults to "wheights.json".

        Returns:
            bool: True if the weights were loaded successfully, False otherwise.
        """
        try:
            with open(filename, 'r') as weights_file:
                model = json.load(weights_file)
                self.weights = np.array(model['wheights'])
                self.bias = model['bias']
                return True
        except FileNotFoundError:
            print(f"Error: File {filename} not found.")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in file {filename}.")
        return False


class Data:
    def __init__(self, num_samples) -> None:
        self.num_samples = num_samples
        

    def _calculate_bmi(self,altura, peso) -> float:
        return  peso / (altura ** 2)
        

    def generate_data_training(self) -> np.ndarray: 
        height = np.random.uniform(1.5, 2, self.num_samples)
        
        Weight = np.random.uniform(0.45, 0.90, self.num_samples)
        bmi = [self._calcular_imc(height[i], Weight[i]) for i in range(self.num_samples)]
        input_vectors = np.column_stack((height, Weight))
        targets = np.array(bmi)
        return input_vectors, targets

input_vectors, targets = Data(10000).gerar_dados()
# Paste the NeuralNetwork class code here
# (and don't forget to add the train method to the class)

import matplotlib.pyplot as plt


learning_rate = 1




def genetic(n):
    sequence = []
    best = float('inf')
    for i in range(n):
        print(f'epoch{i}')
        
        neural_network = NeuralNetwork(learning_rate=learning_rate)

        training_error = neural_network.train(input_vectors, targets, 10000)
        indice = training_error[-1]
        
        if indice < best:
            best = indice
            sequence.append(best)
            neural_network.save_model(best)
        print(indice)

    print(sequence)





#plt.plot(training_error)
#plt.xlabel("Iterations")
#plt.ylabel("Error for all training instances")
#plt.show()
#plt.savefig("cumulative_error.png")





neural_network = NeuralNetwork()
neural_network.load_model()
print(round(neural_network.predict([1.89, 0.67]) *100,2))




