import json
import os

import matplotlib.pyplot as plt
import numpy as np


class NeuralNetwork:

    def __init__(self, learning_rate=10e-4):
        self.weights = np.array([np.random.randn(), np.random.randn()])
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
        self.count = []
        self.cumulative_errors = []

        self.count = ["_" for i in range(100)]

    def _sigmoid(self, x) -> float:
        """
        Função de ativação sigmóide.

        Args:
            x (float): Valor de entrada para a função sigmóide.

        Returns:
            float: Resultado da função sigmóide aplicada a x.
        """
        return 1 / (1 + np.exp(-x))

    def _sigmoid_deriv(self, x) -> float:
        """
        Derivada da função de ativação sigmóide.

        Args:
            x (float): Valor de entrada para a derivada da função sigmóide.

        Returns:
            float: Resultado da derivada da função sigmóide aplicada a x.
        """
        return self._sigmoid(x) * (1 - self._sigmoid(x))

    def predict(self, input_vector: list) -> float:
        """
        Realiza a predição com base no vetor de entrada.

        Args:
            vetor_entrada (list): Vetor de entrada para a predição.

        Returns:
            float: Resultado da predição da Rede Neural.
        """
        neuron = np.dot(input_vector, self.weights) + self.bias
        neuron = self._sigmoid(neuron)
        neuron_result = neuron
        return neuron_result

    def _compute_gradients(self, input_vector, target):
        """
        Calcula os gradientes para atualização dos parâmetros da Rede Neural.

        Args:
            vetor_entrada (list): Vetor de entrada.
            alvo (float): Valor alvo.

        Returns:
            tuple: Gradientes para o bias e os pesos.
        """

        neuron = np.dot(input_vector, self.weights) + self.bias
        neuron = self._sigmoid(neuron)
        prediction = neuron

        derror_dprediction = 2 * (prediction - target)
        dprediction_dlayer1 = self._sigmoid_deriv(neuron)
        dlayer1_dbias = 1
        dlayer1_dweights = (0 * self.weights) + (1 * input_vector)

        derror_dbias = derror_dprediction * dprediction_dlayer1 * dlayer1_dbias
        derror_dweights = derror_dprediction * dprediction_dlayer1 * dlayer1_dweights

        return derror_dbias, derror_dweights

    def _update_parameters(self, derror_dbias: float, derror_dweights: list):
        """
        Atualiza os parâmetros da Rede Neural.

        Args:
            d_error_dbias (float): Gradiente do erro em relação ao bias.
            derror_dweights (list): Gradiente do erro em relação aos pesos.
        """
        self.bias = self.bias - (derror_dbias * self.learning_rate)
        self.weights -= derror_dweights * self.learning_rate

    def cal_percent_train(self, current_iteration: int, iterations: int) -> int:
        """
        Calcula e exibe o percentual de treinamento.

        Args:
            current_iteration: Número da iteração atual.
            iterations (int): Número total de iterações.
        """

        data = round(current_iteration / iterations * 100)

        data = round(current_iteration / iterations * 100)
        progress = "|" * (data // 2) + " " * ((100 - data) // 2)
        print(f"\rProgresso: [{progress}] {data}%", end="", flush=True)

    def train(self, input_vectors: list, targets: list, iterations: int) -> list:
        """
        Treina a Rede Neural.

        Args:
            input_vectors (list): Lista de vetores de entrada.
            targets (list): Lista de valores alvo correspondentes.
            iterations (int): Número de iterações de treinamento.
        Returns:
            : Lista com os erros cumulativos ao longo do treinamento.
        """

        for current_iteration in range(iterations * 100):

            random_data_index = np.random.randint(len(input_vectors))

            input_vector = input_vectors[random_data_index]
            target = targets[random_data_index]

            derror_dbias, derror_dweights = self._compute_gradients(
                input_vector, target
            )

            self._update_parameters(derror_dbias, derror_dweights)

            if current_iteration % 100 == 0:

                cumulative_error = 0
                self.cal_percent_train(current_iteration, iterations * 100)

                for data_instance_index in range(len(input_vectors)):
                    data_point = input_vectors[data_instance_index]
                    target = targets[data_instance_index]

                    prediction = self.predict(data_point)
                    error = np.square(prediction - target)

                    cumulative_error += error
                self.cumulative_errors.append(cumulative_error)

        return self.cumulative_errors

    def save_model(self, erro: float, history: list) -> None:
        """
        Salva o modelo da Rede Neural em um arquivo JSON.

        Args:
            erro (float): Último erro cumulativo obtido durante o treinamento.
            history (list): Lista com o histórico de erros cumulativos.
        """
        model = {
            "wheights": self.weights.tolist(),
            "bias": self.bias,
            "erro": erro,
            "history": history,
        }

        with open("wheights.json", "w") as wheights:
            json.dump(model, wheights)

    def load_model(self, filename="wheights.json") -> bool:
        """
        Carrega um modelo previamente treinado da Rede Neural de
        um arquivo JSON.

        Args:
            filename (str): Nome do arquivo JSON contendo o modelo.

        Returns:
            bool: True se o carregamento for bem-sucedido, False caso contrário.
        """

        try:
            with open(filename, "r") as weights_file:
                model = json.load(weights_file)
                self.weights = np.array(model["wheights"])
                self.bias = model["bias"]
                return True
        except FileNotFoundError:
            print(f"Error: File {filename} not found.")
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON format in file {filename}.")
        return False


class Data:
    """
    Classe para manipulação de dados e geração de conjuntos de treinamento.
    """

    def __init__(self, num_samples=1) -> None:
        self.num_samples = num_samples

    def _calculate_bmi(self, height: float, wheight: float) -> float:
        return wheight / (height**2)

    def generate_data_training(self) -> np.ndarray:
        height = np.random.uniform(1.5, 2, self.num_samples)

        Weight = np.random.uniform(0.45, 0.90, self.num_samples)
        bmi = [
            self._calculate_bmi(height[i], Weight[i]) for i in range(self.num_samples)
        ]
        input_vectors = np.column_stack((height, Weight))
        targets = np.array(bmi)
        return input_vectors, targets

    def error_graphic_generate(self, train_error: str) -> None:
        """
        Gera um gráfico dos erros durante o treinamento.

        Args:
            train_error (str): Nome do arquivo JSON contendo os erros de treinamento.
        """
        with open(train_error, "r") as error:
            error = json.load(error)
            plt.plot(error["history"])
            plt.xlabel("Iterations")
            plt.ylabel("Error for all training instances")
            plt.savefig("graphics/fig")
            plt.show()


def train_with_genetic(
    individuals: int, epochs: int, learning_rate=10e-4, num_data=100000
) -> None:
    """
    Função para treinar uma Rede Neural utilizando um algoritmo genético.

    Esta função realiza o treinamento de uma rede neural utilizando um algoritmo genético,
    onde cada indivíduo representa uma configuração da rede. O treinamento é realizado em
    múltiplos indivíduos por um número especificado de épocas.

    Args:
        individuals (int): O número de indivíduos na população genética.
        epochs (int): O número de épocas de treinamento por indivíduo.

    Returns:
        None

    Exemplo de uso:
        train_with_genetic(individuals=10, epochs=100)
    """

    input_vectors, targets = Data(num_data).generate_data_training()
    sequence = []
    best = float("inf")
    for i in range(individuals):
        print(f"epoch{i}")

        neural_network = NeuralNetwork(learning_rate=learning_rate)
        training_error = neural_network.train(input_vectors, targets, epochs)
        indice = training_error[-1]

        if indice < best:
            best = indice
            sequence.append(best)
            neural_network.save_model(best, training_error)
        print(f" error {indice}")


def main():
    """
    exemplo de como executar os treinos

    train_with_genetic recebe e a quantidade de individuos e as epocas por treinamento
    vc pode passar outro learning_rate e outro num_data se desejar

    """

    train_with_genetic(3, 10)

    # vc pode visualizar o gráfico do melhor inviduo chamando a função
    Data().error_graphic_generate(train_error="wheights.json")


#main()
