# README - Rede Neural Simples em Python

Este é um exemplo de implementação de uma Rede Neural Simples em Python, utilizada para realizar predições simples. O objetivo principal desta implementação é demonstrar o funcionamento básico de uma rede neural para problemas de regressão.

## Requisitos

- Python 3.x
- NumPy
- Matplotlib

## Estrutura do Projeto

- `NeuralNetwork.py`: Contém a implementação da classe NeuralNetwork, responsável por criar, treinar e fazer predições com a rede neural.
- `Data.py`: Contém a implementação da classe Data, utilizada para manipulação de dados e geração de conjuntos de treinamento.
- `train.py`: Script principal para treinar a rede neural.

## Utilização

Para treinar a rede neural, você pode executar o script `train.py` fornecendo o número de indivíduos na população genética e o número de épocas de treinamento por indivíduo como argumentos. Por exemplo:

```bash
python train.py 3 10
```

Este comando treinará a rede neural com uma população genética de 3 indivíduos durante 10 épocas cada.

## Funções Adicionais

Além disso, o projeto oferece a possibilidade de visualizar o gráfico dos erros durante o treinamento. Você pode chamar a função `error_graphic_generate` da classe `Data`, passando o nome do arquivo JSON contendo os erros de treinamento. Por exemplo, após o treinamento:

```python
Data().error_graphic_generate(train_error="wheights.json")
```

Esta função gerará um gráfico mostrando a variação do erro ao longo das iterações de treinamento.

## Notas

- Você pode ajustar os parâmetros de treinamento, como a taxa de aprendizado e o tamanho do conjunto de dados, modificando os argumentos passados para a função `train_with_genetic` no script `train.py`.
- Este é apenas um exemplo simples de uma rede neural para fins educacionais. Para problemas mais complexos, considere utilizar bibliotecas como TensorFlow ou PyTorch.

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou pull requests para melhorias, correções de bugs ou novos recursos.

## Licença

Este projeto está licenciado sob a [MIT License](LICENSE).
