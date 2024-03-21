import streamlit as st
import matplotlib.pyplot as plt
import matplotlib as mpl
from model import NeuralNetwork
import io

def generate_fig(value):
    
    print(value)
    # Criar a figura e os eixos
    fig, ax = plt.subplots(figsize=(8, 0.5), subplot_kw={'clip_on': False})

    # Configurar a barra de cores
    cmap = mpl.cm.hsv
    norm = mpl.colors.Normalize(vmin=12, vmax=46)
    cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                    cax=ax, orientation='horizontal')

    # Valor que você quer destacar na barra de cores
  

    # Adicionar um marcador na posição do valor destacado
    ax.axvline(value, color='black', linestyle='solid', linewidth=4)

    # Salvar a imagem do gráfico
    fig.savefig('grafico.png', dpi=300, bbox_inches='tight')



def create_inputs():
    st.title('Calcule seu imc com uma rede neural')
    st.markdown('Calcule seu imc com uma rede neural')

    weight_input = st.number_input(
        "Peso (kg)",  # Clear label for weight
        value=None,
        placeholder="Insira seu peso...",
        key="weight_input",
        max_value=150,
        min_value=40
    )

    height_input = st.number_input(
        "Altura (m)",  # Clear label for height
        value=None,
        placeholder="Insira sua altura...",
        key="height_input",
        max_value=2.3,
        min_value=1.2
    )

    if st.button("Calcule seu IMC com uma rede neural)"):  # Button to trigger calculation
        if weight_input is not None and height_input is not None:
            # Calculate BMI (Body Mass Index) without a neural network
            bmi = NeuralNetwork()
            bmi.load_model()
            print(height_input, weight_input)
            result = bmi.predict([height_input, weight_input / 100]) * 100

            # Display the calculated BMI
            st.write(f"Seu IMC aproximado pela rede é: {round(result, 2)}")

            # Use o objeto axes retornado para plotar
           
            generate_fig(value=int(round(result)))
            st.image('grafico.png',use_column_width=True)
            # Provide informative text based on the calculated BMI range
            if result < 18.5:
                st.markdown("**Classificação:** Subpeso")
                st.markdown("Você pode estar abaixo do peso ideal. Converse com um médico ou nutricionista para avaliar sua alimentação e hábitos de vida.")
            elif result < 25:
                st.markdown("**Classificação:** Normal")
                st.markdown("Parabéns! Você está dentro do peso ideal")
            elif result < 30:
                st.markdown("**Classificação:** Sobrepeso")
                st.markdown("Você está com um pouco de sobrepeso. Adotar hábitos saudáveis como uma alimentação balanceada e exercícios físicos regulares pode te ajudar a alcançar o peso ideal.")
            elif result < 35:
                st.markdown("**Classificação:** Obesidade Grau I")
                st.markdown("Sua saúde pode estar em risco devido à obesidade. É importante consultar um médico para acompanhamento e buscar um estilo de vida mais saudável.")
            elif result < 40:
                st.markdown("**Classificação:** Obesidade Grau II")
                st.markdown("A obesidade de grau II pode trazer sérios riscos à saúde. Agende uma consulta médica para receber orientação e acompanhamento adequados.")
            else:
                st.markdown("**Classificação:** Obesidade Grau III (Mórbida)")
                st.markdown("A obesidade mórbida requer atenção médica imediata. Consulte um médico para discutir as opções de tratamento mais adequadas para o seu caso.")

        else:
            st.warning("Por favor, insira seu peso e altura para calcular o IMC.")
     

def main():
    create_inputs()

if __name__ == '__main__':
    st.set_page_config(page_title='Home')
    main()
