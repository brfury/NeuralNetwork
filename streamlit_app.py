
import matplotlib as mpl
import matplotlib.pyplot as plt
import streamlit as st

from model import NeuralNetwork


def generate_fig(value):

    fig, ax = plt.subplots(figsize=(8, 0.5), subplot_kw={"clip_on": False})

    cmap = mpl.cm.hsv
    norm = mpl.colors.Normalize(vmin=12, vmax=46)
    cb = fig.colorbar(
        mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation="horizontal"
    )

    ax.axvline(value, color="black", linestyle="solid", linewidth=4)

    fig.savefig("grafico.png", dpi=300, bbox_inches="tight")


def classify_bmi(result):
    classifications = {
        (
            0,
            18.5,
        ): "Você pode estar abaixo do peso ideal. Converse com um médico ou nutricionista para avaliar sua alimentação e hábitos de vida.",
        (18.5, 25): "Parabéns! Você está dentro do peso ideal",
        (
            25,
            30,
        ): "Você está com um pouco de sobrepeso. Adotar hábitos saudáveis como uma alimentação balanceada e exercícios físicos regulares pode te ajudar a alcançar o peso ideal.",
        (
            30,
            35,
        ): "Sua saúde pode estar em risco devido à obesidade. É importante consultar um médico para acompanhamento e buscar um estilo de vida mais saudável.",
        (
            35,
            40,
        ): "A obesidade de grau II pode trazer sérios riscos à saúde. Agende uma consulta médica para receber orientação e acompanhamento adequados.",
        (
            40,
            float("inf"),
        ): "A obesidade mórbida requer atenção médica imediata. Consulte um médico para discutir as opções de tratamento mais adequadas para o seu caso.",
    }

    for (lower, upper), message in classifications.items():
        if lower <= result < upper:
            return message


def create_inputs():
    st.title("Calcule seu IMC com uma rede neural")
    st.markdown("Calcule seu IMC com uma rede neural")

    weight_input = st.number_input(
        "Peso (kg)",
        value=None,
        placeholder="Insira seu peso...",
        key="weight_input",
        max_value=150,
        min_value=40,
    )

    height_input = st.number_input(
        "Altura (m)",
        value=None,
        placeholder="Insira sua altura...",
        key="height_input",
        max_value=2.3,
        min_value=1.2,
    )
    if st.button("Calcule seu IMC"):
        if weight_input is not None and height_input is not None:

            bmi = NeuralNetwork()
            bmi.load_model()
            print(height_input, weight_input)
            result = bmi.predict([height_input, weight_input / 100]) * 100

            st.write(f"Seu IMC aproximado pela rede é: {round(result, 2)}")

            generate_fig(value=int(round(result)))
            st.image("grafico.png", use_column_width=True)

            st.markdown(f"**Classificação:** {classify_bmi(result)}")
        else:
            st.warning("Por favor, insira seu peso e altura para calcular o IMC.")
    git_buton, linkedin = st.columns(2)


    
    with git_buton:
        st.link_button("Acesse o repositório da rede", url='https://github.com/brfury/NeuralNetwork')
            
    with linkedin:
        st.link_button("Linkendin",url='www.linkedin.com/in/datascientistbruno',)

    st.header('links dos artigos')
    st.markdown('''A qui vc pode ler e entender desde o básico de uma rede neural, a construir uma do 
                zero, sem auxílio de libs externas como Pytorch ou TensorFlow 
                [link](https://medium.com/@bruno1912200/entenda-como-uma-rede-neural-funciona-e-construa-sua-rede-neural-do-zero-sem-pytorch-ou-tensorflow-1b0e24e28469) ''' )
    


def main():
    create_inputs()


if __name__ == "__main__":
    st.set_page_config(page_title="Home")
    main()
