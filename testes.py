import matplotlib.pyplot as plt
import matplotlib as mpl

# Criar a figura e os eixos
fig, ax = plt.subplots(figsize=(6, 1), subplot_kw={'clip_on': False})

# Configurar a barra de cores
cmap = mpl.cm.cool
norm = mpl.colors.Normalize(vmin=5, vmax=10)
cb = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                  cax=ax, orientation='horizontal', label='Some Units')

# Valor que você quer destacar na barra de cores
valor_destacado = 7.5

# Adicionar um marcador na posição do valor destacado
ax.axvline(valor_destacado, color='black', linestyle='dotted', linewidth=4)

# Salvar a imagem do gráfico
fig.savefig('grafico.png', dpi=300, bbox_inches='tight')

plt.show()
