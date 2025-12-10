# --- Passo 1: Imagem (Matriz 7x7 fornecida) ---
imagem = [
    [18, 13, 10,  9, 10, 13, 18],
    [13,  8,  5,  4,  5,  8, 13],
    [10,  5,  2,  1,  2,  5, 10],
    [ 9,  4,  1,  0,  1,  4,  9],
    [10,  5,  2,  1,  2,  5, 10],
    [13,  8,  5,  4,  5,  8, 13],
    [18, 13, 10,  9, 10, 13, 18]
]

# --- Definição dos Filtros (Desafio 1 e Passo 2) ---
filtro_laplaciano = [
    [ 0, -1,  0],
    [-1,  4, -1],
    [ 0, -1,  0]
]

# Sobel X (Detecta bordas verticais)
sobel_x = [
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
]

# Sobel Y (Detecta bordas horizontais)
sobel_y = [
    [-1, -2, -1],
    [ 0,  0,  0],
    [ 1,  2,  1]
]

# --- Passo 3: Função de Convolução ---
def aplicar_convolucao(imagem, filtro):
    altura = len(imagem)
    largura = len(imagem[0])
    tamanho_filtro = len(filtro)
    offset = tamanho_filtro // 2

    # Cria matriz zerada
    resultado = [[0 for _ in range(largura)] for _ in range(altura)]

    for i in range(offset, altura - offset):
        for j in range(offset, largura - offset):
            soma = 0
            for fi in range(tamanho_filtro):
                for fj in range(tamanho_filtro):
                    # Nota: Invertemos o índice do filtro se fosse convolução estrita,
                    # mas em processamento de imagem, usualmente aplica-se correlação direta.
                    soma += imagem[i - offset + fi][j - offset + fj] * filtro[fi][fj]
            resultado[i][j] = soma
    return resultado

# --- Desafio 4: Função de Normalização [0, 255] ---
def normalizar_imagem(matriz):
    # Achatar a matriz para encontrar min e max facilmente
    todos_valores = [val for linha in matriz for val in linha]
    min_val = min(todos_valores)
    max_val = max(todos_valores)

    altura = len(matriz)
    largura = len(matriz[0])

    matriz_norm = [[0 for _ in range(largura)] for _ in range(altura)]

    # Evitar divisão por zero se a imagem for toda uniforme
    if max_val == min_val:
        return matriz

    for i in range(altura):
        for j in range(largura):
            # Fórmula Min-Max
            novo_val = (matriz[i][j] - min_val) * 255 / (max_val - min_val)
            matriz_norm[i][j] = int(novo_val) # Converter para inteiro

    return matriz_norm

# --- Execução e Comparação ---

# 1. Laplaciano
res_laplaciano = aplicar_convolucao(imagem, filtro_laplaciano)
norm_laplaciano = normalizar_imagem(res_laplaciano)

# 2. Sobel X
res_sobel_x = aplicar_convolucao(imagem, sobel_x)
norm_sobel_x = normalizar_imagem(res_sobel_x)

# 3. Sobel Y
res_sobel_y = aplicar_convolucao(imagem, sobel_y)
norm_sobel_y = normalizar_imagem(res_sobel_y)

# Função auxiliar para printar bonito
def printar_matriz(nome, matriz):
    print(f"\n--- {nome} ---")
    for linha in matriz:
        # Formata para ficar alinhado na tela
        print([f"{val:3}" for val in linha])

printar_matriz("Imagem Original", imagem)
printar_matriz("Laplaciano (Normalizado)", norm_laplaciano)
printar_matriz("Sobel X (Bordas Verticais - Normalizado)", norm_sobel_x)
printar_matriz("Sobel Y (Bordas Horizontais - Normalizado)", norm_sobel_y)
