# --- Parte I: Modelo de Iluminação de Phong (Superfície Plana) ---
import numpy as np
import matplotlib.pyplot as plt

# --- Passo 1: Definir a função do Modelo de Phong ---
def phong_model(normal, light_dir, view_dir, 
                ka=0.5, kd=0.7, ks=0.9, shininess=20, # Valores da pag 1 [cite: 16]
                Ia=0.1, Id=0.7, Is=0.5):              # Valores da pag 1 [cite: 15]
    
    # 1. Normalização dos vetores (Essencial para o produto escalar funcionar) [cite: 22]
    normal = normal / np.linalg.norm(normal)
    light_dir = light_dir / np.linalg.norm(light_dir)
    view_dir = view_dir / np.linalg.norm(view_dir)

    # 2. Componente Ambiente (Luz base constante) [cite: 25]
    I_amb = Ia * ka

    # 3. Componente Difusa (Depende do ângulo da luz) [cite: 26]
    # np.dot calcula o cosseno do ângulo entre a normal e a luz
    diff = max(np.dot(normal, light_dir), 0)
    I_diff = Id * kd * diff

    # 4. Componente Especular (O brilho/reflexo) [cite: 28]
    # Calcula a direção do reflexo
    reflect_dir = 2 * np.dot(normal, light_dir) * normal - light_dir
    # Calcula o quanto o reflexo aponta para o observador (câmera)
    spec = max(np.dot(view_dir, reflect_dir), 0) ** shininess
    I_spec = Is * ks * spec

    # Intensidade Final [cite: 31]
    return I_amb + I_diff + I_spec

# --- Passo 2: Criar a simulação ---

# Criando uma grade de 50x50 pontos [cite: 33]
x = np.linspace(-1, 1, 50)
y = np.linspace(-1, 1, 50)
X, Y = np.meshgrid(x, y)

# Definindo vetores fixos para este exemplo (Superfície plana apontando para cima Z=1)
normal = np.array([0, 0, 1])     # [cite: 37]
light_dir = np.array([1, 1, 1])  # Luz vindo da diagonal [cite: 38]
view_dir = np.array([0, 0, 1])   # Observador olhando de cima [cite: 39]

# Matriz para guardar a intensidade de luz calculada
intensity = np.zeros_like(X)

# Calcular Phong para cada pixel da grade [cite: 43]
rows, cols = X.shape
for i in range(rows):
    for j in range(cols):
        # Aqui poderíamos variar a normal se fosse uma esfera, 
        # mas no exemplo do PDF a superfície é plana.
        intensity[i, j] = phong_model(normal, light_dir, view_dir)

# --- Passo 3: Visualizar ---
plt.figure(figsize=(6, 6))
plt.imshow(intensity, cmap='inferno', extent=(-1, 1, -1, 1), origin='lower') # [cite: 48]
plt.title("Iluminação de Phong - Superfície Plana")
plt.colorbar(label="Intensidade da Luz")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()


# --- Parte II: Tonalização Constante vs. Interpolada ---

import numpy as np
import matplotlib.pyplot as plt

# --- Passo 1: Função de Phong Adaptada para RGB (Vetores) ---
def phong_reflection(normal, light_dir, view_dir,
                     ka=0.2, kd=0.7, ks=0.5, s=10, # Parâmetros da Parte II [cite: 72]
                     Ia=np.array([0.1, 0.1, 0.1]), # Luz ambiente RGB
                     Id=np.array([1.0, 1.0, 1.0]), # Luz difusa branca
                     Is=np.array([1.0, 1.0, 1.0])): # Luz especular branca

    # Normalização [cite: 76]
    normal = normal / np.linalg.norm(normal)
    light_dir = light_dir / np.linalg.norm(light_dir)
    view_dir = view_dir / np.linalg.norm(view_dir)

    # Ambiente [cite: 79]
    I_amb = ka * Ia

    # Difusa [cite: 81]
    diff = max(np.dot(normal, light_dir), 0.0)
    I_diff = kd * diff * Id

    # Especular [cite: 84]
    reflect_dir = 2 * np.dot(normal, light_dir) * normal - light_dir
    spec = max(np.dot(view_dir, reflect_dir), 0.0) ** s
    I_spec = ks * spec * Is

    # Resultado final, garantindo que não ultrapasse 1.0 (branco)
    result = I_amb + I_diff + I_spec
    return np.clip(result, 0, 1)

# --- Passo 2: Definir Geometria e Luzes ---

# Vértices do triângulo [cite: 89]
vertices = np.array(
    [
        [0, 0],
        [1, 0],
        [0.5, 1]
    ]
)

# Para simular 3D, assumimos Z=0 nos vértices e Normal Z=1
normal = np.array([0, 0, 1])
light_dir = np.array([0.5, 0.5, 1]) # Luz vindo da direita/cima [cite: 93]
view_dir = np.array([0, 0, 1])

# --- Passo 3: Calcular Cores ---

# Calculamos a cor de Phong para CADA vértice individualmente [cite: 95]
colors = []
for v in vertices:
    # Nota: Em um modelo 3D real, a normal mudaria em cada vértice (ex: esfera).
    # Aqui usamos a mesma normal para simplificar, como no PDF.
    color = phong_reflection(normal, light_dir, view_dir)
    colors.append(color)
colors = np.array(colors)

# Tonalização Constante: Média das cores dos vértices [cite: 96]
constant_color = np.mean(colors, axis=0)

# --- Passo 4: Plotar Comparação ---
fig, ax = plt.subplots(1, 2, figsize=(10, 5))

# Gráfico 1: Tonalização Constante (Flat)
# O triângulo inteiro tem UMA única cor [cite: 100]
t1 = plt.Polygon(vertices, color=constant_color)
ax[0].add_patch(t1)
ax[0].set_xlim(-0.2, 1.2)
ax[0].set_ylim(-0.2, 1.2)
ax[0].set_title("Tonalização Constante (Flat)")
ax[0].set_aspect('equal')

# Gráfico 2: Tonalização Interpolada (Gouraud)
# O matplotlib interpola as cores definidas em cada vértice [cite: 103]
# Tripcolor cria a malha e 'gouraud' faz o gradiente suave

# Convert RGB vertex colors to scalar values for tripcolor's C argument
C_vertex_scalars = np.mean(colors, axis=1)

ax[1].tripcolor(vertices[:, 0], vertices[:, 1], C_vertex_scalars, triangles=[[0, 1, 2]],
                shading='gouraud', cmap='viridis') # Added cmap for Gouraud
ax[1].set_xlim(-0.2, 1.2)
ax[1].set_ylim(-0.2, 1.2)
ax[1].set_title("Tonalização Interpolada (Gouraud)")
ax[1].set_aspect('equal')

plt.show()


# --- Bloco 1: Desafios 1 e 2 (Variando Parâmetros e Luz) ---

import numpy as np
import matplotlib.pyplot as plt

# --- Função de Phong (Mesma da base) ---
def phong_reflection(normal, light_dir, view_dir, ka, kd, ks, s, Ia, Id, Is):
    normal = normal / np.linalg.norm(normal)
    light_dir = light_dir / np.linalg.norm(light_dir)
    view_dir = view_dir / np.linalg.norm(view_dir)

    I_amb = ka * Ia
    diff = max(np.dot(normal, light_dir), 0.0)
    I_diff = kd * diff * Id
    reflect_dir = 2 * np.dot(normal, light_dir) * normal - light_dir
    spec = max(np.dot(view_dir, reflect_dir), 0.0) ** s
    I_spec = ks * spec * Is

    return np.clip(I_amb + I_diff + I_spec, 0, 1)

# Configuração Base
vertices = np.array([[0, 0], [1, 0], [0.5, 0.866]]) # Triângulo Equilátero
normal = np.array([0, 0, 1])
view_dir = np.array([0, 0, 1])
Ia = Id = Is = np.array([1.0, 1.0, 1.0])

# --- Preparar Comparação (Desafios 1 e 2) ---
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("Desafios 1 e 2: Materiais e Luz", fontsize=16)

# Configurações para testar
cenarios = [
    # Título, Light Dir, Ka, Kd, Ks, Shininess (s)
    ("Padrão", [0.5, 0.5, 1], 0.2, 0.7, 0.5, 10),
    ("Muito Brilho (Plástico)", [0.5, 0.5, 1], 0.1, 0.5, 1.0, 100), # Desafio 1
    ("Fosco (Borracha)", [0.5, 0.5, 1], 0.4, 0.9, 0.0, 1),      # Desafio 1
    ("Luz da Esquerda", [-1.0, 0.5, 1], 0.2, 0.7, 0.5, 10)      # Desafio 2
]

# Loop para gerar os 4 gráficos
for idx, (titulo, l_dir, ka, kd, ks, s) in enumerate(cenarios):
    ax = axs[idx//2, idx%2]

    # Calcular cores nos vértices
    colors = []
    for v in vertices:
        c = phong_reflection(normal, np.array(l_dir), view_dir, ka, kd, ks, s, Ia, Id, Is)
        colors.append(c)
    
    # Convert RGB vertex colors to scalar values for tripcolor's C argument
    C_vertex_scalars = np.mean(colors, axis=1)

    # Plotar Interpolado (Gouraud)
    # Pass C_vertex_scalars as the positional 'C' argument
    ax.tripcolor(vertices[:,0], vertices[:,1], C_vertex_scalars, triangles=[[0,1,2]], 
                 shading='gouraud', cmap='viridis') # Added cmap for Gouraud

    ax.set_title(f"{titulo}\n(s={s}, Luz={l_dir})")
    ax.set_aspect('equal')
    ax.axis('off')

plt.tight_layout()
plt.show()


# --- Desafio 3: Novo Polígono (Hexágono) ---

# Criar vértices de um hexágono
t = np.linspace(0, 2*np.pi, 7)[:-1] # 6 ângulos
hex_x = np.cos(t)
hex_y = np.sin(t)
vertices_hex = np.column_stack((hex_x, hex_y))

# Centro para ajudar na triangulação (opcional, mas bom para Gouraud)
vertices_hex = np.vstack(([0,0], vertices_hex)) # Adiciona centro no índice 0

# Definir triângulos conectando o centro às bordas (indices)
triangulos = []
for i in range(1, 7):
    triangulos.append([0, i, i+1 if i < 6 else 1])

# Iluminação
light_dir = np.array([1, -1, 1]) # Luz vindo de baixo/direita
normal = np.array([0, 0, 1])

colors_hex = []
for v in vertices_hex:
    # Pequeno truque: variar levemente a normal baseada na posição X
    # para simular uma superfície levemente curva, senão o hexágono plano fica todo igual.
    # Se quiser plano perfeito, use normal fixa.
    n_local = np.array([v[0]*0.5, v[1]*0.5, 1])
    c = phong_reflection(n_local, light_dir, view_dir, 0.2, 0.6, 0.8, 50, Ia, Id, Is)
    colors_hex.append(c)

# Calcular cor média para o Constante
cor_media = np.mean(colors_hex, axis=0)

# Plotagem
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle("Desafio 3: Hexágono (Curvo) - Constante vs Interpolado")

# 1. Constante (Flat)
# Desenhamos os triângulos todos da mesma cor
for tri in triangulos:
    poligono = plt.Polygon(vertices_hex[tri], color=cor_media)
    ax[0].add_patch(poligono)
ax[0].set_xlim(-1.1, 1.1); ax[0].set_ylim(-1.1, 1.1)
ax[0].set_title("Flat Shading (Constante)")
ax[0].set_aspect('equal')

# 2. Interpolado (Gouraud)
# Para Gouraud shading com tripcolor, 'C' deve ser um array de valores escalares por vértice.
# facecolors é para cores por triângulo, não por vértice, mesmo com shading='gouraud'.
# Convertendo os valores RGB dos vértices para um escalar (média) para 'C'.
C_vertex_scalars = np.mean(colors_hex, axis=1)

# Fix: Pass C_vertex_scalars as a positional argument
ax[1].tripcolor(vertices_hex[:,0], vertices_hex[:,1], C_vertex_scalars, triangles=triangulos,
                shading='gouraud', cmap='viridis') # 'viridis' é um colormap padrão, você pode escolher outro
ax[1].set_title("Gouraud Shading (Interpolado)")
ax[1].set_aspect('equal')
ax[1].set_xlim(-1.1, 1.1); ax[1].set_ylim(-1.1, 1.1)

plt.show()


# --- Desafio 4: Textura Simples sobre Polígono ---

from matplotlib.patches import Polygon

# 1. Criar uma Textura Procedural (Xadrez) para não depender de arquivos externos
# Tamanho 100x100 pixels
textura = np.zeros((100, 100))
check_size = 10
for y in range(100):
    for x in range(100):
        if ((x // check_size) + (y // check_size)) % 2 == 0:
            textura[y, x] = 1 # Branco

fig, ax = plt.subplots(figsize=(6, 6))

# 2. Definir o Polígono (Um triângulo invertido)
poly_verts = np.array([[20, 20], [80, 20], [50, 90]])
poly_patch = Polygon(poly_verts, facecolor='none', edgecolor='red', linewidth=2)

# 3. Mostrar a imagem (Textura)
im = ax.imshow(textura, cmap='gray', origin='lower')

# 4. A Mágica: Aplicar o Polígono como "Máscara" (Clip) da imagem
# Isso faz a imagem aparecer APENAS dentro do triângulo
im.set_clip_path(poly_patch)
ax.add_patch(poly_patch) # Adiciona a borda vermelha para vermos o limite

ax.set_title("Desafio 4: Textura Mapeada (Recorte)")
plt.show()
