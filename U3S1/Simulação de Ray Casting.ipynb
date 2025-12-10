import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

# --- 1. Modelagem da Cena ---
# Triângulos definidos por seus vértices
tri1 = np.array([[1, 3], [2, 5], [3, 3]])
tri2 = np.array([[2, 2], [3, 4], [4, 2]])

# Centro de projeção (Câmera) e Ponto de Mira (Target)
CA = np.array([0, 0]) # Camera
P = np.array([3, 3])  # Ponto alvo do raio

# --- 2. Função de Interseção (Ray Casting Simplificado) ---
def ray_intersects(triangle, origin, target, steps=200):
    """
    Verifica se um raio de 'origin' para 'target' intersecta o triângulo.
    Retorna: (Intersectou?, Ponto de interseção, Distância)
    """
    path = Path(triangle)
    direction = target - origin

    # Discretização do raio (amostragem)
    # Aumentei 'steps' para 200 para maior precisão
    for t in np.linspace(0, 1.5, steps): # 1.5 para o raio ir além do ponto P se necessário
        point = origin + t * direction

        # Se o ponto amostrado está dentro do triângulo
        if path.contains_point(point):
            dist = np.linalg.norm(point - origin)
            return True, point, dist

    return False, None, float('inf')

# --- 3. Lógica Principal ---
# Verificar interseções para cada objeto
hit1, pt1, dist1 = ray_intersects(tri1, CA, P)
hit2, pt2, dist2 = ray_intersects(tri2, CA, P)

visivel = "Nenhum"
ponto_final = P # O raio termina em P se não bater em nada
cor_hit = 'black'

if hit1 and hit2:
    # Oclusão: Vence quem tiver a menor distância
    if dist1 < dist2:
        visivel = "Triângulo 1 (Laranja)"
        ponto_final = pt1
        cor_hit = 'orange'
    else:
        visivel = "Triângulo 2 (Azul)"
        ponto_final = pt2
        cor_hit = 'blue'
elif hit1:
    visivel = "Triângulo 1 (Laranja)"
    ponto_final = pt1
    cor_hit = 'orange'
elif hit2:
    visivel = "Triângulo 2 (Azul)"
    ponto_final = pt2
    cor_hit = 'blue'

print(f'Objeto visível: {visivel}')

# --- 4. Visualização ---
plt.figure(figsize=(8, 8))

# Desenhar Triângulos
t1_plot = plt.Polygon(tri1, color='orange', alpha=0.5, label='Triângulo 1')
t2_plot = plt.Polygon(tri2, color='blue', alpha=0.5, label='Triângulo 2')
plt.gca().add_patch(t1_plot)
plt.gca().add_patch(t2_plot)

# Desenhar Raio (Do CA até o ponto de impacto ou até P)
plt.plot([CA[0], ponto_final[0]], [CA[1], ponto_final[1]], 'k-', linewidth=1.5, label='Raio Visível')
# Desenhar continuação tracejada se houve impacto (apenas para ilustrar oclusão)
if visivel != "Nenhum":
    plt.plot([ponto_final[0], P[0]], [ponto_final[1], P[1]], 'k:', alpha=0.3, label='Raio Ocluso')

# Pontos Chave
plt.scatter(*CA, color='black', s=100, marker='s', label='Câmera (CA)')
plt.scatter(*P, color='red', s=50, label='Direção (P)')
if visivel != "Nenhum":
    plt.scatter(*ponto_final, color='red', s=100, marker='x', zorder=5, label='Ponto de Impacto')

plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.title(f'Ray Casting com Oclusão\nVisível: {visivel}')
plt.xlim(-1, 6)
plt.ylim(-1, 6)
plt.gca().set_aspect('equal')
plt.show()


# --- DESAFIO, OUTRO CODIGO JÁ ---

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path

# --- Configuração da Cena ---
# Lista de objetos (dicionários para facilitar propriedades)
objetos = [
    {'id': 1, 'verts': np.array([[1, 3], [2, 5], [3, 3]]), 'color': 'orange', 'path': None},
    {'id': 2, 'verts': np.array([[2, 2], [3, 4], [4, 2]]), 'color': 'blue', 'path': None},
    {'id': 3, 'verts': np.array([[0.5, 1], [1.5, 2.5], [2.5, 1]]), 'color': 'green', 'path': None}
]

# Pré-calcular os Paths (otimização)
for obj in objetos:
    obj['path'] = Path(obj['verts'])

CA = np.array([0, 0])     # Câmera
P_dir = np.array([3, 4])  # Direção do olhar
LUZ = np.array([4, 5])    # Fonte de Luz

# --- Funções de Ray Casting ---

def cast_ray(origin, target, objects_list, ignore_id=None):
    """
    Lança um raio e retorna o objeto mais próximo interceptado.
    ignore_id: usado para sombras (não fazer sombra em si mesmo na saída)
    """
    direction = target - origin
    closest_dist = float('inf')
    hit_obj = None
    hit_point = None

    # Aumentamos a precisão da amostragem
    steps = 300

    for obj in objects_list:
        if obj['id'] == ignore_id: continue

        # Verifica pontos ao longo do vetor
        # Usamos linspace estendido para garantir cobertura
        for t in np.linspace(0, 1.2, steps):
            p = origin + t * direction
            if obj['path'].contains_point(p):
                dist = np.linalg.norm(p - origin)
                # Se achou um ponto mais próximo que o anterior
                if dist < closest_dist:
                    closest_dist = dist
                    hit_obj = obj
                    hit_point = p
                # Como estamos indo do começo ao fim do raio,
                # a primeira colisão com ESTE objeto é a entrada. Pare e vá para o prox objeto.
                break

    return hit_obj, hit_point

def check_shadow(surface_point, light_pos, objects_list, self_id):
    """
    Lança um raio do ponto de superfície até a luz.
    Se bater em algo, está na sombra.
    """
    # Pequeno deslocamento (bias) para o raio não bater no próprio objeto de origem
    direction = light_pos - surface_point
    origin_biased = surface_point + (direction * 0.05)

    blocker, _ = cast_ray(origin_biased, light_pos, objects_list, ignore_id=self_id)
    return blocker is not None # Retorna True se houver bloqueio (sombra)

# --- Execução ---

# 1. Raio Primário (Câmera -> Cena)
obj_visto, ponto_impacto = cast_ray(CA, P_dir, objetos)

# 2. Raio Secundário (Ponto de Impacto -> Luz)
na_sombra = False
if obj_visto:
    na_sombra = check_shadow(ponto_impacto, LUZ, objetos, obj_visto['id'])

# --- Plotagem ---
plt.figure(figsize=(8, 8))

# Desenhar Objetos
for obj in objetos:
    poly = plt.Polygon(obj['verts'], color=obj['color'], alpha=0.5, label=f"Triangulo {obj['id']}")
    plt.gca().add_patch(poly)

# Desenhar Câmera e Luz
plt.scatter(*CA, color='black', marker='s', s=100, label='Câmera')
plt.scatter(*LUZ, color='gold', marker='*', s=200, label='Luz')

# Desenhar Raio de Visão
if obj_visto:
    # Linha até o impacto
    plt.plot([CA[0], ponto_impacto[0]], [CA[1], ponto_impacto[1]], 'k-', label='Raio Visão')

    # Ponto de impacto (Cor muda se estiver na sombra)
    cor_impacto = 'black' if na_sombra else 'yellow'
    status_sombra = " (NA SOMBRA)" if na_sombra else " (ILUMINADO)"

    plt.scatter(*ponto_impacto, c=cor_impacto, s=100, zorder=10, edgecolors='red', label='Impacto'+status_sombra)

    # Desenhar raio de sombra (Luz -> Impacto)
    style = 'r--' if na_sombra else 'y--'
    plt.plot([ponto_impacto[0], LUZ[0]], [ponto_impacto[1], LUZ[1]], style, alpha=0.6, linewidth=1)

else:
    # Se não bateu em nada, desenha raio infinito
    plt.plot([CA[0], P_dir[0]], [CA[1], P_dir[1]], 'k--', alpha=0.3)

plt.title('Ray Casting com Múltiplos Objetos e Sombra')
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)
plt.xlim(-1, 6)
plt.ylim(-1, 6)
plt.gca().set_aspect('equal')
plt.show()
