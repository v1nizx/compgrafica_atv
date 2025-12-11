import numpy as np
import imageio
from vispy import app, gloo

# --- Configuração da Geometria (Cubo) ---
def make_cube():
    # Vértices (cópia exata da lógica anterior)
    v = np.array([
        [-0.5, -0.5, -0.5], [+0.5, -0.5, -0.5],
        [+0.5, +0.5, -0.5], [-0.5, +0.5, -0.5],
        [-0.5, -0.5, +0.5], [+0.5, -0.5, +0.5],
        [+0.5, +0.5, +0.5], [-0.5, +0.5, +0.5],
    ], dtype=np.float32)

    faces = np.array([
        [0,1,2], [0,2,3], [4,5,6], [4,6,7],
        [0,4,7], [0,7,3], [1,5,6], [1,6,2],
        [3,2,6], [3,6,7], [0,1,5], [0,5,4],
    ], dtype=np.uint32)

    positions, normals = [], []
    for f in faces:
        a, b, c = v[f[0]], v[f[1]], v[f[2]]
        n = np.cross(b - a, c - a)
        n /= (np.linalg.norm(n) + 1e-8)
        positions += [a, b, c]
        normals += [n, n, n]
        
    return np.array(positions, dtype=np.float32), np.array(normals, dtype=np.float32)

# --- Shaders (Vertex e Fragment) ---
VERTEX_SHADER = """
attribute vec3 a_position;
attribute vec3 a_normal;
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;
varying vec3 v_normal;
varying vec3 v_pos_world;
void main() {
    vec4 pos_world = u_model * vec4(a_position, 1.0);
    v_pos_world = pos_world.xyz;
    v_normal = mat3(u_model) * a_normal;
    gl_Position = u_proj * u_view * pos_world;
}
"""

FRAGMENT_SHADER = """
uniform vec3 u_light_pos;
uniform vec3 u_view_pos;
uniform vec3 u_color_diffuse;
uniform vec3 u_color_specular;
uniform vec3 u_la; uniform vec3 u_ld; uniform vec3 u_ls;
uniform float u_ka; uniform float u_kd; uniform float u_ks; uniform float u_s;
varying vec3 v_normal;
varying vec3 v_pos_world;
void main() {
    vec3 N = normalize(v_normal);
    vec3 L = normalize(u_light_pos - v_pos_world);
    vec3 V = normalize(u_view_pos - v_pos_world);
    vec3 R = reflect(-L, N);
    
    // Iluminação Phong
    vec3 lamb = u_ka * u_la * u_color_diffuse;
    float diff = max(dot(N, L), 0.0);
    vec3 Idiff = u_kd * diff * u_ld * u_color_diffuse;
    float spec = pow(max(dot(R, V), 0.0), u_s);
    vec3 Ispec = u_ks * spec * u_ls * u_color_specular;
    
    gl_FragColor = vec4(lamb + Idiff + Ispec, 1.0);
}
"""

# --- Funções Matemáticas ---
def perspective(fovy, aspect, zn, zf):
    f = 1.0 / np.tan(0.5 * fovy); M = np.zeros((4,4), dtype=np.float32)
    M[0,0] = f/aspect; M[1,1] = f; M[2,2] = (zf+zn)/(zn-zf); M[2,3] = (2*zf*zn)/(zn-zf); M[3,2] = -1.0
    return M

def look_at(eye, center, up):
    f = (center - eye); f /= np.linalg.norm(f)
    u = up / np.linalg.norm(up); s = np.cross(f, u); s /= np.linalg.norm(s); u = np.cross(s, f)
    M = np.eye(4, dtype=np.float32); M[0,:3]=s; M[1,:3]=u; M[2,:3]=-f
    T = np.eye(4, dtype=np.float32); T[:3,3]=-eye
    return M @ T

def rotate_y(t):
    c, s = np.cos(t), np.sin(t)
    return np.array([[c,0,s,0],[0,1,0,0],[-s,0,c,0],[0,0,0,1]], dtype=np.float32)

# --- Execução Principal ---
if __name__ == '__main__':
    try:
        # 1. Cria um Canvas invisível para inicializar o Contexto OpenGL do Windows
        # Sem isso, o gloo.Program não funciona
        canvas = app.Canvas(show=False)
        
        print("Preparando renderização...")
        W, H = 512, 512
        n_frames = 60 # Ajuste conforme necessário
        
        # Preparar dados
        positions, normals = make_cube()
        program = gloo.Program(VERTEX_SHADER, FRAGMENT_SHADER)
        program['a_position'] = positions
        program['a_normal'] = normals
        
        # Framebuffer (Renderização Offscreen)
        color = gloo.RenderBuffer((H, W), format='rgba8')
        depth = gloo.RenderBuffer((H, W), format='depth24')
        fbo = gloo.FrameBuffer(color=color, depth=depth)
        
        # Câmera e Luz
        eye_pos = np.array([0, 2, 3], dtype=np.float32)
        view = look_at(eye_pos, np.zeros(3, dtype=np.float32), np.array([0,1,0], dtype=np.float32))
        proj = perspective(np.deg2rad(60), 1.0, 0.1, 100.0)
        
        program['u_view'] = view
        program['u_proj'] = proj
        program['u_view_pos'] = eye_pos
        program['u_light_pos'] = np.array([2, 2, 2], dtype=np.float32)
        
        # Cores Base
        program['u_la'] = program['u_ld'] = program['u_ls'] = np.ones(3, dtype=np.float32)
        program['u_color_diffuse'] = np.array([0.2, 0.6, 0.9], dtype=np.float32)
        program['u_color_specular'] = np.ones(3, dtype=np.float32)
        program['u_ka'] = 0.3
        program['u_kd'] = 0.8
        
        frames = []
        
        # Ativa o Framebuffer
        with fbo:
            gloo.set_viewport(0, 0, W, H)
            gloo.set_state(depth_test=True)
            
            print(f"Renderizando {n_frames} quadros...")
            for i in range(n_frames):
                t = i / float(n_frames)
                
                # Atualiza Uniforms (Rotação e Material)
                program['u_model'] = rotate_y(2 * np.pi * t)
                program['u_ks'] = float(0.1 + 0.9 * t) # Fosco -> Brilhante
                program['u_s'] = float(8.0 + 64.0 * t) # Espalhado -> Concentrado
                
                # Desenha
                gloo.clear(color=(0.1, 0.1, 0.1, 1.0), depth=True)
                program.draw('triangles')
                
                # Captura a imagem
                img = gloo.read_pixels((0, 0, W, H), alpha=False)
                # Inverter verticalmente pois OpenGL tem origem na base
                frames.append(np.flipud(img))
        
        # Salva o GIF
        nome_arquivo = 'animacao_windows.gif'
        imageio.mimsave(nome_arquivo, frames, fps=30, loop=0)
        print(f"Sucesso! Arquivo salvo como: {nome_arquivo}")
        
    except Exception as e:
        print(f"Erro: {e}")
        import traceback
        traceback.print_exc()
