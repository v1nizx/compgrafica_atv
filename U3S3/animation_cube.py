import sys
import numpy as np
import imageio

# Configura vispy para usar pyopengl
import vispy
vispy.use('pyqt6', 'gl2')
from vispy import app, gloo

# ==========================================
# 1. CONFIGURAÇÃO DE GEOMETRIA E MATEMÁTICA
# ==========================================

def make_cube():
    """Gera vértices e índices para um cubo unitário."""
    v = np.array([
        [-0.5, -0.5, -0.5], [+0.5, -0.5, -0.5],
        [+0.5, +0.5, -0.5], [-0.5, +0.5, -0.5],
        [-0.5, -0.5, +0.5], [+0.5, -0.5, +0.5],
        [+0.5, +0.5, +0.5], [-0.5, +0.5, +0.5],
    ], dtype=np.float32)

    faces = np.array([
        [0,1,2], [0,2,3], [4,5,6], [4,6,7], # Z
        [0,4,7], [0,7,3], [1,5,6], [1,6,2], # X
        [3,2,6], [3,6,7], [0,1,5], [0,5,4], # Y
    ], dtype=np.uint32)

    positions, normals = [], []
    for f in faces:
        a, b, c = v[f[0]], v[f[1]], v[f[2]]
        n = np.cross(b - a, c - a)
        n /= (np.linalg.norm(n) + 1e-8)
        positions += [a, b, c]
        normals += [n, n, n]
    return np.array(positions, dtype=np.float32), np.array(normals, dtype=np.float32)

def perspective(fovy, aspect, zn, zf):
    f = 1.0 / np.tan(0.5 * fovy)
    M = np.zeros((4, 4), dtype=np.float32)
    M[0, 0] = f / aspect
    M[1, 1] = f
    M[2, 2] = (zf + zn) / (zn - zf)
    M[2, 3] = (2 * zf * zn) / (zn - zf)
    M[3, 2] = -1.0
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

# ==========================================
# 2. SHADERS (MODELO DE PHONG)
# ==========================================

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
precision mediump float;
uniform vec3 u_light_pos;
uniform vec3 u_view_pos;
uniform vec3 u_color_diffuse;
uniform vec3 u_color_specular;
uniform vec3 u_la;
uniform vec3 u_ld;
uniform vec3 u_ls;
uniform float u_ka;
uniform float u_kd;
uniform float u_ks;
uniform float u_s;
varying vec3 v_normal;
varying vec3 v_pos_world;

void main() {
    vec3 N = normalize(v_normal);
    vec3 L = normalize(u_light_pos - v_pos_world);
    vec3 V = normalize(u_view_pos - v_pos_world);
    vec3 R = reflect(-L, N);
    vec3 lamb = u_ka * u_la * u_color_diffuse;
    float diff = max(dot(N, L), 0.0);
    vec3 Idiff = u_kd * diff * u_ld * u_color_diffuse;
    float spec = pow(max(dot(R, V), 0.0), u_s);
    vec3 Ispec = u_ks * spec * u_ls * u_color_specular;
    gl_FragColor = vec4(lamb + Idiff + Ispec, 1.0);
}
"""

# ==========================================
# 3. EXECUÇÃO PRINCIPAL
# ==========================================

class CubeCanvas(app.Canvas):
    def __init__(self, width, height, n_frames, target_fps, filename):
        app.Canvas.__init__(self, size=(width, height), title='Renderizando cubo...')
        self.width = width
        self.height = height
        self.n_frames = n_frames
        self.target_fps = target_fps
        self.filename = filename
        self.frames = []
        self.current_frame = 0
        self.program = None  # Será inicializado no on_draw
        
        # Timer para animação
        self._timer = app.Timer(interval=1/60, connect=self.on_timer, start=True)
        
        self.show()
    
    def _init_program(self):
        """Inicializa o programa OpenGL (deve ser chamado dentro do contexto GL)."""
        # Prepara Programa e Dados
        positions, normals = make_cube()
        self.program = gloo.Program(VERTEX_SHADER, FRAGMENT_SHADER)
        self.program['a_position'] = gloo.VertexBuffer(positions)
        self.program['a_normal'] = gloo.VertexBuffer(normals)
        
        # Configura Câmera e Luz
        eye = np.array([0, 2, 3], dtype=np.float32)
        self.program['u_view'] = look_at(eye, np.zeros(3, dtype=np.float32), np.array([0,1,0], dtype=np.float32))
        self.program['u_proj'] = perspective(np.deg2rad(60), 1.0, 0.1, 100.0)
        self.program['u_view_pos'] = eye
        self.program['u_light_pos'] = np.array([2, 2, 2], dtype=np.float32)
        
        # Configura Cores Iniciais
        self.program['u_la'] = self.program['u_ld'] = self.program['u_ls'] = np.ones(3, dtype=np.float32)
        self.program['u_color_diffuse'] = np.array([0.2, 0.6, 0.9], dtype=np.float32)
        self.program['u_color_specular'] = np.ones(3, dtype=np.float32)
        self.program['u_ka'] = 0.3
        self.program['u_kd'] = 0.8
        self.program['u_ks'] = 0.5
        self.program['u_s'] = 32.0
        
        gloo.set_state(depth_test=True)
        
    def on_timer(self, event):
        self.update()
    
    def on_draw(self, event):
        # Inicializa o programa na primeira execução (dentro do contexto GL)
        if self.program is None:
            self._init_program()
            
        if self.current_frame >= self.n_frames:
            return
            
        t = self.current_frame / self.n_frames
        
        # Atualiza Rotação
        self.program['u_model'] = rotate_y(2 * np.pi * t)
        
        # Atualiza Material (Transição Fosco -> Brilhante)
        self.program['u_ks'] = float(0.1 + 0.9 * t)
        self.program['u_s'] = float(8.0 + 64.0 * t)
        
        # Renderiza
        gloo.set_viewport(0, 0, self.width, self.height)
        gloo.clear(color=(0.1, 0.1, 0.1, 1.0), depth=True)
        self.program.draw('triangles')
        
        # Captura o frame
        img = gloo.read_pixels((0, 0, self.width, self.height), alpha=False)
        self.frames.append(np.flipud(img))
        
        self.current_frame += 1
        print(f"\r🎥 Renderizando: {self.current_frame}/{self.n_frames}", end='', flush=True)
        
        # Verifica se terminou
        if self.current_frame >= self.n_frames:
            self._timer.stop()
            self.save_gif()
            self.close()
            app.quit()
    
    def save_gif(self):
        print(f"\n💾 Salvando GIF com {len(self.frames)} frames...")
        print(f"   Frame 0 - Min: {self.frames[0].min()}, Max: {self.frames[0].max()}")
        print(f"   Frame -1 - Min: {self.frames[-1].min()}, Max: {self.frames[-1].max()}")
        imageio.mimsave(self.filename, self.frames, fps=self.target_fps, loop=0)
        print(f"✅ Sucesso! Arquivo gerado: {self.filename}")

if __name__ == '__main__':
    try:
        print("🚀 Inicializando OpenGL...")
        
        canvas = CubeCanvas(
            width=512, 
            height=512, 
            n_frames=60, 
            target_fps=20, 
            filename='cubo_phong_local.gif'
        )
        app.run()

    except ImportError as e:
        print("❌ Erro de Biblioteca: Certifique-se de instalar 'vispy', 'numpy', 'imageio' e 'PyQt6'.")
        print(f"Detalhe: {e}")
    except Exception as e:
        print(f"❌ Erro Inesperado: {e}")