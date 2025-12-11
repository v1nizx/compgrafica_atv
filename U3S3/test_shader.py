import numpy as np
from vispy import app, gloo

VERTEX = """
attribute vec3 a_position;
void main() {
    gl_Position = vec4(a_position, 1.0);
}
"""

FRAGMENT = """
void main() {
    gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0);
}
"""

print("Testando shaders simples...")
try:
    canvas = app.Canvas(size=(100, 100), show=False)
    with canvas:
        program = gloo.Program(VERTEX, FRAGMENT)
        program['a_position'] = np.array([[0,0,0]], dtype=np.float32)
        print("✅ Shaders compilaram com sucesso!")
except Exception as e:
    print(f"❌ Erro: {e}")
