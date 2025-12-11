import sys
import numpy as np
import imageio
from vispy import app, scene
from vispy.visuals.transforms import MatrixTransform

# Desabilita mensagens de warning de DPI
import os
os.environ['QT_ENABLE_HIGHDPI_SCALING'] = '0'

def main():
    # Configurações
    W, H = 512, 512
    n_frames = 60
    fps = 20
    filename = 'cubo_phong_local.gif'
    
    print("🚀 Inicializando OpenGL...")
    
    # Cria canvas com scene
    canvas = scene.SceneCanvas(size=(W, H), show=True, title='Renderizando cubo...')
    view = canvas.central_widget.add_view()
    
    # Configura câmera
    view.camera = scene.cameras.TurntableCamera(
        fov=60, 
        distance=4,
        elevation=30,
        azimuth=0
    )
    
    # Cria cubo usando Box visual
    cube = scene.visuals.Box(
        width=1, height=1, depth=1,
        color=(0.2, 0.6, 0.9, 1),
        edge_color='white',
        parent=view.scene
    )
    
    frames = []
    print(f"🎥 Renderizando {n_frames} quadros...")
    
    for i in range(n_frames):
        t = i / n_frames
        
        # Rotaciona o cubo
        angle = 360 * t
        cube.transform = MatrixTransform()
        cube.transform.rotate(angle, (0, 1, 0))
        
        # Renderiza e captura
        canvas.update()
        app.process_events()
        
        # Captura imagem
        img = canvas.render()
        frames.append(img[:, :, :3])  # Remove canal alpha
        
        print(f"\r🎥 Renderizando: {i+1}/{n_frames}", end='', flush=True)
    
    canvas.close()
    
    print(f"\n💾 Salvando GIF com {len(frames)} frames...")
    print(f"   Frame 0 - Min: {frames[0].min()}, Max: {frames[0].max()}")
    imageio.mimsave(filename, frames, fps=fps, loop=0)
    print(f"✅ Sucesso! Arquivo gerado: {filename}")

if __name__ == '__main__':
    try:
        main()
    except ImportError as e:
        print(f"❌ Erro de Biblioteca: {e}")
    except Exception as e:
        print(f"❌ Erro Inesperado: {e}")
        import traceback
        traceback.print_exc()
