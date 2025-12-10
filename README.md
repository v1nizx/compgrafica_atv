# 📸 Computação Gráfica e Processamento de Imagens - Exercícios Práticos

Este repositório contém a implementação de exercícios práticos focados em **Processamento Digital de Imagens (PDI)** e **Modelagem de Iluminação (Computação Gráfica)**. As atividades exploram desde a implementação manual de algoritmos de convolução até a simulação de modelos físicos de luz.

## 🛠️ Tecnologias Utilizadas

  * **Linguagem:** Python 3
  * **Bibliotecas:**
      * `numpy` (Manipulação matricial e álgebra linear)
      * `matplotlib` (Visualização de dados e plotagem 2D/3D)
      * `opencv-python` (Visão computacional)
      * `scikit-image` (Processamento de imagens e datasets de exemplo)

-----

## 📂 Parte 1: Filtros Digitais e Processamento de Imagens

O objetivo desta etapa foi compreender o funcionamento matemático dos filtros digitais, tanto no domínio espacial quanto no domínio da frequência.

### 1\. Implementação Manual de Convolução

Implementação "from scratch" (sem bibliotecas de PDI) para entender a matemática por trás dos kernels.

  * **Filtro Laplaciano:** Detecção de bordas baseada na segunda derivada.
  * **Filtros de Sobel (X e Y):** Detecção de bordas direcionais (horizontais e verticais).
  * **Normalização:** Aplicação da técnica Min-Max para exibir resultados no intervalo [0, 255].

### 2\. Filtros com OpenCV e Scikit-Image

Uso de bibliotecas otimizadas para aplicar e analisar:

  * **Filtro Negativo e Limiarização (Thresholding):** Segmentação simples.
  * **Suavização Gaussiana:** Redução de ruído.
  * **Análise de Histograma:** Verificação de contraste.

### 3\. Domínio da Frequência (Transformada de Fourier)

Análise espectral de imagens utilizando a FFT (Fast Fourier Transform).

  * Visualização do espectro de magnitude com e sem deslocamento (*fftshift*).
  * **Filtros Passa-Alta:** Realce de bordas no domínio da frequência.
  * **Filtros Passa-Baixa (Gaussiano):** Suavização no domínio da frequência.
  * **Compressão DCT:** Demonstração de perda de informação ao zerar coeficientes de alta frequência.

-----

## 💡 Parte 2: Iluminação e Tonalização (Modelo de Phong)

Simulação de interação da luz com superfícies utilizando o Modelo de Reflexão de Phong e técnicas de sombreamento (*shading*).

### 1\. Modelo de Iluminação de Phong

Implementação vetorial da equação de Phong, considerando três componentes:

  * 🔴 **Ambiente:** Luz base constante.
  * 🟢 **Difusa:** Luz dependente do ângulo de incidência (Lambert).
  * 🔵 **Especular:** O brilho "metálico" ou "plástico" dependente do ângulo de visão.

### 2\. Tonalização: Flat vs. Gouraud

Comparação visual entre métodos de preenchimento de polígonos:

  * **Constant Shading (Flat):** Uma cor única por polígono (aparência facetada).
  * **Gouraud Shading:** Interpolação das cores calculadas nos vértices (aparência suave/3D).

### 3\. Desafios de Renderização

Soluções para cenários específicos propostos:

  * **Materiais:** Simulação de materiais foscos (borracha) vs. brilhantes (plástico) alterando o expoente especular ($s$).
  * **Geometria Complexa:** Renderização de um Hexágono com simulação de curvatura nas normais.
  * **Texturização:** Mapeamento de textura (imagem) dentro de um polígono utilizando *Clipping Paths*.

-----

## 🚀 Como Executar

Certifique-se de ter as dependências instaladas:

```bash
pip install numpy matplotlib opencv-python scikit-image
```

Os scripts foram desenvolvidos para rodar preferencialmente em ambientes Jupyter Notebook (como **Google Colab**) para melhor visualização dos gráficos gerados pelo `matplotlib`.

-----

## 📝 Autor

Desenvolvido por **Marcos Vinicius** como parte de atividades acadêmicas de Computação Gráfica e PDI.

-----
