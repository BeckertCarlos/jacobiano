# Cinemática Diferencial (Stanford & SCARA)

Este repositório contém a implementação em Python para o cálculo computacional do **Jacobiano Geométrico** e a simulação da **Cinemática Diferencial** dos manipuladores **Stanford (6 DOF)** e **SCARA (4 DOF)**, com base no Capítulo 4 do livro *Robot Modeling and Control* (Spong, Hutchinson e Vidyasagar, 2ª Edição).

O código foi projetado para validar de forma analítica e numérica as equações do livro, utilizando a biblioteca simbólica **SymPy**, atendendo a todos os requisitos solicitados em aula.

---

## 🚀 Estrutura

O código está dividido em funções modulares que separam a matemática base da configuração de cada robô.

### 1. Importações e Funções Matemáticas Base

```python
import sympy as sp
```
* **`import sympy as sp`**: Importa a biblioteca SymPy, essencial para manipulação algébrica e simbólica. Ela nos permite trabalhar com variáveis como `theta` e `d` sem precisar de valores numéricos, gerando equações analíticas.

---

```python
def matriz_dh(theta, d, a, alpha):
    """Retorna a matriz de transformação homogênea de Denavit-Hartenberg tradicional."""
    return sp.Matrix([
        [sp.cos(theta), -sp.sin(theta)*sp.cos(alpha),  sp.sin(theta)*sp.sin(alpha), a*sp.cos(theta)],
        [sp.sin(theta),  sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],
        [0,              sp.sin(alpha),               sp.cos(alpha),               d],
        [0,              0,                           0,                           1]
    ])
```
* **`matriz_dh(...)`**: Implementa a matriz de transformação homogênea padrão de Denavit-Hartenberg (A_i). Ela recebe os 4 parâmetros cinemáticos (theta_i, d_i, a_i, alpha_i) e constrói a matriz 4x4 que rotaciona e translada o sistema de coordenadas do elo i-1 para o elo i.

---

```python
def anti_simetrica(v):
    """Retorna a matriz anti-simétrica S(v) para substituir o produto vetorial."""
    return sp.Matrix([
        [0,     -v[2],  v[1]],
        [v[2],   0,    -v[0]],
        [-v[1],  v[0],  0]
    ])
```
* **`anti_simetrica(v)`**: Cria a matriz anti-simétrica S(v) associada a um vetor tridimensional v. Essa estrutura é usada para substituir o produto vetorial tradicional (cross) por um produto puramente matricial, visto que:
  `z_{i-1} x (o_n - o_{i-1}) = S(z_{i-1}) * (o_n - o_{i-1})`

---

### 2. O Core Engine: `analisar_manipulador`

Esta função genérica processa qualquer robô a partir de sua tabela DH e do tipo de suas juntas.

```python
    # 1. Calcular Matrizes de Transformação via DH
    T = sp.eye(4)
    Ts = []
    zs = [sp.Matrix([0, 0, 1])]  # z0 inicial
    os = [sp.Matrix([0, 0, 0])]  # o0 inicial

    for theta, d, a, alpha in DH_tabela:
        A = matriz_dh(theta, d, a, alpha)
        T = sp.simplify(T * A)
        Ts.append(T)
        zs.append(sp.simplify(T[:3, 2]))
        os.append(sp.simplify(T[:3, 3]))

    o_n = os[-1] # Posição do efetuador final
```
* **`T = sp.eye(4)`**: Inicializa a matriz acumulada como uma matriz identidade 4x4 (referencial da base).
* **`zs` e `os`**: Listas que guardam os eixos de rotação/translação (z) e as origens dos sistemas (o). Inicializados com z0 = [0, 0, 1]^T e o0 = [0, 0, 0]^T.
* **`T = sp.simplify(T * A)`**: Multiplica sucessivamente as matrizes DH para obter a transformação acumulada T_i^0 = A_1 * A_2 * ... * A_i.
* **`T[:3, 2]`**: Extrai a terceira coluna da matriz de rotação (linhas 0 a 2, coluna 2), que representa o vetor unitário z_i.
* **`T[:3, 3]`**: Extrai a quarta coluna da matriz (linhas 0 a 2, coluna 3), que representa o vetor de posição da origem o_i.
* **`o_n = os[-1]`**: Captura a origem do último referencial, ou seja, a posição do efetuador final.

---

```python
    # 2. Impressão das Matrizes do Livro para Conferência
    print("\n--- MATRIZES INTERMEDIÁRIAS DO LIVRO (CONFERÊNCIA) ---")
    for rotulo, expressao in buscar_matrizes(os, zs).items():
        print(f"\n-> Matriz {rotulo} do livro:")
        sp.pprint(sp.simplify(expressao), use_unicode=True)
```
* **`buscar_matrizes(os, zs)`**: Uma função de callback passada por parâmetro que mapeia as origens e eixos calculados com as equações nominais do livro do Spong para garantir que o modelo DH adotado está 100% correto.

---

```python
    # 3. Construção do Jacobiano (Método 1: Produto Vetorial)
    Jv_cross, Jw_cross = [], []
    for i in range(len(DH_tabela)):
        if tipos_juntas[i] == 'R':  # Junta Rotativa
            Jv_cross.append(zs[i].cross(o_n - os[i]))
            Jw_cross.append(zs[i])
        else:                      # Junta Prismática
            Jv_cross.append(zs[i])
            Jw_cross.append(sp.Matrix([0, 0, 0]))
    J_cross = sp.Matrix.vstack(sp.Matrix.hstack(*Jv_cross), sp.Matrix.hstack(*Jw_cross))
```
* **`if tipos_juntas[i] == 'R'`**: Se a junta for rotativa, a velocidade linear depende do braço de alavanca (`zs[i].cross(o_n - os[i])`) e a angular recebe a contribuição direta do eixo (`zs[i]`).
* **`else` (Prismática)**: A velocidade linear é a própria direção de translação do eixo (`zs[i]`), e a angular é zero (`[0, 0, 0]^T`).
* **`sp.Matrix.vstack(...)`**: Agrupa as colunas horizontalmente (`hstack`) e depois empilha a parte linear (J_v) em cima da angular (J_w) verticalmente (`vstack`).

---

```python
    # 4. Construção do Jacobiano (Método 2: Matrizes Anti-simétricas)
    Jv_skew, Jw_skew = [], []
    for i in range(len(DH_tabela)):
        if tipos_juntas[i] == 'R':
            Jv_skew.append(anti_simetrica(zs[i]) * (o_n - os[i]))
            Jw_skew.append(zs[i])
        else:
            Jv_skew.append(zs[i])
            Jw_skew.append(sp.Matrix([0, 0, 0]))
    J_skew = sp.Matrix.vstack(sp.Matrix.hstack(*Jv_skew), sp.Matrix.hstack(*Jw_skew))

    # Verificar se ambos os métodos coincidem perfeitamente
    assert sp.simplify(J_cross - J_skew) == sp.zeros(6, len(DH_tabela)), "Erro: Métodos divergem!"
```
* **`anti_simetrica(zs[i]) * (o_n - os[i])`**: Faz o mesmo cálculo anterior, mas via multiplicação de matrizes pela anti-simétrica.
* **`assert ...`**: Uma trava de segurança. Garante que a subtração de ambos os Jacobianos resulta em uma matriz nula, provando que os dois métodos matemáticos implementados chegam exatamente no mesmo lugar.

---

```python
    # 5. Impressão do Jacobiano Linha por Linha (Requisito do Enunciado)
    print("\n--- JACOBIANO GEOMÉTRICO (IMPRESSO LINHA POR LINHA) ---")
    J_simplificado = sp.simplify(J_cross)
    for r in range(6):
        print(f"Linha {r+1}:")
        sp.pprint(J_simplificado[r, :], use_unicode=True)
```
* **`for r in range(6):`**: Itera pelas 6 linhas do Jacobiano (v_x, v_y, v_z, w_x, w_y, w_z) imprimindo cada uma de forma isolada e limpa através do `sp.pprint` (pretty print).

---

```python
    # 6. Equação da Cinemática Diferencial (xi = J * dq)
    xi = J_simplificado * sp.Matrix(dq)
    
    # 7. Simulação Numérica
    xi_num = xi.evalf(subs=valores_numericos)
    print("\n--- SIMULAÇÃO NUMÉRICA ---")
    print("Vetor de velocidades operacionais calculadas xi = [v_x, v_y, v_z, w_x, w_y, w_z]^T:")
    sp.pprint(xi_num, use_unicode=True)
```
* **`xi = J_simplificado * sp.Matrix(dq)`**: Implementa a equação fundamental da cinemática diferencial xi = J(q) * dq.
* **`xi.evalf(subs=valores_numericos)`**: Substitui os símbolos algébricos por valores do mundo real (ângulos, extensões de elos e velocidades articulares) para simular o comportamento numérico do robô em funcionamento real.

---

### 3. Configuração dos Modelos (Stanford e SCARA)

O restante do script apenas define as tabelas DH específicas extraídas do livro para alimentar a função principal.

```python
    # Tabela DH do Stanford (Conforme Spong)
    DH_stanford = [
        (theta1,  0,  0, -sp.pi/2),
        (theta2, d2,  0,  sp.pi/2),
        (0,      d3,  0,  0),        # Prismática: d3 é variável, theta=0
        (theta4,  0,  0, -sp.pi/2),
        (theta5,  0,  0,  sp.pi/2),
        (theta6, d6,  0,  0)
    ]
```
* Define a cinemática do **Manipulador de Stanford** (R-R-P-R-R-R). Note que a junta 3 possui parâmetro `0` em theta e a variável simbólica `d3` em d, denotando sua natureza prismática.

```python
    # Tabela DH do SCARA
    DH_scara = [
        (theta1,  0, a1, 0),
        (theta2,  0, a2, sp.pi),
        (0,     -d3,  0, 0),       # Prismática: d3 invertido conforme Spong
        (theta4, d4,  0, 0)
    ]
```
* Define a cinemática do **SCARA** (R-R-P-R). Nota importante para a defesa oral: O Spong define o eixo z_2 apontando para baixo (ao contrário da maioria dos livros). Por causa disso, a torção do elo anterior vira alpha_2 = pi e a variável da junta prismática entra invertida (`-d3`) para que o vetor posição do efetuador reflita exatamente a expressão d_3 - d_4 do livro.

---

## 🛠️ Como Executar o Código

### Pré-requisitos
Certifique-se de ter o Python 3 e a biblioteca `sympy` instalados:
```bash
pip install sympy
```

### Execução
Rode o script diretamente pelo terminal:
```bash
python cinematica_diferencial.py
```

O programa irá exibir no terminal as matrizes do livro para conferência, o Jacobiano impresso linha por linha para ambos os robôs e os vetores de velocidade operacional calculados para o cenário simulado.