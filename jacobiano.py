import sympy as sp

def dh_matrix(theta, d, a, alpha):
    """Retorna a matriz de transformação homogênea de Denavit-Hartenberg tradicional."""
    return sp.Matrix([
        [sp.cos(theta), -sp.sin(theta)*sp.cos(alpha),  sp.sin(theta)*sp.sin(alpha), a*sp.cos(theta)],
        [sp.sin(theta),  sp.cos(theta)*sp.cos(alpha), -sp.cos(theta)*sp.sin(alpha), a*sp.sin(theta)],
        [0,              sp.sin(alpha),               sp.cos(alpha),               d],
        [0,              0,                           0,                           1]
    ])

def skew_symmetric(v):
    """Retorna a matriz anti-simétrica S(v) para substituir o produto vetorial."""
    return sp.Matrix([
        [0,     -v[2],  v[1]],
        [v[2],   0,    -v[0]],
        [-v[1],  v[0],  0]
    ])

def analisar_manipulador(nome, DH_tabela, tipos_juntas, q, dq, valores_numericos, buscar_matrizes):
    """Função genérica para calcular, validar e simular a cinemática diferencial."""
    print(f"\n" + "="*60)
    print(f" ANÁLISE DO MANIPULADOR: {nome} ".center(60, "="))
    print("="*60)

    # 1. Calcular Matrizes de Transformação via DH
    T = sp.eye(4)
    Ts = []
    zs = [sp.Matrix([0, 0, 1])]  # z0 inicial
    os = [sp.Matrix([0, 0, 0])]  # o0 inicial

    for theta, d, a, alpha in DH_tabela:
        A = dh_matrix(theta, d, a, alpha)
        T = sp.simplify(T * A)
        Ts.append(T)
        zs.append(sp.simplify(T[:3, 2]))
        os.append(sp.simplify(T[:3, 3]))

    o_n = os[-1] # Posição do efetuador final

    # 2. Impressão das Matrizes do Livro para Conferência
    print("\n--- MATRIZES INTERMEDIÁRIAS DO LIVRO (CONFERÊNCIA) ---")
    for rotulo, expressao in buscar_matrizes(os, zs).items():
        print(f"\n-> Matriz {rotulo} do livro:")
        sp.pprint(sp.simplify(expressao), use_unicode=True)

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

    # 4. Construção do Jacobiano (Método 2: Matrizes Anti-simétricas)
    Jv_skew, Jw_skew = [], []
    for i in range(len(DH_tabela)):
        if tipos_juntas[i] == 'R':
            Jv_skew.append(skew_symmetric(zs[i]) * (o_n - os[i]))
            Jw_skew.append(zs[i])
        else:
            Jv_skew.append(zs[i])
            Jw_skew.append(sp.Matrix([0, 0, 0]))
    J_skew = sp.Matrix.vstack(sp.Matrix.hstack(*Jv_skew), sp.Matrix.hstack(*Jw_skew))

    # Verificar se ambos os métodos coincidem perfeitamente
    assert sp.simplify(J_cross - J_skew) == sp.zeros(6, len(DH_tabela)), "Erro: Métodos divergem!"

    # 5. Impressão do Jacobiano Linha por Linha (Requisito do Enunciado)
    print("\n--- JACOBIANO GEOMÉTRICO (IMPRESSO LINHA POR LINHA) ---")
    J_simplificado = sp.simplify(J_cross)
    for r in range(6):
        print(f"Linha {r+1}:")
        sp.pprint(J_simplificado[r, :], use_unicode=True)

    # 6. Equação da Cinemática Diferencial (xi = J * dq)
    xi = J_simplificado * sp.Matrix(dq)
    
    # 7. Simulação Numérica
    xi_num = xi.evalf(subs=valores_numericos)
    print("\n--- SIMULAÇÃO NUMÉRICA ---")
    print("Vetor de velocidades operacionais calculadas xi = [v_x, v_y, v_z, w_x, w_y, w_z]^T:")
    sp.pprint(xi_num, use_unicode=True)


# =============================================================================
# CONFIGURAÇÃO DO EXEMPLO 4.7: STANFORD MANIPULATOR (R-R-P-R-R-R)
# =============================================================================
def executar_stanford():
    # Variáveis simbólicas das juntas e elos
    theta1, theta2, theta4, theta5, theta6 = sp.symbols('theta1 theta2 theta4 theta5 theta6', real=True)
    d3 = sp.symbols('d3', real=True) # Junta 3 é prismática
    d2, d6 = sp.symbols('d2 d6', real=True) # Constantes físicas dos elos
    
    # Velocidades articulares
    dt1, dt2, dd3, dt4, dt5, dt6 = sp.symbols('dtheta1 dtheta2 dd3 dtheta4 dtheta5 dtheta6', real=True)

    # Tabela DH do Stanford (Conforme Spong)
    DH_stanford = [
        (theta1,  0,  0, -sp.pi/2),
        (theta2, d2,  0,  sp.pi/2),
        (0,      d3,  0,  0),        # Prismática: d3 é variável, theta=0
        (theta4,  0,  0, -sp.pi/2),
        (theta5,  0,  0,  sp.pi/2),
        (theta6, d6,  0,  0)
    ]
    tipos = ['R', 'R', 'P', 'R', 'R', 'R']
    q = [theta1, theta2, d3, theta4, theta5, theta6]
    dq = [dt1, dt2, dd3, dt4, dt5, dt6]

    # Dicionário de valores para validação numérica
    valores = {
        theta1: sp.pi/6, theta2: sp.pi/4, d3: 0.5, theta4: 0, theta5: sp.pi/3, theta6: sp.pi/2,
        d2: 0.2, d6: 0.1,
        dt1: 1.0, dt2: 0.5, dd3: 0.2, dt4: 0.0, dt5: 0.8, dt6: 1.2
    }

    def extrair_matrizes_livro(os, zs):
        return {
            "o3 (Eq. 4.65)": os[3],
            "o6 (Eq. 4.64)": os[6],
            "z0 (Eq. 4.66)": zs[0],
            "z1 (Eq. 4.66)": zs[1],
            "z2 (Eq. 4.67)": zs[2],
            "z3 (Eq. 4.67)": zs[3],
            "z4 (Eq. 4.68)": zs[4],
            "z5 (Eq. 4.69)": zs[5]
        }

    analisar_manipulador("Stanford Manipulator (Exemplo 4.7)", DH_stanford, tipos, q, dq, valores, extrair_matrizes_livro)


# =============================================================================
# CONFIGURAÇÃO DO EXEMPLO 4.8: SCARA MANIPULATOR (R-R-P-R)
# =============================================================================
def executar_scara():
    # Variáveis simbólicas das juntas e elos
    theta1, theta2, theta4 = sp.symbols('theta1 theta2 theta4', real=True)
    d3 = sp.symbols('d3', real=True) # Junta 3 é prismática
    a1, a2, d4 = sp.symbols('a1 a2 d4', real=True) # Constantes físicas dos elos
    
    # Velocidades articulares
    dt1, dt2, dd3, dt4 = sp.symbols('dtheta1 dtheta2 dd3 dtheta4', real=True)

    # Tabela DH do SCARA (Para bater com os eixos invertidos z2 e z3 do livro Spong, usa-se -d3 e alpha2=pi)
    DH_scara = [
        (theta1,  0, a1, 0),
        (theta2,  0, a2, sp.pi),
        (0,     -d3,  0, 0),       # Prismática: translação negativa devido à inversão de z2 no Spong
        (theta4, d4,  0, 0)
    ]
    tipos = ['R', 'R', 'P', 'R']
    q = [theta1, theta2, d3, theta4]
    dq = [dt1, dt2, dd3, dt4]

    # Dicionário de valores para validação numérica
    valores = {
        theta1: sp.pi/4, theta2: sp.pi/6, d3: 0.3, theta4: sp.pi/3,
        a1: 0.5, a2: 0.5, d4: 0.1,
        dt1: 0.5, dt2: 0.5, dd3: -0.1, dt4: 1.0
    }

    def extrair_matrizes_livro(os, zs):
        return {
            "o1 (Eq. 4.71)": os[1],
            "o2 (Eq. 4.71)": os[2],
            "o4 (Eq. 4.72)": os[4],
            "z0": zs[0],
            "z1": zs[1],
            "z2": zs[2],
            "z3": zs[3]
        }

    analisar_manipulador("SCARA Manipulator (Exemplo 4.8)", DH_scara, tipos, q, dq, valores, extrair_matrizes_livro)


if __name__ == '__main__':
    executar_stanford()
    executar_scara()
