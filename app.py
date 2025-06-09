import streamlit as st
import numpy as np
import plotly.graph_objects as go
import plotly.subplots as sp
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(
    page_title="Simulador RK4 - Sistemas Dinámicos",
    page_icon="🔬",
    layout="wide"
)

# === Función de forzamiento ===
def forzamiento(t, tipo, A, omega):
    """
    Calcula el forzamiento externo en función del tiempo
    
    Args:
        t: tiempo
        tipo: tipo de forzamiento ('sin', 'cos', 'const')
        A: amplitud
        omega: frecuencia angular
    
    Returns:
        Valor del forzamiento
    """
    if tipo == "sin":
        return A * np.sin(omega * t)
    elif tipo == "cos":
        return A * np.cos(omega * t)
    elif tipo == "const":
        return A
    else:
        return 0

# === Método Runge-Kutta de 4to orden ===
def runge_kutta_4(derivadas, condiciones_iniciales, h, N):
    """
    Implementa el método Runge-Kutta de 4to orden para resolver EDO
    
    Args:
        derivadas: función que define el sistema de EDO
        condiciones_iniciales: condiciones iniciales del sistema
        h: paso de integración
        N: número de iteraciones
    
    Returns:
        t: vector de tiempo
        y: matriz de soluciones
    """
    t = np.zeros(N)
    y = np.zeros((len(condiciones_iniciales), N))
    y[:,0] = condiciones_iniciales
    
    for i in range(N-1):
        t_i = t[i]
        y_i = y[:,i]
        
        k1 = h * derivadas(t_i, y_i)
        k2 = h * derivadas(t_i + 0.5*h, y_i + 0.5*k1)
        k3 = h * derivadas(t_i + 0.5*h, y_i + 0.5*k2)
        k4 = h * derivadas(t_i + h, y_i + k3)
        
        y[:,i+1] = y_i + (k1 + 2*k2 + 2*k3 + k4)/6
        t[i+1] = t_i + h
    
    return t, y

# === Análisis del tipo de amortiguamiento ===
def analizar_amortiguamiento(m, c, k):
    """
    Determina el tipo de amortiguamiento del sistema
    
    Args:
        m: masa (o inductancia en RLC)
        c: coeficiente de amortiguamiento (o resistencia en RLC)
        k: constante del resorte (o 1/C en RLC)
    
    Returns:
        Tipo de amortiguamiento como string
    """
    delta = c**2 - 4 * m * k
    if delta < 0:
        return "Subamortiguado"
    elif delta == 0:
        return "Críticamente amortiguado"
    else:
        return "Sobreamortiguado"

# === Función para crear gráficas ===
def crear_graficas(t, y, titulo_y, label_y, titulo_fase, label_x_fase, label_y_fase):
    """
    Crea las gráficas de respuesta temporal y diagrama de fase
    
    Args:
        t: vector de tiempo
        y: matriz de soluciones
        titulo_y: título para la gráfica temporal
        label_y: etiqueta del eje Y en gráfica temporal
        titulo_fase: título para el diagrama de fase
        label_x_fase: etiqueta del eje X en diagrama de fase
        label_y_fase: etiqueta del eje Y en diagrama de fase
    
    Returns:
        Figura de Plotly con subplots
    """
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=(titulo_y, titulo_fase),
        vertical_spacing=0.12
    )
    
    # Gráfica temporal
    fig.add_trace(
        go.Scatter(x=t, y=y[0,:], mode='lines', name=label_y, line=dict(color='blue')),
        row=1, col=1
    )
    
    # Diagrama de fase
    fig.add_trace(
        go.Scatter(x=y[0,:], y=y[1,:], mode='lines', name='Trayectoria de fase', line=dict(color='red')),
        row=2, col=1
    )
    
    # Configurar ejes
    fig.update_xaxes(title_text="Tiempo t", row=1, col=1)
    fig.update_yaxes(title_text=label_y, row=1, col=1)
    fig.update_xaxes(title_text=label_x_fase, row=2, col=1)
    fig.update_yaxes(title_text=label_y_fase, row=2, col=1)
    
    fig.update_layout(height=600, showlegend=True)
    
    return fig

# === Sistema Masa-Resorte ===
def sistema_masa_resorte():
    """
    Interfaz y simulación para el sistema masa-resorte-amortiguado
    """
    st.header("Sistema Masa-Resorte-Amortiguado")
    
    # Crear columnas para organizar los inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Parámetros del Sistema")
        m = st.number_input("Masa (m)", min_value=0.01, value=1.0, step=0.1, format="%.2f")
        c = st.number_input("Coeficiente de amortiguamiento (c)", min_value=0.0, value=0.5, step=0.1, format="%.2f")
        k = st.number_input("Constante del resorte (k)", min_value=0.01, value=1.0, step=0.1, format="%.2f")
        
        st.subheader("Condiciones Iniciales")
        x0 = st.number_input("Posición inicial (x₀)", value=1.0, step=0.1, format="%.2f")
        v0 = st.number_input("Velocidad inicial (v₀)", value=0.0, step=0.1, format="%.2f")
    
    with col2:
        st.subheader("Forzamiento Externo")
        tipo_forz = st.selectbox("Tipo de forzamiento", ["sin", "cos", "const"])
        A = st.number_input("Amplitud (A)", value=0.0, step=0.1, format="%.2f")
        omega = st.number_input("Frecuencia angular (ω)", value=1.0, step=0.1, format="%.2f")
        
        st.subheader("Parámetros Numéricos")
        N = st.slider("Número de iteraciones", min_value=10, max_value=199, value=100)
        h = st.slider("Paso de integración (h)", min_value=0.01, max_value=0.20, value=0.1, step=0.01)
    
    # Botón para ejecutar simulación
    if st.button("Ejecutar Simulación", type="primary"):
        # Validación de parámetros
        if N >= 200 or h > 0.2:
            st.error("Parámetros inválidos. N debe ser < 200 y h ≤ 0.2")
            return
        
        # Definimos las derivadas del sistema
        def derivadas(t, Y):
            x, v = Y
            dxdt = v
            dvdt = (1/m) * (forzamiento(t, tipo_forz, A, omega) - c*v - k*x)
            return np.array([dxdt, dvdt])
        
        # Ejecutamos RK4
        condiciones_iniciales = [x0, v0]
        t, y = runge_kutta_4(derivadas, condiciones_iniciales, h, N)
        
        # Crear y mostrar gráficas
        fig = crear_graficas(t, y, "Posición vs Tiempo", "Posición x(t)", 
                           "Diagrama de Fase", "Posición x", "Velocidad v")
        st.plotly_chart(fig, use_container_width=True)
        
        # Análisis de resultados
        col1, col2 = st.columns(2)
        with col1:
            tipo_amort = analizar_amortiguamiento(m, c, k)
            st.success(f"**Tipo de amortiguamiento:** {tipo_amort}")
        
        with col2:
            error_aprox = abs(y[0,-1] - y[0,-2])
            st.info(f"**Error de aproximación estimado:** {error_aprox:.6f}")
        
        # Mostrar algunos valores finales
        st.subheader("Valores Finales")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Posición final", f"{y[0,-1]:.4f}")
        with col2:
            st.metric("Velocidad final", f"{y[1,-1]:.4f}")
        with col3:
            st.metric("Tiempo final", f"{t[-1]:.4f}")

# === Sistema Circuito RLC ===
def sistema_rlc():
    """
    Interfaz y simulación para el circuito RLC
    """
    st.header("Circuito RLC")
    
    # Crear columnas para organizar los inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Parámetros del Circuito")
        L = st.number_input("Inductancia (L)", min_value=0.01, value=1.0, step=0.1, format="%.2f")
        R = st.number_input("Resistencia (R)", min_value=0.0, value=0.5, step=0.1, format="%.2f")
        C = st.number_input("Capacitancia (C)", min_value=0.01, value=1.0, step=0.1, format="%.2f")
        
        st.subheader("Condiciones Iniciales")
        q0 = st.number_input("Carga inicial (q₀)", value=1.0, step=0.1, format="%.2f")
        i0 = st.number_input("Corriente inicial (i₀)", value=0.0, step=0.1, format="%.2f")
    
    with col2:
        st.subheader("Forzamiento Externo")
        tipo_forz = st.selectbox("Tipo de forzamiento", ["sin", "cos", "const"], key="rlc_forz")
        A = st.number_input("Amplitud (A)", value=0.0, step=0.1, format="%.2f", key="rlc_A")
        omega = st.number_input("Frecuencia angular (ω)", value=1.0, step=0.1, format="%.2f", key="rlc_omega")
        
        st.subheader("Parámetros Numéricos")
        N = st.slider("Número de iteraciones", min_value=10, max_value=199, value=100, key="rlc_N")
        h = st.slider("Paso de integración (h)", min_value=0.01, max_value=0.20, value=0.1, step=0.01, key="rlc_h")
    
    # Botón para ejecutar simulación
    if st.button("Ejecutar Simulación", type="primary", key="rlc_button"):
        # Validación de parámetros
        if N >= 200 or h > 0.2:
            st.error("Parámetros inválidos. N debe ser < 200 y h ≤ 0.2")
            return
        
        # Definimos las derivadas del sistema
        def derivadas(t, Y):
            q, i = Y
            dqdt = i
            didt = (1/L) * (forzamiento(t, tipo_forz, A, omega) - R*i - q/C)
            return np.array([dqdt, didt])
        
        # Ejecutamos RK4
        condiciones_iniciales = [q0, i0]
        t, y = runge_kutta_4(derivadas, condiciones_iniciales, h, N)
        
        # Crear y mostrar gráficas
        fig = crear_graficas(t, y, "Carga vs Tiempo", "Carga q(t)", 
                           "Diagrama de Fase", "Carga q", "Corriente i")
        st.plotly_chart(fig, use_container_width=True)
        
        # Análisis de resultados (equivalencia con masa-resorte)
        col1, col2 = st.columns(2)
        with col1:
            m_equiv = L
            c_equiv = R  
            k_equiv = 1/C
            tipo_amort = analizar_amortiguamiento(m_equiv, c_equiv, k_equiv)
            st.success(f"**Tipo de amortiguamiento:** {tipo_amort}")
        
        with col2:
            error_aprox = abs(y[0,-1] - y[0,-2])
            st.info(f"**Error de aproximación estimado:** {error_aprox:.6f}")
        
        # Mostrar algunos valores finales
        st.subheader("Valores Finales")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Carga final", f"{y[0,-1]:.4f}")
        with col2:
            st.metric("Corriente final", f"{y[1,-1]:.4f}")
        with col3:
            st.metric("Tiempo final", f"{t[-1]:.4f}")

# === Aplicación Principal ===
def main():
    """
    Función principal de la aplicación Streamlit
    """
    st.title("🔬 Simulador RK4 - Sistemas Dinámicos")
    st.markdown("---")
    

    
    # Selector de sistema
    sistema = st.selectbox(
        "Selecciona el sistema a simular:",
        ["Sistema Masa-Resorte-Amortiguado", "Circuito RLC"],
        index=0
    )
    
    st.markdown("---")
    
    # Ejecutar el sistema seleccionado
    if sistema == "Sistema Masa-Resorte-Amortiguado":
        sistema_masa_resorte()
    elif sistema == "Circuito RLC":
        sistema_rlc()

if __name__ == "__main__":
    main()