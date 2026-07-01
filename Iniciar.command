#!/bin/bash
# Ir al directorio donde está guardado este script
cd "$(dirname "$0")"

echo "=========================================================="
echo "      INICIANDO YOUTUBE AUDIO REMASTERER CON IA           "
echo "=========================================================="
echo ""

# 1. Verificar si existe el entorno virtual (venv)
if [ ! -d "venv" ]; then
    echo "[1/3] Creando entorno virtual aislado (venv)..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "ERROR: No se pudo crear el entorno virtual. Asegúrate de tener Python 3 instalado."
        read -p "Presiona ENTER para salir..."
        exit 1
    fi
fi

# 2. Activar venv e instalar/verificar dependencias
echo "[2/3] Verificando dependencias (esto puede tardar la primera vez)..."
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt --prefer-binary

if [ $? -ne 0 ]; then
    echo "ERROR: Falló la instalación de dependencias de Python."
    read -p "Presiona ENTER para salir..."
    exit 1
fi

# 3. Compilar el servidor de Rust si no existe
if [ ! -f "target/release/remaster_audio" ]; then
    echo "Compilando el motor en Rust para máxima velocidad (solo la primera vez)..."
    # Asegurar que las variables de entorno de Rust estén en el PATH
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi
    cargo build --release
    if [ $? -ne 0 ]; then
        echo "ERROR: Falló la compilación del motor en Rust."
        read -p "Presiona ENTER para salir..."
        exit 1
    fi
fi

# 4. Iniciar el servidor web y abrir el navegador
echo "[3/3] Iniciando servidor y abriendo Dashboard..."
echo "Abre tu navegador en: http://localhost:8000"
echo ""
echo "Para apagar el servidor, cierra esta ventana de Terminal o presiona CTRL + C"
echo "=========================================================="

# Abrir el navegador por defecto
open "http://localhost:8000"

# Iniciar la aplicación en Rust
./target/release/remaster_audio
