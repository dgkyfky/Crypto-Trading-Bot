#!/usr/bin/env bash
# --------------------------------------------------------------
# run_strategy.sh  –  Lanza BTC1D_GaussianChannel.py y guarda log
# --------------------------------------------------------------

set -euo pipefail                      # aborta si algo falla
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cd "$DIR"                              # entra en la carpeta del proyecto

# 1) Cargar .env (si existe)
# ▒▒ 1. Cargar variables de entorno desde .env
if [[ -f ".env" ]]; then
  set -a          # exporta automáticamente cada variable que se defina
  source ".env"   # interpreta el archivo como Bash
  set +a
fi

# 2) (opcional) activar entorno virtual
# source venv/bin/activate

# 3) Ejecutar el script de Python
/usr/local/bin/python3 "$DIR/BTC1D_GaussianChannel.py" >> "$DIR/cron.log" 2>&1



#Testear en consola: ./run_strategy.sh  

#En terminal dar permisos:
# Añade lectura para tu usuario y para los demás
#chmod u+r,g+r,o+r run_strategy.sh      # ó, de forma tradicional:
#chmod 755 run_strategy.sh              # rwx r-x r-x
# (Opcional) Elimina atributos/ACL sospechosos
#xattr -d com.apple.quarantine run_strategy.sh 2>/dev/null || true
#chmod -N run_strategy.sh               # quita ACL si existía


#Editar el crontab ==> en terminal: crontab -e y edito el * * * * *
# Moverlo afuera de Documents por un tema de permisos
