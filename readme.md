La data a usar es el dataset de kaggle con los partidos internacionales de 1872 a 2026, es un dataset conocido.

Se lo descarga y se lo descomprime en una carpeta llamada "1872-2026"

Luego se ejecuta el archivo transformate_data.py

Con esto creamos nuevas columnas a un nuevo dataset que creamos para entrenar el modelo, ademas de estado actual de los equipos

Luego se ejecuta el archivo model.py, este se encarga de entrenar el modelo con el dataset creado anteriormente. Solicita una confirmacion de nombre para guardar un modelo o no

Con el modelo ya creado, ejecutamos primero generate_1round_results.py para generar los resultados de la primera fecha en una carpeta predicts

Luego podemos generar un informe con get_round_topresults.py

