Se analiza por redes las capas que escriben en el buffer teniendo en cuenta algunos elementos como:
total de activaciones de las capas, las activaciones con fallos, la cantidad de errores en cada capa, el accurancy obtenido
al inyectar esos errores, diferencia entre los outputs  por capas entre el modelo con fallos y sin fallos, y las activaciones en 0 así como los porcentajes de la mismas del total de activaciones respectivamente.


Se realiza un segundo experimento donde en lugar de tener en cuenta el valor obtenido en las activaciones al inyectar fallos, se coloca en esa posición(Tensor obtenido): para ello se envia en la variable
index_list, la columna y fila siguiente a la que estará dañada, tomandose la posición de este valor y luego en el tensor resultante se sustituye el valor que tendrá error(index_list del valor que tentrá fallo) por el valor 
 valor siguiente , esto se hace con el fin de analizar el comportamiento del accurancy, en aras de buscar alternativas para dar solución al problema principal, del análisis en cuestión. 