from csv import reader
from math import sqrt, exp, pi
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, conlist

# ------------------- Paso 1: Cargar CSV -------------------
def load_csv(filename):
    dataset = list()
    with open(filename, 'r', encoding='utf-8') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset

# ------------------- Paso 2: Convertir datos a enteros -------------------
def str_column_to_int(dataset, column):
    """Convierte columna de strings a enteros y devuelve el diccionario de mapeo"""
    class_values = [row[column] for row in dataset[1:]]  # ignorar encabezado
    unique = set(class_values)
    lookup = {value: i for i, value in enumerate(unique)}
    for row in dataset[1:]:
        row[column] = lookup[row[column]]
    return lookup

# ------------------- Paso 3: Resumen por clase -------------------
def separate_by_class(dataset):
    separated = dict()
    for row in dataset[1:]:  # ignorar encabezado
        class_value = row[-1]  # última columna = clase
        if class_value not in separated:
            separated[class_value] = list()
        # ignoramos columna id (columna 0)
        separated[class_value].append([float(x) for x in row[1:-1]])
    return separated

def mean(numbers):
    return sum(numbers) / float(len(numbers))

def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers) - 1)
    return sqrt(variance)

def summarize_dataset(dataset):
    summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
    return summaries

def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries

# ------------------- Paso 4: Probabilidad Gaussiana -------------------
def calculate_probability(x, mean, stdev):
    eps = 1e-6  # evitar división por cero
    stdev = max(stdev, eps)
    exponent = exp(-((x - mean) ** 2) / (2 * stdev ** 2))
    return (1 / (sqrt(2 * pi) * stdev)) * exponent

# ------------------- Paso 5: Probabilidad por clase -------------------
def calculate_class_probabilities(summaries, input_vector):
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = 1
        for i in range(len(class_summaries)):
            mean_val, stdev_val, _ = class_summaries[i]
            x = float(input_vector[i])
            probabilities[class_value] *= calculate_probability(x, mean_val, stdev_val)
    return probabilities

def predict(summaries, input_vector):
    probabilities = calculate_class_probabilities(summaries, input_vector)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label

# ------------------- ENTRENAMIENTO INICIAL -------------------
filename = "dataset_votacion.csv"
dataset = load_csv(filename)

# Convertir columnas a enteros, ignorando id (columna 0)
for i in range(1, len(dataset[0])):
    if i == len(dataset[0]) - 1:  # última columna = clase
        partido_lookup = str_column_to_int(dataset, i)
    else:
        str_column_to_int(dataset, i)

summaries = summarize_by_class(dataset)

# Cantidad de atributos (sin contar id ni clase)
num_attributes = len(dataset[0]) - 2

# ------------------- API con FastAPI -------------------
app = FastAPI()

# Endpoint raíz para comprobar que la API está viva
@app.get("/")
def root():
    return {"mensaje": "API corriendo correctamente"}

class Registro(BaseModel):
    datos: conlist(int, min_length=num_attributes, max_length=num_attributes)

@app.post("/predecir")
def predecir(registro: Registro):
    input_data = registro.datos
    if len(input_data) != num_attributes:
        raise HTTPException(status_code=400, detail=f"Se requieren {num_attributes} atributos")
    prediccion_num = predict(summaries, input_data)
    partido_real = [k for k, v in partido_lookup.items() if v == prediccion_num][0]
    return {"prediccion": partido_real}
