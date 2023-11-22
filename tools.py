import numpy as np



# Transformar los valores de la columna age a valores numéricos
def map_age(age):
    if age == '[0-10)':
        return 5
    elif age == '[10-20)':
        return 15
    elif age == '[20-30)':
        return 25
    elif age == '[30-40)':
        return 35
    elif age == '[40-50)':
        return 45
    elif age == '[50-60)':
        return 55
    elif age == '[60-70)':
        return 65
    elif age == '[70-80)':
        return 75
    elif age == '[80-90)':
        return 85
    elif age == '[90-100)':
        return 95
    else:
        return np.nan
    
def map_gender(gender):
    if gender == 'Male':
        return 1
    elif gender == 'Female':
        return 0
    

