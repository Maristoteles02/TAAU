import numpy as np



# Transformar los valores de la columna age a valores numéricos
def map_age(age):
    """
    Maps age ranges to their corresponding values.

    Parameters:
        age (str): The age range to be mapped.

    Returns:
        int: The mapped value for the given age range. Returns np.nan if the age range is not recognized.
    """
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
    """
    Maps the gender string to a binary value.

    Args:
        gender (str): The gender string to be mapped.

    Returns:
        int: The binary value representing the gender. 1 for 'Male', 0 for 'Female'.
    """
    if gender == 'Male':
        return 1
    elif gender == 'Female':
        return 0
    

def map_readmitted(readmitted):
    if readmitted == 'NO':
        return 0
    elif readmitted == '<30':
        return 1
    elif readmitted == '>30':
        return 2 
    

def map_diabetes_med(diabetes_med):
    if diabetes_med == 'No':
        return 0
    elif diabetes_med == 'Yes':
        return 1
    else:
        return np.nan
    


#['No' 'Up' 'Steady' 'Down']



def map_change(change):
    if change == 'No':
        return 0
    elif change == 'Ch':
        return 1
    else:
        return np.nan
    
def map_diag(diag):
    """ICD9 codes for diagnoses."""
    if float(diag) in range(390, 460) or float(diag) == 785:
        return "circulatory"
    elif float(diag) in range(460, 520) or float(diag) == 786:
        return "respiratory"
    elif float(diag) in range(520, 580) or float(diag) == 787:
        return "digestive"
    elif float(diag) == 250:
        return "diabetes"
    elif float(diag) in range(800, 1000):
        return "injury"
    elif float(diag) in range(710, 740):
        return "musculoskeletal"
    elif float(diag) in range(580, 630) or float(diag) == 788:
        return "genitourinary"
    elif float(diag) in range(140, 240):
        return "neoplasms"
    elif float(diag) in range(630, 680):
        return "pregnancy"
    else:
        return "other"
