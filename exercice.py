#!/usr/bin/env python
# -*- coding: utf-8 -*-


# TODO: Importez vos modules ici
import numpy as np
from scipy.integrate import quad    # pour les intégrales
from sympy import integrate
import math
from cmath import polar
import matplotlib.pyplot as plt

#1. Créer un array présentant 64 valeurs uniformément réparties entre -1.3 et 2.5.
# TODO: Définissez vos fonctions ici (il en manque quelques unes)
def linear_values() -> np.ndarray:
    return np.linspace(-1.3, 2.5, 64)

# ---------------------------------------------------------------------------------------#

# 2. Créer une fonction qui convertit une liste de coordonnées cartésiennes (x, y) en coordonnées polaires (rayon, angle)
def coordinate_conversion(cartesian_coordinates: np.ndarray) -> np.ndarray:
    #return np.array([polar(coord) for coord in cartesian_coordinates])

    # Solution vu en classe
    a = np.zeros([len(cartesian_coordinates), 2])

    for i in range(len(cartesian_coordinates)):
        r = np.sqrt(cartesian_coordinates[i][0] ** 2 + cartesian_coordinates[i][1] ** 2)
        angle = np.arctan2(cartesian_coordinates[i][1], cartesian_coordinates[i][0])
        polar_coordinate = (r, angle)
        a[i] = polar_coordinate
    return a
    '''
    # Fonctionne pour 1 coordonnée
    rayon = math.sqrt((cartesian_coordinates[0]**2 + cartesian_coordinates[1]**2)) # pythagore
    angle = math.atan(cartesian_coordinates[1]/cartesian_coordinates[0])    # angle = arctan(b/a)

    return np.array([rayon, angle])
    '''


# 3. Créer un programme qui trouve l’index de la valeur la plus proche d’un nombre fournit dans un array.
def find_closest_index(values: np.ndarray, number: float) -> int:
    # Calcule les différences entre le nombre et toutes les valeurs
    difference = [abs(values[i] - number) for i in range(np.size(values))]

    return np.argmin(difference)    # Trouve l'indice de l'élément avec la plus petite différence

# ---------------------------------------------------------------------------------------#

# EXERCICE 4
# Créons une fonction avec la fonction sin
def fonction_sin(x):
    '''
    Calcule image de x. Mais on peut faire une meilleure fonction.
    '''
    return x**2 * math.sin(1 / x**2) + x

def fonction_sin_efficace(x:np.ndarray) ->np.array:
    '''
    Méthode plus efficace pour calculer la fonction sin en utilisant numpy.
    Calcule tous les sins d'un seul coup.
    '''
    return x**2 * np.sin(1 / x**2) + x

# 4. Créer un graphe de y=𝑥^2  sin⁡(1∕𝑥^2 )+𝑥 dans l’intervalle [-1, 1] avec 250 points.
def draw_graph(fonction, intervalle, n_points):
    # Trouvons les coordonnées x et y
    x = np.linspace(intervalle[0], intervalle[1], n_points)
    #y = [fonction_sin(valeur) for valeur in x]
    y = fonction_sin_efficace(x)

    # Créons le graphique
    graphique = plt.plot(x,y)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.title(f'Graphique de la fonction {fonction} pour {n_points} points de l\'intervalle {intervalle}')
    plt.setp(graphique, 'color', 'g')   # changer la couleur de la ligne du graphique
    plt.show()

# ---------------------------------------------------------------------------------------#
# EXERCICE 5
def fonction_expo(x):
    return math.exp(-x**2)

# 5. Évaluer l’intégrale ∫_(−∞)^∞ 𝑒^(−𝑥^2) 𝑑𝑥. Afficher dans un graphique ∫𝑒^(−𝑥^2) 𝑑𝑥 pour x = [-4, 4].
def calculer_integrale(fonction, borne_inf, borne_sup):
    # Calculons l'estimée de l'intégrale
    integrale, erreur = quad(fonction, borne_inf, borne_sup)
    print(f'L\'intégrale e^(-x^2) évaluée entre les bornes {borne_inf} et {borne_sup} vaut : {integrale} avec une erreur de {erreur}')
    return integrale, erreur

def dessiner_graphe(intervalle):
    # Calculons le graphique
    n_points = 50
    x = np.linspace(intervalle[0], intervalle[1], n_points)
    y = [integrate(fonction_expo, valeur) for valeur in x]
    print(x, y)
    '''
    # Créons le graphique
    graphique = plt.plot(x,y)
    plt.ylabel('y')
    plt.xlabel('x')
    plt.title(f'Graphique de l\'intégrale e^(-x^2) pour {n_points} points de l\'intervalle {intervalle}')
    plt.setp(graphique, 'color', 'g')   # changer la couleur de la ligne du graphique
    plt.show()
    '''

if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici
    # Exercice 1
    print(linear_values())

    # Exercice 2
    coordonnee_cartesienne = np.array([[5,5], [1,2]])    # (x,y)
    print(f'Les coordonnées polaires (rayon, angle) de {coordonnee_cartesienne} (x,y) sont : {coordinate_conversion(coordonnee_cartesienne)}')

    # Exercice 3
    valeurs = np.array([1,4,6,8,10,14,20])
    nombre = 3
    print(f'Dans les valeurs suivantes : {valeurs}, l\'indice de la valeur la plus proche à {nombre} est : {find_closest_index(valeurs, nombre)}')

    # Exercice 4
    #draw_graph('y = x^^2 sin(1/x^2) + x', [-1, 1], 250)

    # Exercice 5
    # à compléter!
    borne_inf = np.NINF
    borne_sup = np.Infinity
    intervalle = [-4,4]
    calculer_integrale(fonction_expo, borne_inf, borne_sup)
    dessiner_graphe(intervalle)


