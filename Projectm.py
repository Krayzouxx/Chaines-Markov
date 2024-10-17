from Shadock import simulation, stochastique, puits, stationnaire, Diagonalisation
import numpy as np
import matplotlib.pyplot as plt

# Définition de la matrice de transition P et du vecteur initial pi0
P = np.array([[5 / 6, 1 / 12, 1 / 12], [1 / 4, 1 / 2, 1 / 4], [1 / 4, 0, 3 / 4]])

# Q1: Vérification de la connectivité et de l'aperiodicité (réponse théorique)
print("Q1: La chaîne est fortement connexe et aperiodique car il existe une possibilité de transition entre les états.")

# Q2: Vérification que la matrice est stochastique
print("\nQ2: Vérification de la matrice stochastique :")
print("La matrice est-elle stochastique ?", stochastique(P))

# Q3: Recherche d'un état absorbant (puits)
print("\nQ3: Recherche d'un état absorbant (puits) :")
for i in range(len(P)):
    print(f"L'état {i+1} est-il un puits ?", puits(P, i))

# Q4: Simulation de l'évolution des probabilités dans le temps
print("\nQ4: Simulation de l'évolution des probabilités :")
pi0 = [1, 0, 0]
t, pi = simulation(P, pi0, 0, 100)

# Affichage des probabilités en fonction du temps
plt.figure(figsize=(10, 6))
plt.plot(t, pi[:, 0], label="Bonne santé (État 1)", color='green')
plt.plot(t, pi[:, 1], label="Enrhumé (État 2)", color='orange')
plt.plot(t, pi[:, 2], label="Malade (État 3)", color='red')

# Ajout des titres et légendes
plt.title("Évolution des probabilités de M. Shadock dans le temps", fontsize=14)
plt.xlabel("Temps (jours)", fontsize=12)
plt.ylabel("Probabilité", fontsize=12)
plt.legend(loc='best')
plt.grid(True)

# Affichage du graphique
plt.show()

# Q5: Calcul de la probabilité stationnaire
print("\nQ5: Calcul de la probabilité stationnaire :")
pi_stationnaire = stationnaire(P)
print("La distribution stationnaire est :", pi_stationnaire)

# Q6: Diagonalisation de la matrice de transition
print("\nQ6: Vérification de la diagonalisation de la matrice :")
is_diagonalizable, V, D = Diagonalisation(P)
print("La matrice est-elle diagonalizable ?", is_diagonalizable)
print("Valeurs propres :", np.diag(D))

# Q7: Vérification que λ = 1 est une valeur propre
print("\nQ7: Valeurs propres et comportement asymptotique :")
print("Valeurs propres de P :", np.diag(D))
print("La valeur propre λ=1 est présente :", 1 in np.diag(D))

# Q8: Simulation avec différentes conditions initiales
print("\nQ8: Simulation avec une condition initiale différente :")
pi0_different = [0.4, 0.2, 0.4]  # Nouvelle condition initiale
[t, pi] = simulation(P, pi0_different, 0, 100)
print(f"Probabilité après 5 jours : {pi[5]}")
print(f"Probabilité après 10 jours : {pi[10]}")
print(f"Probabilité après 50 jours : {pi[50]}")
print(f"Probabilité après 100 jours : {pi[100]}")

# Q9: Convergence vers la même distribution stationnaire pour différentes conditions initiales
print("\nQ9: Convergence vers la distribution stationnaire :")
print("On observe que quelle que soit la condition initiale, la probabilité converge vers la même distribution.")

# Q10: Observation de la rapidité de convergence
print("\nQ10: Observation de la rapidité de convergence :")
lambda_2 = sorted(np.abs(np.diag(D)))[-2]  # 2ème plus grande valeur propre
print("Le module de la 2ème plus grande valeur propre est :", lambda_2)
print("Plus cette valeur est petite, plus la convergence est rapide.")
