# Exercice 3 : pavage d'un rectangle avec des dominos
# ---------------------------------------------------
# On considère un rectangle de dimensions 3xN, et des dominos de
# dimensions 2x1. On souhaite calculer le nombre de façons de paver le
# rectangle avec des dominos.

# Ecrire une fonction qui calcule le nombre de façons de paver le
# rectangle de dimensions 3xN avec des dominos.
# Indice: trouver une relation de récurrence entre le nombre de façons
# de paver un rectangle de dimensions 3xN et le nombre de façons de
# paver un rectangle de dimensions 3x(N-1), 3x(N-2) et 3x(N-3).


def domino_paving(n: int) -> int:
    """
    Calcule le nombre de façons de paver un rectangle de dimensions 3xN
    avec des dominos.
    """
    # BEGIN SOLUTION
    if n == 1:
      return 2
    if n == 2:
      return 3 * domino_paving(n-1)
    else:
      return 6 + domino_paving(n-1)
    # END SOLUTION

# print(domino_paving(1))
# print(domino_paving(2))
# print(domino_paving(3))
# print(domino_paving(4))
# print(domino_paving(5))
