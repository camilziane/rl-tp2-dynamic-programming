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
    if n % 2 == 1:
        return 0
    p, t = 3, 0
    for _ in range(n // 2 - 1):
        p, t = p * 3 + t * 2 + 2, t + p
    return p


# print(domino_paving(1))
# print(domino_paving(2))
# print(domino_paving(3))
# print(domino_paving(4))
# print(domino_paving(5))
