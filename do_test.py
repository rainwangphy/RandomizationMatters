import nashpy

A = [[1, 2], [3, 0]]
B = [[0, 2], [3, 1]]
game = nashpy.Game(A, B)
for eq in game.support_enumeration():
    print(eq)
