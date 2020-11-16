## Mixed Nash Equilibrium is All You Need in Adversarial Training


1. Apply double-oracle method to the adversarial training
2. Apply PGD attack to efficiently attack multiple classifiers
3. Apply continual learning to restrict the size of the neural networks


Basically, 

1. PGD attack is applied to the mixed classifiers with a distribution, where the loss is just a wighted sum of the 
loss of all classifiers. 
2. Then, use the XdG method for contextual gating scheme



