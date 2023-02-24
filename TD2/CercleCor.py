# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 11:49:43 2021

@author: guyadern
"""
# corvar est de dimension (n,2) : contient dans la colonne 0 : la corrélation entre la composante principale 1 et les variables de départ 
# et dans la colonne 1 la corrélation entre la composante principale 2 et les variables de départ

# Cercle des corrélations
fig, axes = plt.subplots(figsize=(8,8))
axes.set_xlim(-1,1)
axes.set_ylim(-1,1)

# On ajoute les axes
plt.plot([-1,1],[0,0],color='silver',linestyle='-',linewidth=1)
plt.plot([0,0],[-1,1],color='silver',linestyle='-',linewidth=1)
# On ajoute un cercle
cercle = plt.Circle((0,0),1,color='blue',fill=False)
axes.add_artist(cercle)
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.title('Cercle des corrélations')
plt.scatter(corvar[:,0],corvar[:,1])
#affichage des étiquettes (noms des variables)
for j in range(p):
  plt.annotate(X.columns[j],(corvar[j,0],corvar[j,1]))

plt.show()