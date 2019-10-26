#Metro project.      Benjamin lepers 26/01/2019


The goal is to visit all the paris station with one ticket valid for 20h = 72000s.
The connection between the node and the time duration between them
is given in the file metro_complet.txt

The program is Ben_lep_metro.py and pynb version is also provided.
in the .py file, the function to make the simulation with scan_initnode()
 function is desactive. In the file pynb, the simulation as already been done
The parameter p in the function min time, 
can be set to p = min(index), or max(index) or randint(min(index),max(index)).
with p = min(index) or max(index) the algo is deterministic.
p = min(index) with the initial starting node 176 (station madelein).

Remark: the use of the adjacent matrix helps me to program and facilitates the time calculation 
and the check of continuity. Computationnaly, and for large problem, using a matrix is 
probably less adapted.

Futur work:
 Would have been interesting to work further on the stations themselves and not the nodes, and
to implement or use an annealed algorithm in order to find a better solution.



