load nurserynumric
ctree=fitctree(nurserynumric(:,1:8),nurserynumric(:,9))
view(ctree) % text description
view(ctree,'mode','graph') 
