function eval = erreval(predictions, realLabels)
     hit = 0;
     for i = 1:length(predictions)
         if predictions(i)==realLabels(i)
             hit = hit+1;
         end
     end
     eval=hit/length(realLabels);
end