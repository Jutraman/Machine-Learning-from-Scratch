function [trainData, validData] = getTenFoldData(nursery, i)
    foldSize = 900;
    trainData = [];
    for j=1:10
        if j==i
            validData = nursery(foldSize*(j-1)+1:foldSize*j,:);
        else
            trainData = [trainData; nursery(foldSize*(j-1)+1:foldSize*j,:)];
        end
    end
end 