import DrawDecisionTree.*

D=readtable("./data.xlsx");
data = table2array(D);

trainD = data(1:2000, :);
testD = data(2001:3000, :);
tree = DesicionTreeTraining(trainD);
DrawDecisionTree(tree);

trainerr=erreval(predict(tree, trainD), trainD(:, end));
testerr=erreval(predict(tree, testD), testD(:, end));
fprintf(1, strcat("traning rmse:", num2str(trainerr),...
                "testing rmse", num2str(testerr), "\n"));
            
fprintf(1, "10fold go!\n")
for i=1:10

    [trainD10, validD] = getTenFoldData(data(1:2000, :), i);
    tree1 = DesicionTreeTraining(trainD10);
    DrawDecisionTree(tree1);
    trainerr=erreval(predict(tree, trainD10), trainD10(:, end));
    validerr=erreval(predict(tree, validD), validD(:, end));
    fprintf(1, strcat("traning rmse:", num2str(trainerr),...
                "testing rmse", num2str(validerr), "\n"));
end

function [trainData, validData] = getTenFoldData(data, i)
    foldSize = 20;
    trainData = [];
    for j=1:10
        if j==i
            validData = data(foldSize*(j-1)+1:foldSize*j,:);
        else
            trainData = [trainData; data(foldSize*(j-1)+1:foldSize*j,:)];
        end
    end
end    

function err=error(dataset)
    err = sum((dataset(:,end)-mean(dataset(:,end))).^2);
end

function [lData, rData]=splitData(data,fea,value)
    lData = data(data(:, fea)<value, :);
    rData = data(data(:, fea)>=value, :);
end

function tree = DesicionTreeTraining(data)
    feaName = ["AT", "V", "AP", "RH"];
    min_sample = 20;
    if length(data) <= min_sample
        tree = node("", {}, mean(data(:,end)), nan, nan);
    else
        [bestAttr, bestThre] = chooseAttribute(data);
        [lData, rData] = splitData(data, bestAttr, bestThre);
        lSubtree = DesicionTreeTraining(lData);
        rSubtree = DesicionTreeTraining(rData);
        op = strcat(feaName(bestAttr), '<', num2str(bestThre));
        tree = node(op, {lSubtree rSubtree}, nan, bestAttr, bestThre);
    end
end

function node = node(op, kids, prediction, attribute, threshold)
    node.op = op;
    node.kids = kids;
    node.prediction = prediction;
    node.attribute = attribute;
    node.threshold = threshold;
end
   
function [bestAttr, bestThre] = chooseAttribute(data)
    besterr=inf;
    bestAttr=0;
    bestThre=0;
    for fea=1:4
        for valueIdx = 1:size(data,1)
            value=data(valueIdx,fea);
            [set1,set2]=splitData(data,fea,value);
            if isempty(set1)
                continue;
            end
            now_err=error(set1)+error(set2);
            if now_err<besterr
                besterr=now_err;
                bestAttr=fea;
                bestThre=value;
            end
        end
    end
end

function predictions = predict(tree, data)
    predictions = zeros(length(data), 1);
    for i=1:length(data)
        predictions(i) = predictOnce(tree, data(i,1: end-1));
    end
end

function prediction = predictOnce(tree, sample)
    if isempty(tree.kids)
        prediction = tree.prediction;
    else
        if sample(tree.attribute) < tree.threshold
            prediction = predictOnce(tree.kids{1}, sample);
        else
            prediction = predictOnce(tree.kids{2}, sample);
        end
    end
end


function eval = erreval(predictions, realLabels)
    eval=sqrt(mean((predictions-realLabels).^2));
end


% function predict=predicttree(tree,data,pre)
% global pre
%     if ~(isnan(tree.prediction))
%         pre(end+1)=tree.prediction
%         predict=pre
%     else
%         ldata=data( data(:,tree.attribute)<tree.threshold,:)
%         rdata=data( data(:,tree.attribute)>=tree.threshold,:)
%         ltree=tree.kids{1}
%         rtree=tree.kids{2}
%         predicttree(ltree,ldata)
%         predicttree(rtree,rdata)
%     end
% end





            

    
        


