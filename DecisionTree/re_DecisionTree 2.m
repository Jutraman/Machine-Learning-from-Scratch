import DrawDecisionTree.*
nursery = importfile('nursery_numric.txt',1,12960);
nursery = nursery(randperm(length(nursery)),:);
trainD = nursery(2000:4000, :);
testD = nursery(9001:12959, :);
tree = DecisionTreeTraining(trainD);
DrawDecisionTree(tree, "My tree");

trainerr=erreval(predict(tree, trainD), trainD(:, end));
testerr=erreval(predict(tree, testD), testD(:, end));
train_recallrate=recall(predict(tree, trainD), trainD(:, end));
test_recallrate=recall(predict(tree, testD), testD(:, end));
f1measure=2*trainerr*train_recallrate/(trainerr+train_recallrate);
fprintf(1, strcat("traning accuracy:", num2str(trainerr),...
                 "\ntesting accuracy:", num2str(testerr),... 
                 "\ntrain_recall_rate:", num2str(train_recallrate),... 
                "\ntest_recall_rate:", num2str(test_recallrate), ...
                "\nF1 measure:",num2str(f1measure),"\n\n"));
%10fold           
fprintf(1, "10fold go!\n")
for i=1:10

    [trainD10, validD] = getTenFoldData(nursery, i);
    tree1 = DecisionTreeTraining(trainD10);
    DrawDecisionTree(tree1);
    trainerr=erreval(predict(tree, trainD10), trainD10(:, end));
    validerr=erreval(predict(tree, validD), validD(:, end));
    train_recallrate=recall(predict(tree, trainD10), trainD10(:, end));
    valid_recallrate=recall(predict(tree, validD), validD(:, end));
    f1measure=2*trainerr*train_recallrate/(trainerr+train_recallrate);
    %plot(i,f1measure);
    fprintf('It is %d time\n',i)
    fprintf(1, strcat("traning accuracy:", num2str(trainerr),...
                 "\nvalid accuracy:", num2str(validerr),... 
                 "\ntrain_recall_rate:", num2str(train_recallrate),... 
                "\ntest_recall_rate:", num2str(valid_recallrate), ...
                "\nF1 measure:",num2str(f1measure),"\n\n"));
end


function [trainData, validData] = getTenFoldData(nursery, i)
    foldSize = 1290;
    trainData = [];
    for j=1:10
        if j==i
            validData = nursery(foldSize*(j-1)+1:foldSize*j,:);
        else
            trainData = [trainData; nursery(foldSize*(j-1)+1:foldSize*j,:)];
        end
    end
end    

function nuserynumric = importfile(filename, startRow, endRow)

%% Initialize variables.
delimiter = '\t';
if nargin<=2
    startRow = 1;
    endRow = inf;
end

%% Read columns of data as text:
% For more information, see the TEXTSCAN documentation.
formatSpec = '%q%q%q%q%q%q%q%q%q%[^\n\r]';

%% Open the text file.
fileID = fopen(filename,'r');

%% Read columns of data according to the format.
% This call is based on the structure of the file used to generate this
% code. If an error occurs for a different file, try regenerating the code
% from the Import Tool.
dataArray = textscan(fileID, formatSpec, endRow(1)-startRow(1)+1, 'Delimiter', delimiter, 'TextType', 'string', 'HeaderLines', startRow(1)-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
for block=2:length(startRow)
    frewind(fileID);
    dataArrayBlock = textscan(fileID, formatSpec, endRow(block)-startRow(block)+1, 'Delimiter', delimiter, 'TextType', 'string', 'HeaderLines', startRow(block)-1, 'ReturnOnError', false, 'EndOfLine', '\r\n');
    for col=1:length(dataArray)
        dataArray{col} = [dataArray{col};dataArrayBlock{col}];
    end
end

%% Close the text file.
fclose(fileID);

%% Convert the contents of columns containing numeric text to numbers.
% Replace non-numeric text with NaN.
raw = repmat({''},length(dataArray{1}),length(dataArray)-1);
for col=1:length(dataArray)-1
    raw(1:length(dataArray{col}),col) = mat2cell(dataArray{col}, ones(length(dataArray{col}), 1));
end
numericData = NaN(size(dataArray{1},1),size(dataArray,2));

for col=[1,2,3,4,5,6,7,8,9]
    % Converts text in the input cell array to numbers. Replaced non-numeric
    % text with NaN.
    rawData = dataArray{col};
    for row=1:size(rawData, 1)
        % Create a regular expression to detect and remove non-numeric prefixes and
        % suffixes.
        regexstr = '(?<prefix>.*?)(?<numbers>([-]*(\d+[\,]*)+[\.]{0,1}\d*[eEdD]{0,1}[-+]*\d*[i]{0,1})|([-]*(\d+[\,]*)*[\.]{1,1}\d+[eEdD]{0,1}[-+]*\d*[i]{0,1}))(?<suffix>.*)';
        try
            result = regexp(rawData(row), regexstr, 'names');
            numbers = result.numbers;
            
            % Detected commas in non-thousand locations.
            invalidThousandsSeparator = false;
            if numbers.contains(',')
                thousandsRegExp = '^[-/+]*\d+?(\,\d{3})*\.{0,1}\d*$';
                if isempty(regexp(numbers, thousandsRegExp, 'once'))
                    numbers = NaN;
                    invalidThousandsSeparator = true;
                end
            end
            % Convert numeric text to numbers.
            if ~invalidThousandsSeparator
                numbers = textscan(char(strrep(numbers, ',', '')), '%f');
                numericData(row, col) = numbers{1};
                raw{row, col} = numbers{1};
            end
        catch
            raw{row, col} = rawData{row};
        end
    end
end


%% Replace non-numeric cells with NaN
R = cellfun(@(x) ~isnumeric(x) && ~islogical(x),raw); % Find non-numeric cells
raw(R) = {NaN}; % Replace non-numeric cells

%% Create output variable
nuserynumric = cell2mat(raw);
nuserynumric = nuserynumric(2:end,1:end);
end

function [lData, rData]=splitData(data,fea,value)
    lData = data(data(:, fea)<value, :);
    rData = data(data(:, fea)>=value, :);
end

function tree = DecisionTreeTraining(data)
    feaName = ["parents", "has_nurs", "form", "children","housing","finance","social","health"];
    [majorLbl, majorNum] = majorityValue(data(:,end));
    if majorNum == length(data(:, end))
        tree = node("", {}, majorLbl, nan, nan);
    else
        [bestAttr, bestThre] = chooseAttribute(data);
        [lData, rData] = splitData(data, bestAttr, bestThre);
        lSubtree = DecisionTreeTraining(lData);
        rSubtree = DecisionTreeTraining(rData);
        op = strcat(feaName(bestAttr), '<', num2str(bestThre));
        tree = node(op, {lSubtree rSubtree}, nan, bestAttr, bestThre);
    end
end

function tree = node(op, kids, prediction, attribute, threshold)
    tree.op = op;
    tree.kids = kids;
    tree.prediction = prediction;
    tree.attribute = attribute;
    tree.threshold = threshold;
end

function [majorLbl, majorNum] = majorityValue(lbls)
    uniqLbls = unique(lbls);
    majorLbl = 0;
    majorNum = 0;
    for i=1:size(uniqLbls, 1)
        nLbl = sum(lbls(:, 1) == uniqLbls(i));
        if majorNum < nLbl
            majorLbl = uniqLbls(i);
            majorNum = nLbl;
        end
    end
end

function [bestAtr, bestThld] = chooseAttribute(data)
    maxGain = 0;
    ftrs = data(:,1:end-1);
    lbls = data(:,end);
    baseEnt = entropy(lbls);
    for i=1:size(ftrs, 2)
        thlds = getThlds(ftrs(:, i));
        for j=1:size(thlds, 1)
            [eltData, gtData] = splitData(data, i, thlds(j, 1));
            newEnt = size(eltData, 1) / size(data, 1) * entropy(eltData(:, end))...
               + size(gtData, 1) / size(data, 1) * entropy(gtData(:, end));
            gain = baseEnt - newEnt;
            if gain > maxGain
                maxGain = gain;
                bestAtr = i;
                bestThld = thlds(j, 1);
            end
        end
    end
end

function thlds = getThlds(ftr)
    thlds = [];
    sortedFtr = sortrows(ftr);
    for i=1:size(ftr, 1)-1
        if sortedFtr(i) ~= sortedFtr(i+1)
            if isempty(thlds)
                thlds(end+1) = (sortedFtr(i) + sortedFtr(i+1)) / 2;
            else
                if (sortedFtr(i) + sortedFtr(i+1)) / 2 ~= thlds(end)
                    thlds(end+1) = (sortedFtr(i) + sortedFtr(i+1)) / 2;
                end
            end
        end
    end
end     

function ent = entropy(lbls)
    nRows = size(lbls, 1);
    uniqLbls = unique(lbls);
    ent = 0;
    for i=1:size(uniqLbls, 1)
        nLbl = sum(lbls(:, 1) == uniqLbls(i));
        prob = nLbl / nRows;
        ent = ent - prob * log2(prob);
    end
end

function predictions = predict(tree, nursery)
    predictions = zeros(length(nursery), 1);
    for i=1:length(nursery)
        predictions(i) = predictOnce(tree, nursery(i,1: end-1));
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
     hit = 0;
     for i = 1:length(predictions)
         if predictions(i)==realLabels(i)
             hit = hit+1;
         end
     end
     eval=hit/length(realLabels);
end

function recallrate = recall(predictions, realLabels)
     pre_true = 0;
     real_true = 0;
     for i = 1:length(predictions)
         if predictions(i) == 1 && predictions(i)==realLabels(i)
             pre_true = pre_true + 1;
         end
     end
     for i = 1:length(realLabels)
         if realLabels(i) == 1 
             real_true = real_true + 1;
         end
     end
     recallrate = pre_true/real_true;
end