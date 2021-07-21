import DrawDecisionTree.*

nursery = importfile('nursery_numric4.txt',1,12960);
trainD = nursery(1:9000, :);
testD = nursery(9001:12960, :);
tree = DecisionTreeTraining(trainD);
DrawDecisionTree(tree);

trainerr=erreval(predict(tree, trainD), trainD(:, end));
testerr=erreval(predict(tree, testD), testD(:, end));
fprintf(1, strcat("traning rmse:", num2str(trainerr),...
                "testing rmse", num2str(testerr), "\n"));
            
fprintf(1, "10fold go!\n")
for i=1:10

    [trainD10, validD] = getTenFoldData(nursery(1:2000, :), i);
    tree1 = DecisionTreeTraining(trainD10);
    DrawDecisionTree(tree1);
    trainerr=erreval(predict(tree, trainD10), trainD10(:, end));
    validerr=erreval(predict(tree, validD), validD(:, end));
    fprintf(1, strcat("traning rmse:", num2str(trainerr),...
                "testing rmse", num2str(validerr), "\n"));
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
end


function [trainData, validData] = getTenFoldData(nursery, i)
    foldSize = 20;
    trainData = [];
    for j=1:10
        if j==i
            validData = nursery(foldSize*(j-1)+1:foldSize*j,:);
        else
            trainData = [trainData; nursery(foldSize*(j-1)+1:foldSize*j,:)];
        end
    end
end    

function err=error(dataset)
    err = sum((dataset(:,end)-mean(dataset(:,end))).^2);
end

function [lData, rData]=splitData(nursery,fea,value)
    lData = nursery(nursery(:, fea)<value, :);
    rData = nursery(nursery(:, fea)>=value, :);
end

function tree = DecisionTreeTraining(nursery)
    %feaName = ["AT", "V", "AP", "RH"];
    feaName = ["parents", "has_nurs", "form", "children","housing","finance","social","health"];
    min_sample = 20;
    if length(nursery) <= min_sample
        tree = node("", {}, mean(nursery(:,end)), nan, nan);
    else
        %[bestAttr, bestThre] = chooseAttribute(nursery);
        bestAttr=bestFeature(nursery);
        %bestAttr=column;
        disp(bestAttr);
        bestThre=nursery(:,bestAttr);
        
        [lData, rData] = splitData(nursery, bestAttr, bestThre);
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
   
function [bestAttr, bestThre] = chooseAttribute(nursery)
    besterr=inf;
    bestAttr=0;
    %[bestAttr] = bestFeature(nursery)
    bestThre=0;
    for fea=1:8
        for valueIdx = 1:size(nursery,1)
            value=nursery(valueIdx,fea);
            [set1,set2]=splitData(nursery,fea,value);
            if isempty(set1)
                continue;
            end
            now_err=error(set1)+error(set2);
            %[now_err]=getEntropy(nursery)
            if now_err>besterr
                besterr=now_err;
                bestAttr=fea;
                bestThre=value;
            end
        end
    end
end

%****************************************
%getEntropy.m
%****************************************
%?????
function [entropy] = getEntropy(data)       %???double??
    entropy = 0;
    [m,n] = size(data);
    label = data(:, n);
    label_distinct = unique(label);
    label_num = length(label_distinct);
    proc = cell(label_num,2);
    proc(:, 1) = label_distinct(:, 1);
    proc(:, 2) = num2cell(zeros(label_num, 1));
     for i = 1:label_num
        for j = 1:m
            if proc{i, 1} == data{j, n}
                proc{i, 2} = proc{i, 2} + 1;
            end
        end
        proc{i, 2} = proc{i, 2} / m;
    end
    for i = 1:label_num
        entropy = entropy - proc{i, 2} * log2(proc{i, 2});
    end
end

%****************************************
%getGain.m
%****************************************
%??????
function [gain] = getGain(entropy, data, column)        %???double??
    [m,n] = size(data);
    feature = data(:, column);
    feature_distinct = unique(feature);
    feature_num = length(feature_distinct);
    feature_proc = cell(feature_num, 2);
    feature_proc(:, 1) = feature_distinct(:, 1);
    feature_proc(:, 2) = num2cell(zeros(feature_num, 1));
    f_entropy = 0;
     for i = 1:feature_num
       feature_row = 0;
       for j = 1:m
           if feature_proc{i, 1} == data{j, column}
               feature_proc{i, 2} = feature_proc{i, 2} + 1;
               feature_row = feature_row + 1;
           end
       end
       feature_data = cell(feature_row,n);
       feature_row = 1;
       for j = 1:m
           if feature_distinct{i, 1} == data{j, column}
               feature_data(feature_row, :) = data(j, :);
               feature_row = feature_row + 1;
           end
       end
       f_entropy = f_entropy + feature_proc{i, 2} / m * getEntropy(feature_data);
    end
    gain = entropy - f_entropy;
end

%****************************************
%bestFeature.m
%****************************************
%????????
function [column] = bestFeature(data)       %???double??
    [~,n] = size(data);
    featureSize = n - 1;
    gain_proc = cell(featureSize, 2);
    entropy = getEntropy(data);
    for i = 1:featureSize
        gain_proc{i, 1} = i;
        gain_proc{i, 2} = getGain(entropy, data, i);
    end
    max = gain_proc{1,2};
    max_label = 1;
    for i = 1:featureSize
        if gain_proc{i, 2} >= max
            max = gain_proc{i, 2};
            max_label = i;
        end
    end
    column = max_label;
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
    eval=sqrt(mean((predictions-realLabels).^2));
end