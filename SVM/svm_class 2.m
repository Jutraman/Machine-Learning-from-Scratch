nursery = importfile('nursery_numric.txt',1,12960);
nursery = nursery(randperm(length(nursery)),:);
trainD = nursery(1:2000, :);
testD = nursery(9001:12959, :);

%Mdl1 = fitcsvm(trainD(:,1:8),trainD(:,9), 'KernelFunction','Polynomial', 'OptimizeHyperparameters',{'BoxConstraint','PolynomialOrder'},  'HyperparameterOptimizationOptions',struct('ShowPlots',false));
%Mdl = fitcsvm(trainD(:,1:8),trainD(:,9), 'KernelFunction','gaussian', 'BoxConstraint',16,'KernelScale',1.7411); 
%Mdl = fitcsvm(trainD(:,1:8),trainD(:,9), 'KernelFunction','Polynomial','BoxConstraint',0.0156,'PolynomialOrder',4);
% [label,~]=predict(Mdl,testD(:,1:8));
% real_label = testD(:,9);
% acc = erreval(label,real_label);
% recallrate = recall(label,real_label);
% fprintf('acc = %d , recall = %d \n',acc,recallrate);
lst=linspace(1.005,3,20);
lst2=linspace(1.005,4,4);

%[bestacc,bestc,bestg] = SVMcgForClass(nursery);
%[bestacc,bestc,bestg] = polySVMcgForClass(nursery);
%fprintf('bestacc = %d, bestc = %d, bestg = %d',bestacc,bestc,bestg);

fprintf(1, "10fold go!\n")
for i=1:10
    [trainD10, validD] = getTenFoldData(nursery, i);
    Mdl = fitcsvm(trainD10(:,1:8),trainD10(:,9),'KernelFunction','linear', 'BoxConstraint',1);
    trainerr=erreval(predict(Mdl, trainD10(:,1:8)), trainD10(:, end));
    validerr=erreval(predict(Mdl, validD(:,1:8)), validD(:, end));
    train_recallrate=recall(predict(Mdl, trainD10(:,1:8)), trainD10(:, end));
    valid_recallrate=recall(predict(Mdl, validD(:,1:8)), validD(:, end));
    fprintf('It is %d time\n',i)
    fprintf(1, strcat("traning accuracy:", num2str(trainerr),...
                 "\nvalid accuracy:", num2str(validerr),... 
                 "\ntrain_recall_rate:", num2str(train_recallrate),... 
                "\ntest_recall_rate:", num2str(valid_recallrate), "\n\n"));
end
[label,~]=predict(Mdl,testD(:,1:8));
real_label = testD(:,9);
acc = erreval(label,real_label);
recallrate = recall(label,real_label);
fprintf('acc = %d , recall = %d \n',acc,recallrate);   

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
