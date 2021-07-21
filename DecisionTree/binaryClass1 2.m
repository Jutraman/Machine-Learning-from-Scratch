%****************************************
%main.m
%****************************************
[~,data] = nursery      %?????
[~,feature] = nursery(:,1:8)   %?????
Node = createTree(data, feature);       %?????
drawTree(Node)                          %?????

%****************************************
%createTree.m
%****************************************
%?????ID3??
%data????
%feature????
function [node] = createTree(data, feature)
    type = mostType(data);  %cell??
    [m, n] = size(data);
    %????node
    %value????????null???????????
    %name???????
    %branch??????
    %children????
    node = struct('value','null','name','null','branch','null','children',{});
    temp_type = data{1, n};
    temp_b = true;
    for i = 1 : m
        if temp_type ~= data{i, n}
            temp_b = false;
        end
    end
    %?????????????node??????
    if temp_b == true
        node(1).value = data(1, n); %cell??
        return;
    end
    %?????????????????????
    if isempty(feature) == 1
        node.value = type;  %cell??
        return;
    end 
    %????????
    feature_bestColumn = bestFeature(data); %???????double??
    best_feature = data(:,feature_bestColumn); %??????cell??
    best_distinct = unique(best_feature); %??????
    best_num = length(best_distinct); %????????
    best_proc = cell(best_num, 2);
    best_proc(:, 1) = best_distinct(:, 1);
    best_proc(:, 2) = num2cell(zeros(best_num, 1));
    %??????????
    for i = 1:best_num
        %?node????bach_node??????data??????best_proc(i, 1)????Dv
        bach_node = struct('value', 'null', 'name', 'null', 'branch', 'null', 'children',{});
        Dv_index = 0;
        for j = 1:m
            if data{j, feature_bestColumn} == best_proc{i, 1}
                Dv_index = Dv_index + 1;
            end
        end
        Dv = cell(Dv_index, n);
        Dv_index2 = 1;
        for j = 1:m
            if best_proc{i, 1} == data{j, feature_bestColumn}
                Dv(Dv_index2, :) = data(j, :);
                Dv_index2 = Dv_index2 + 1;
            end
        end
        Dfeature = feature;
        %Dv?????????????????
        if isempty(Dv) == 1
            bach_node.value = type;
            bach_node.name = feature(feature_bestColumn);
            bach_node.branch = best_proc(i, 1);
            node.children(i) = bach_node;
            return;
        else
            Dfeature(feature_bestColumn) = [];
            Dv(:,feature_bestColumn) = [];
            %????createTree??
            bach_node = createTree(Dv, Dfeature);
            bach_node(1).branch = best_proc(i, 1);
            bach_node(1).name = feature(feature_bestColumn);
            node(1).children(i) = bach_node;
        end
    end
end

%****************************************
%mostType.m
%****************************************
%?????????
function [res] = mostType(data)         %???cell??
    [m,n] = size(data);
    res = data(:, n);
    res_distinct = unique(res);
    res_num = length(res_distinct);
    res_proc = cell(res_num,2);
    res_proc(:, 1) = res_distinct(:, 1);
    res_proc(:, 2) = num2cell(zeros(res_num,1));
    for i = 1:res_num
        for j = 1:m
            if res_proc{i, 1} == data{j, n};
                res_proc{i, 2} = res_proc{i, 2} + 1;
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

%****************************************
%drawTree.m
%****************************************
% ?????
function [] = drawTree(node)
    % ???
    nodeVec = [];
    nodeSpec = {};
    edgeSpec = [];
    [nodeVec,nodeSpec,edgeSpec,~] = travesing(node,0,0,nodeVec,nodeSpec,edgeSpec);
    treeplot(nodeVec);
    [x,y] = treelayout(nodeVec);
    [~,n] = size(nodeVec);
    x = x';
    y = y';
    text(x(:,1),y(:,1),nodeSpec,'FontSize',15,'FontWeight','bold','VerticalAlignment','bottom','HorizontalAlignment','center');
    x_branch = [];
    y_branch = [];
    for i = 2:n
        x_branch = [x_branch; (x(i,1)+x(nodeVec(i),1))/2];
        y_branch = [y_branch; (y(i,1)+y(nodeVec(i),1))/2];
    end
    text(x_branch(:,1),y_branch(:,1),edgeSpec(1,:),'FontSize',12,'Color','blue','FontWeight','bold','VerticalAlignment','bottom','HorizontalAlignment','center');
end

% ???
function [nodeVec,nodeSpec,edgeSpec,current_count] = travesing(node,current_count,last_node,nodeVec,nodeSpec,edgeSpec)
    nodeVec = [nodeVec last_node];
    if isempty(node.value)
        nodeSpec = [nodeSpec node.children(1).name];
    else 
        if strcmp(node.value, '?')
            nodeSpec = [nodeSpec '??'];
        else
            nodeSpec = [nodeSpec '??'];
        end
    end
    edgeSpec = [edgeSpec node.branch];
    current_count = current_count + 1;
    current_node = current_count;
    if ~isempty(node.value)
        return;
    end
    for next_ndoe = node.children
        [nodeVec,nodeSpec,edgeSpec,current_count] = travesing(next_ndoe,current_count,current_node,nodeVec,nodeSpec,edgeSpec);
    end
end