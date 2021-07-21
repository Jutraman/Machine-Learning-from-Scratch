function [bestacc,bestc,bestg] = polySVMcgForClass(nursery,~,cmin,cmax,gmin,gmax,v,cstep,gstep,accstep)
%SVMcg cross validation by faruto

%%
% by faruto
%Email:patrick.lee@foxmail.com QQ:516667408 http://blog.sina.com.cn/faruto BNU
%last modified 2010.01.17

%% 若转载请注明：
% faruto and liyang , LIBSVM-farutoUltimateVersion
% a toolbox with implements for support vector machines based on libsvm, 2009.
%
% Chih-Chung Chang and Chih-Jen Lin, LIBSVM : a library for
% support vector machines, 2001. Software available at
% http://www.csie.ntu.edu.tw/~cjlin/libsvm

%% about the parameters of SVMcg
if nargin < 10
    accstep = 4.5;
end
if nargin < 8
    cstep = 0.8;
    gstep = 1;
end
if nargin < 7
    v = 5;
end
if nargin < 5
    gmax = 4;
    gmin = 3;
end
if nargin < 3
    cmax = 6;
    cmin = -6;
end
%% X:c Y:g cg:CVaccuracy
[X,Y] = meshgrid(cmin:cstep:cmax,gmin:gstep:gmax);
[m,n] = size(X);
cg = zeros(m,n);

eps = 10^(-4);

%% record acc with different c & g,and find the bestacc with the smallest c
bestc = 1;
bestg = 0.1;
bestacc = 0;
basenum = 2;
basenum2 = 0;
acc_mean = 0;
count = 0;
for i = 1:m
    if count == 30
        break
    end
    for j = 1:n
        %cmd = ['-v ',num2str(v),' -c ',num2str( basenum^X(i,j) ),' -g ',num2str( basenum^Y(i,j) )];
        for k = 1:v
        [trainD10, validD] = getTenFoldData(nursery, k);
        model = fitcsvm(trainD10(:,1:8), trainD10(:,9),'KernelFunction','Polynomial', 'BoxConstraint',basenum^X(i,j),'PolynomialOrder',basenum2+Y(i,j));
        [label,~]=predict(model,validD(:,1:8));
        real_label = validD(:,9);
        acc_mean = (acc_mean*(k-1) + erreval(label,real_label))/k;
        end
        cg(i,j) = acc_mean;
        
        %if cg(i,j) <= 55
            %continue;
        %end
        
        if cg(i,j) > bestacc
            bestacc = cg(i,j);
            bestc = basenum^X(i,j);
            bestg = basenum^Y(i,j);
        end        
        
        if abs( cg(i,j)-bestacc )<=eps && bestc > basenum^X(i,j)
            bestacc = cg(i,j);
            bestc = basenum^X(i,j);
            bestg = basenum^Y(i,j);
        end        
        count = count + 1;
        disp(count)
    end
end
%% to draw the acc with different c & g
figure;
[C,h] = contour(X,Y,cg,70:accstep:100);
clabel(C,h,'Color','r');
xlabel('log2c','FontSize',12);
ylabel('log2g','FontSize',12);
firstline = 'SVC参数选择结果图(等高线图)[GridSearchMethod]';
secondline = ['Best c=',num2str(bestc),' g=',num2str(bestg), ...
    ' CVAccuracy=',num2str(bestacc),'%'];
title({firstline;secondline},'Fontsize',12);
grid on;

figure;
meshc(X,Y,cg);
% mesh(X,Y,cg);
% surf(X,Y,cg);
axis([cmin,cmax,gmin,gmax,30,100]);
xlabel('log2c','FontSize',12);
ylabel('log2g','FontSize',12);
zlabel('Accuracy(%)','FontSize',12);
firstline = 'SVC参数选择结果图(3D视图)[GridSearchMethod]';
secondline = ['Best c=',num2str(bestc),' g=',num2str(bestg), ...
    ' CVAccuracy=',num2str(bestacc),'%'];
title({firstline;secondline},'Fontsize',12);