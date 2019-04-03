close all
clear
clc
addpath(genpath(cd))
%% load the data
% load winedata.mat
load breast-cancer-wisconsin
% load ionosphere
% load Parliment1984
% load heartdata
load lymphography
%%
% preprocess data to remove Nan entries
for ii=1:size(Tdata,2)
    nanindex=isnan(Tdata(:,ii));
    Tdata(nanindex,:)=[];
end
labels=Tdata(:,end);                  %classes
attributesData=Tdata(:,1:end-1);      %wine data
% for ii=1:size(attributesData,2)       %normalize the data
%     attributesData(:,ii)=normalize(attributesData(:,ii));
% end
[rows,colms]=size(attributesData);  %size of data    
%% seprate the data into training and testing
[trainIdx,~,testIdx]=dividerand(rows,0.8,0,0.2);
trainData=attributesData(trainIdx,:);   %training data
testData=attributesData(testIdx,:);     %testing data
trainlabel=labels(trainIdx);            %training labels
testlabel=labels(testIdx);              %testing labels
%% KNN classification
Mdl = fitcknn(trainData,trainlabel,'NumNeighbors',5,'Standardize',1);
predictedLables_KNN=predict(Mdl,testData);
cp=classperf(testlabel,predictedLables_KNN);
err=cp.ErrorRate;
accuracy=cp.CorrectRate;
%% SA optimisation for feature selection
dim=size(attributesData,2);
lb=0;ub=1;
x0=round(rand(1,dim));
fun=@(x) objfun(x,trainData,testData,trainlabel,testlabel,dim);
options = optimoptions(@simulannealbnd,'MaxIterations',150,...
            'PlotFcn','saplotbestf');
[x,fval,exitflag,output]  = simulannealbnd(fun,x0,zeros(1,dim),ones(1,dim),options) ;
Target_pos_SA=round(x);
% final evaluation for GOA tuned selected features
[error_SA,accuracy_SA,predictedLables_SA]=finalEval(Target_pos_SA,trainData,testData,...
                                                                   trainlabel,testlabel);
%% GOA optimisation for feature selection
SearchAgents_no=10; % Number of search agents
Max_iteration=100; % Maximum numbef of iterations
[Target_score,Target_pos,GOA_cg_curve, Trajectories,fitness_history,...
          position_history]=binaryGOA(SearchAgents_no,Max_iteration,lb,ub,dim,...
                                            trainData,testData,trainlabel,testlabel);
% final evaluation for GOA tuned selected features
[error_GOA,accuracy_GOA,predictedLables_GOA]=finalEval(Target_pos,trainData,testData,trainlabel,testlabel);                                                               

%%
% plot for Predicted classes
figure
plot(testlabel,'s','LineWidth',1,'MarkerSize',12)
hold on
plot(predictedLables_KNN,'o','LineWidth',1,'MarkerSize',6)
hold on
plot(predictedLables_GOA,'x','LineWidth',1,'MarkerSize',6)
hold on
plot(predictedLables_SA,'^','LineWidth',1,'MarkerSize',6)
% hold on
% plot(predictedLables,'.','LineWidth',1,'MarkerSize',3)
legend('Original Labels','Predicted by All','Predcited by GOA Tuned',...          
                                 'Predcited by SA Tuned','Location','best')
title('Output Label comparison of testing Data')
xlabel('-->No of test points')
ylabel('Test Data Labels' )
axis tight

% pie chart for accuracy corresponding to number of features
figure
subplot(1,2,1)
labels={num2str(size(testData,2)),num2str(numel(find(Target_pos))),...
                                      num2str(numel(find(Target_pos_SA)))};

pie([(size(testData,2)),numel(find(Target_pos)),numel(find(Target_pos_SA))],labels)
title('Number of features selected')
legendlabels={'Total Features','Features after GOA Selection',...
                                                    'Features after SA Selection'};
legend(legendlabels,'Location','southoutside','Orientation','vertical')

subplot(1,2,2)
labels={num2str(accuracy*100),num2str(accuracy_GOA*100),num2str(accuracy_SA*100)};
pie([accuracy,accuracy_GOA,accuracy_SA].*100,labels)                                                        
title('Accuracy for features selected')
legendlabels={'Total Features','Features after GOA Selection',...
                                                       'Features after SA Selection'};
legend(legendlabels,'Location','southoutside','Orientation','vertical')
               