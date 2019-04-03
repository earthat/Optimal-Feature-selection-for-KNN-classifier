function [error,accuracy,Y]=finalEval(index,trainigdata ,testingdata,trainiglabels,testinglabels)
newtrainigdata=trainigdata(:,find(index));
newtestingdata=testingdata(:,find(index));
Mdl = fitcknn(newtrainigdata,trainiglabels,'NumNeighbors',5,'Standardize',1);
Y=predict(Mdl,newtestingdata);
cp=classperf(testinglabels,Y);
error=cp.ErrorRate;
accuracy=cp.CorrectRate;