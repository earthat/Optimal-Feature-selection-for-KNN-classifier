function objval=objfun(index,trainigdata ,testingdata,trainiglabels,testinglabels,N)
index=round(index);
index = checkempty(index,N);
newtrainigdata=trainigdata(:,find(index));
newtestingdata=testingdata(:,find(index));
Mdl = fitcknn(newtrainigdata,trainiglabels,'NumNeighbors',5,'Standardize',1);
Y=predict(Mdl,newtestingdata);
cp=classperf(testinglabels,Y);
err=cp.ErrorRate;
R=numel(find(index==1));
alpha=0.7;
beta=1-alpha;
objval=alpha*err+beta*(R/N);