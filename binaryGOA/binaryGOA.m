
% The Grasshopper Optimization Algorithm
function [TargetFitness,TargetPosition,Convergence_curve,Trajectories,...
               fitness_history, position_history]=binaryGOA(N, Max_iter, lb,ub, dim,...
                                              trainData,testData,trainlabel,testlabel)

disp('GOA is now estimating the global optimum for your problem....')
flag=0;
if size(ub,1)==1
    ub=ones(dim,1)*ub;
    lb=ones(dim,1)*lb;
end

if (rem(dim,2)~=0) % this algorithm should be run with a even number of variables. 
    %This line is to handle odd number of variables
    dim = dim+1;
    ub = [ub; 100];
    lb = [lb; -100];
    flag=1;
end

%Initialize the population of grasshoppers

GrassHopperPositions=round(initialization(N,dim,ub,lb));
GrassHopperFitness = zeros(1,N);

fitness_history=zeros(N,Max_iter);
position_history=zeros(N,Max_iter,dim);
Convergence_curve=zeros(1,Max_iter);
Trajectories=zeros(N,Max_iter);

% cMax=1;
% cMin=0.00004;
cMax=2.079;
cMin=0.00004;
%Calculate the fitness of initial grasshoppers

for i=1:size(GrassHopperPositions,1)
    if flag == 1
        GrassHopperPositions(i,1:end-1) = checkempty(GrassHopperPositions(i,1:end-1),dim);
        GrassHopperFitness(1,i)=objfun(GrassHopperPositions(i,1:end-1),...
                                trainData,testData,trainlabel,testlabel,dim);
    else
        GrassHopperPositions(i,:) = checkempty(GrassHopperPositions(i,:),dim);
        GrassHopperFitness(1,i)=objfun(GrassHopperPositions(i,:),...
                                trainData,testData,trainlabel,testlabel,dim);
    end
    fitness_history(i,1)=GrassHopperFitness(1,i);
    position_history(i,1,:)=GrassHopperPositions(i,:);
    Trajectories(:,1)=GrassHopperPositions(:,1);
end

[sorted_fitness,sorted_indexes]=sort(GrassHopperFitness);

% Find the best grasshopper (target) in the first population 
for newindex=1:N
    Sorted_grasshopper(newindex,:)=GrassHopperPositions(sorted_indexes(newindex),:);
end

TargetPosition=Sorted_grasshopper(1,:);
TargetFitness=sorted_fitness(1);

% Main loop
l=2; % Start from the second iteration since the first iteration was 
       %dedicated to calculating the fitness of antlions
while l<Max_iter+1
    
    c=cMax-l*((cMax-cMin)/Max_iter); % Eq. (2.8) in the paper
    
    for i=1:size(GrassHopperPositions,1)
        temp= GrassHopperPositions';
        for k=1:2:dim
            S_i=zeros(2,1);
            for j=1:N
                if i~=j
                    % Calculate the distance between two grasshoppers
                    Dist=distance(temp(k:k+1,j), temp(k:k+1,i)); 
                    
                    r_ij_vec=(temp(k:k+1,j)-temp(k:k+1,i))/(Dist+eps); % xj-xi/dij in Eq. (2.7)
                    xj_xi=2+rem(Dist,2); % |xjd - xid| in Eq. (2.7) 
                    
                    s_ij=((ub(k:k+1) - lb(k:k+1))*c/2)*S_func(xj_xi).*r_ij_vec; % The first part inside the big bracket in Eq. (2.7)
                    S_i=S_i+s_ij;
                end
            end
            S_i_total(k:k+1, :) = S_i;
            
        end
        
        deltaX= c * S_i_total'+ (TargetPosition); % Eq. (2.7) in the paper
        for tt=1:size(deltaX,2)
            T_deltaX(tt)=1/(1+exp(-deltaX(tt)));
            if rand<T_deltaX(tt)
                X_new(tt) =1;
            else
                X_new(tt)=0;
            end
        end
        GrassHopperPositions_temp(i,:)=X_new'; 
    end
    % GrassHopperPositions
    GrassHopperPositions=(GrassHopperPositions_temp);
    
    for i=1:size(GrassHopperPositions,1)
        
        % Calculating the objective values for all grasshoppers
        if flag == 1
            GrassHopperPositions(i,1:end-1) = checkempty(GrassHopperPositions(i,1:end-1),dim);
            GrassHopperFitness(1,i)=objfun(GrassHopperPositions(i,1:end-1),...
                                trainData,testData,trainlabel,testlabel,dim);
        else
            GrassHopperPositions(i,:) = checkempty(GrassHopperPositions(i,:),dim);
            GrassHopperFitness(1,i)=objfun(GrassHopperPositions(i,:),...
                                trainData,testData,trainlabel,testlabel,dim);
        end
        fitness_history(i,l)=GrassHopperFitness(1,i);
        position_history(i,l,:)=GrassHopperPositions(i,:);
        
        Trajectories(:,l)=GrassHopperPositions(:,1);
        
        % Update the target
        if GrassHopperFitness(1,i)<TargetFitness
            TargetPosition=GrassHopperPositions(i,:);
            TargetFitness=GrassHopperFitness(1,i);
        end
    end
        
    Convergence_curve(l)=TargetFitness;
    disp(['In GOA iteration #', num2str(l), ' , target''s objective = ', num2str(TargetFitness)])
    
    l = l + 1;
end


if (flag==1)
    TargetPosition = TargetPosition(1:dim-1);
end


