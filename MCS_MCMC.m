%This function introduce the replacement of Levy Flight in MCS with MCMC random walk

function [p,F,Fg,Fw,pg,evalHist,diversity,hsl_it] = MCS_MCMC(K, NestI, S, vardef)

%% ----  Inputs  ----
% K = total number of generations 
% NestI = initial positions of nests
% vardef gives the upper and lower bounds of the particle position
% size(vardef) = (2,No dimensions)  vardef(1,:) = upper bounds, vardef(2,:) = lower bound.

% S is a structure containing the parameters for MCS
%  S.pa        %Fraction of eggs discarded each generation
%  S.A         %Maximum distance a cuckoo can travel in one step as
%  fraction of search space diagonal
%  S.maxstep   %Maximum number of steps to take in a levy flight
%  S.plot      %If you want the results plotted set this to 1
%  S.fname     %The function name or handle, if this function gives a complex value the optimser considers it out of bounds
%  S.constrain %Set to 1 if you want to constrain the search inside the
%  bounds defined by vardef or zero if you want unconstrained


%% ----- Outputs -----
% p = time history of nest position (as a cell array)
% F = time history of objective function value of each nest (as a cell
% array)
% pg = optimum position found

%%-----Coefficients-----
pa =S.pa;       %Fraction of nests replaced each generation
ptop = 1-pa;    %Fraction of the best nests use in the enforced search
f = S.fname;    %Function name


%%Find number of dimensions and number of nests
[NoNests,NoDim] = size(NestI);


%%Scale the coordinates between 1 and 0 based on vardef
NormFact = zeros(1,NoDim);
MinFact = zeros(1,NoDim);
for i=1:NoDim
    NormFact(1,i) = vardef(1,i)-vardef(2,i);
    if eq(NormFact(1,i),0)
        NormFact(1,i) = 0;
    end
    MinFact(1,i) = vardef(2,i);
    NestI(:,i) = (NestI(:,i)-MinFact(1,i))./NormFact(1,i);
end


MaxNoSteps = S.maxstep; %Mean number of steps to take in the random walk
A = (sqrt(NoDim)/MaxNoSteps)*S.A;   %Max Step size calculated  

%%-----Constants-----
%Number of nests to discard each generation and number of nests used in
%enforcement
NoDiscard = round(NoNests*pa);
NoTop = round(NoNests*ptop);

% Allocate cells to hold the time history of nest position and objective
% function value
p = cell(K);
F = cell(K);

TT = cell(K);  % testing track
VV = cell(K);

diversity = zeros(K,1);						
evalHist = zeros(K,1);

%% Allocate matrices for current nest position and fitness
pi = zeros(NoNests,NoDim);
Fi = zeros(NoNests,1);
ptemp = zeros(1,NoDim);


%%1 -----Calculate fitness for initial nests-----
for i = 1:NoNests
    pi(i,:) = NestI(i,:);
    if or(and(all(le(pi(i,:),1)),all(ge(pi(i,:),0))),eq(S.constrain,0))
        pix = (pi(i,:).*NormFact)+MinFact;
        Ftemp = feval(f,pix);
    else
        Ftemp = sqrt(-1);
    end
    if isreal(Ftemp)
        Fi(i,1) = Ftemp;
    else
        %Set fitness to zero since co-ordinates are invalid
        Fi(i,1) = realmax;
    end
end


%Record values in cells
p{1} = pi;
F{1} = Fi;


%Calculate diversity
Ldiag = 0;							

for ii=1:NoDim										%
    pmean(ii) = mean(pi(:,ii));
end

for ii=1:NoNests
    
    distT = norm(pi(ii,:)-pmean);
    Ldiag = max(Ldiag,distT);
    
    diversity(1,1) = diversity(1,1) + distT;
end


diversity(1,1) = diversity(1,1)/(Ldiag*NoNests);	

%-----Plotting statement-----
if eq(S.plot,1)
    %Plot positions
    FPlot(1,1) = min(Fi);
    figure(1)
    clf
    subplot(2,1,1)
    plot(pi(:,1),pi(:,2),'o')
    set(gca,'XLim',[0 1],'YLim',[0 1])
    subplot(2,1,2)
    plot(FPlot(1,1),'-+r')
    refreshdata
    drawnow
end

pa = S.pa;               
ptop = 1-pa;                           
hsl_it=[];

%Now the main loop, itterate over all generations
for G = 2:K
    
    
    %a) sort the current nests in order of fitness
    
    %First put vectors into matrix form
    piFi = [pi Fi];
    %Sort by Fi in assending order
    piFiS = sortrows(piFi,NoDim+1);
    
    pi = piFiS(:,1:NoDim);
    Fi = piFiS(:,NoDim+1);
	
    %2 - Loop over each Cuckoo which has been discarded
    for i = 1:NoDiscard
        a = A/(G^(1/2)); %Larger step for exploration walks
        %Using Levy flight
        %a) Random walk
        NoSteps = round(exprnd(MaxNoSteps,1,1));
        NoSteps = min(MaxNoSteps,NoSteps);
        
        dx = levi_walk(NoSteps,NoDim);
        
        for j = 1:NoDim
            ptemp(1,j) = a*dx(1,j) + pi(NoNests - i + 1, j);
        end
        
        %Check position is inside bounds
        if or(and(all(le(ptemp,1)),all(ge(ptemp,0))),eq(S.constrain,0))
            ptempx = (ptemp.*NormFact)+MinFact;
            Ftemp = feval(f,ptempx);
        else
            Ftemp = sqrt(-1);
        end
        
        if isreal(Ftemp)
            %Point valid so update fitness
            %b) Calculate fitness of egg at ptemp
            pi(NoNests - i + 1, :) = ptemp;
            Fi(NoNests - i + 1, 1) = Ftemp;
        else
            %Then point is outside bound so don't continue
        end
        
    end
	
    %3 - Loop over each Cuckoo not to be Discarded
    for C = 1:(NoTop)
        
        %Pick one of the top eggs to cross with
        randNest = round(1 + (NoTop-1).*rand(1));
        if randNest == C
            % Perform random walk instead
            a = A/(G^2); %Smaller step for local walk
             NoSteps = round(exprnd(MaxNoSteps,1,1));
             NoSteps = min(MaxNoSteps,NoSteps);
            
            dx = mcmc_walk(NoSteps,NoDim);
            
            for j = 1:NoDim
               
               
                    ptemp(1,j) = a*dx(1,j) + pi(NoNests - C + 1, j);
                
            end
            
        else
            if Fi(randNest,1)>Fi(C,1)
                %Search in direction of C by golden ratio
                %Calculate distance
                dist(1,:) = pi(C,:) - pi(randNest,:);
                %Multiply distance by a random number
                dist = dist/(0.5*(1+sqrt(5)));
                ptemp = pi(randNest,:) + dist(1,:);
                
            elseif Fi(C,1)>Fi(randNest,1)
                %Search in direction of randNest by golden ratio
                %Calculate distance
                dist(1,:) = pi(randNest,:) - pi(C,:);
                %Multiply distance by a random number
                dist = dist/(0.5*(1+sqrt(5)));
                ptemp = pi(C,:) + dist(1,:);
                
            else
                %Fitnesses are the same search half way
                
                dist(1,:) = pi(randNest,:) - pi(C,:);
                %Multiply distance by a random number
                dist = dist*0.5;
                ptemp = pi(C,:) + dist(1,:);
            end
            
        end
        
        %Check position is inside bounds
        if or(and(all(le(ptemp,1)),all(ge(ptemp,0))),eq(S.constrain,0))
            ptempx = (ptemp.*NormFact)+MinFact;
            Ftemp = feval(f,ptempx);
        else
            Ftemp = sqrt(-1);
        end
        
        
        if isreal(Ftemp)
            %Point valid so update fitness
            
            %c) Select random nest and replace/update position if fitness is
            %better
            
            %Select random index
            randNest = round(1 + (NoNests-1).*rand(1));
            
            if Fi(randNest,1)>Ftemp
                %Then replace egg
                pi(randNest,:) = ptemp;
                Fi(randNest,1) = Ftemp;
            else
                %Discard new egg
            end
            
        else
            %Then point is outside bound so don't continue
        end
        
        
    end
    
    %Record values in cells
    p{G} = pi;
    F{G} = Fi;
    if eq(S.plot,1)
        %Plot positions
        
        FPlot(G,1) = min(Fi);
        figure(1)
        clf
        hold on
        subplot(2,1,1)
        plot(pi(:,1),pi(:,2),'o')
        set(gca,'XLim',[0 1],'YLim',[0 1])
        subplot(2,1,2)
        plot(FPlot(1:G,1),'-+r')
        refreshdata
        drawnow
    end
    fprintf('iteration :%6d mse: %.6f\n', G,min(Fi));
	
    hsl_it=[hsl_it min(Fi)];
end

%Find best solution
[Fg,ind] = min(Fi);
%% Find worst solution
[Fw,indw] = max(Fi);
pg = (pi(ind,:).*NormFact)+MinFact;

end

function [dx] = mcmc_walk(NoSteps,NoDim,S)
%%-----Initial data-----
% Distributions 
nu= 5;     % Target PDF parameter
p= @(t) (t.^(-nu/2-1)).*(exp(-4./(2*t)));   %Target "PDF"

%-----Parameters-----
N = NoDim;    % Number of samples (iterations)
dx = unifrnd(0,1,1,N);  % Samples drawn form the target "PDF"


%%-----Procedure-----
     theta_ast = unifrnd(0,1,1,N);       % Sampling from the proposal PDF
     alpha     = p(theta_ast)./p(dx);  % Ratio of the density at theta_ast and theta points
     ind=find(rand(1,N)<=min(alpha,1));
    dx(ind) = theta_ast(ind);
    

end

